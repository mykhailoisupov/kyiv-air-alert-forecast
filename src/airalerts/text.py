"""Text features from Air Force channel posts, and the ablation that tests them.

A post counts towards the features at origin t only if it was published
strictly before t. Edited posts keep their timestamp but lose their text,
because the text now shown may have been written after publication.

Usage: python -m airalerts.text --as-of 2026-10-12T18:00:00Z
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from airalerts import data, evaluate, grid
from airalerts.model import ModelSpec
from airalerts.telegram import load_posts

MINUTE = 60
# Fixed before any result was seen. Keyword group -> lookback windows in minutes.
GROUPS = {
    "any": (None, (60, 180)),
    "kyiv": (r"ки[їє]в|столиц", (15, 30, 60, 180)),
    "missile": (r"ракет|баліст|міг-31|ту-95|ту-160|ту-22|зліт|пуск|калібр|кинджал", (60, 180)),
}
SINCE_KYIV = "min_since_kyiv_post"
SINCE_CAP = 24 * 60
TEXT_FEATURES = tuple(f"posts_{name}_{w}m" for name, (_, windows) in GROUPS.items() for w in windows) + (SINCE_KYIV,)

BASE = ModelSpec("logit_16w", train_weeks=16, reference_logits=("hour_28d",))
WITH_TEXT = ModelSpec("logit_16w_text", 16, reference_logits=("hour_28d",), extra_log_features=TEXT_FEATURES)
# Hourly origins repeat the main experiment; 15-minute origins give text its best chance.
EXPERIMENTS = (("hourly origins", "1h", (1, 3, 6)), ("15-minute origins", "15min", (1,)))


def build_text_features(posts: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Post counts in trailing windows that end just before each origin."""
    t_s = grid.epoch_seconds(origins)
    usable_text = posts["text"].where(~posts["edited"], "")
    out = pd.DataFrame(index=origins)
    for name, (pattern, windows) in GROUPS.items():
        matched = posts if pattern is None else posts[usable_text.str.contains(pattern, case=False, regex=True)]
        times = grid.epoch_seconds(matched["published_utc"])
        before = np.searchsorted(times, t_s, side="left")  # published strictly before t
        for w in windows:
            out[f"posts_{name}_{w}m"] = before - np.searchsorted(times, t_s - w * MINUTE, side="left")
        if name == "kyiv":
            last = times[np.maximum(before - 1, 0)] if len(times) else np.zeros_like(t_s)
            out[SINCE_KYIV] = np.where(before > 0, np.minimum((t_s - last) / MINUTE, SINCE_CAP), SINCE_CAP)
    # Origins whose longest window reaches back before the first scraped post are not usable.
    longest = max(w for _, windows in GROUPS.values() for w in windows)
    covered = t_s - longest * MINUTE >= grid.epoch_seconds(posts["published_utc"]).min()
    out.loc[~covered] = np.nan
    return out


def run(events: pd.DataFrame, posts: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    rows, coverage = [], {}
    for label, freq, horizons in EXPERIMENTS:
        table = grid.make_grid(events, as_of, horizons, freq=freq)
        table = table.join(build_text_features(posts, table.index))
        report, predictions, _ = evaluate.evaluate(table, horizons, [BASE, WITH_TEXT])
        coverage[label] = {"test_start_utc": report["test_start_utc"], "test_end_utc": report["test_end_utc"]}
        for h in horizons:
            since_reform = predictions[h][predictions[h].index >= data.REFORM]
            for period, part in (("test_period", predictions[h]), ("since_reform", since_reform)):
                y = part["y"].to_numpy()
                base, text = part[BASE.name].to_numpy(), part[WITH_TEXT.name].to_numpy()
                days = part.index.tz_convert(data.KYIV_TZ).strftime("%Y-%m-%d").to_numpy()
                low, high = evaluate.skill_interval(y, text, base, days)
                rows.append(
                    {
                        "origins": label,
                        "horizon_h": h,
                        "period": period,
                        "rows": len(part),
                        "brier_without_text": round(evaluate.brier(y, base), 5),
                        "brier_with_text": round(evaluate.brier(y, text), 5),
                        "skill_of_text_%": round(100 * (1 - evaluate.brier(y, text) / evaluate.brier(y, base)), 1),
                        "ci95_%": f"{100 * low:+.1f} to {100 * high:+.1f}",
                    }
                )
    return pd.DataFrame(rows), coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--channel", default="kpszsu")
    parser.add_argument("--as-of", help="UTC cutoff; nothing after it is used")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of).tz_convert("UTC") if args.as_of else None
    events, info = data.load_events(args.raw_dir, as_of)
    as_of = pd.Timestamp(info["as_of_utc"])
    posts = load_posts(args.raw_dir / "telegram" / f"{args.channel}.jsonl")
    posts = posts[posts["published_utc"] <= as_of].reset_index(drop=True)
    table, coverage = run(events, posts, as_of)
    summary = {
        "channel": args.channel,
        "posts": int(len(posts)),
        "first_post_utc": posts["published_utc"].min().isoformat(),
        "last_post_utc": posts["published_utc"].max().isoformat(),
        "edited_share": round(float(posts["edited"].mean()), 4),
        "experiments": coverage,
    }
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "tables" / "text_ablation.csv", index=False)
    (args.out_dir / "text_ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
