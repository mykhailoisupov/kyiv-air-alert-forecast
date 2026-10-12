"""Live forecast: the chance that a new alert starts in Kyiv within 1, 3 and 6 hours.

Usage: python -m airalerts.now [--as-of 2026-10-12T18:00:00Z] [--no-download]

Everything is computed as of one instant. Alerts that start later are dropped, the end of
an alert still running is hidden, and the model is fitted only on origins whose outcome
was already known at that instant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from airalerts import baselines, data, evaluate, grid
from airalerts.model import FINAL, HORIZONS, design, new_estimator

SIMPLE_NAMES = {
    "const_28d": "flat 28-day rate",
    "hour_28d": "hour-of-day rate, 28 days",
    "hour_all": "hour-of-day rate, all history",
    "markov_28d": "recent-onset rate, 28 days",
}
SELECTION_WEEKS = 8  # the best simple forecast is picked on this many weeks of known outcomes


def best_simple(frame: pd.DataFrame, target: str, as_of: pd.Timestamp) -> str:
    """The simple forecast with the lowest Brier score on recent origins whose outcome is known."""
    recent = frame[(frame.index >= as_of - SELECTION_WEEKS * evaluate.WEEK) & frame[target].notna()]
    y = recent[target].to_numpy()
    return min(evaluate.REFERENCES, key=lambda name: evaluate.brier(y, recent[name].to_numpy()))


def forecast(events: pd.DataFrame, as_of: pd.Timestamp, horizons: tuple[int, ...] = HORIZONS) -> dict:
    """State and onset probabilities at `as_of`, from nothing that happened after it."""
    events = data.truncate(events, as_of)
    hourly = pd.date_range(events["start"].min().ceil("h"), as_of.floor("h"), freq="1h", unit="s")
    origins = hourly.union(pd.DatetimeIndex([as_of]).as_unit("s"))
    table = grid.build_features(events, origins).join(grid.build_labels(events, origins, horizons, as_of))
    now_row = table.loc[as_of]
    out = {
        "as_of_utc": as_of.isoformat(),
        "active": bool(now_row["active"]),
        "minutes_since_start": round(float(now_row["since_start_h"]) * 60, 1),
        "minutes_since_end": round(float(now_row["since_end_h"]) * 60, 1),
        "horizons": {},
    }
    if out["active"]:
        return out  # the onset forecast is defined for a quiet city only
    for h in horizons:
        target = f"onset_{h}h"
        known = table[target].notna()
        frame = table[~table["active"] & (known | (table.index == as_of))].copy()
        # The row for `as_of` has no outcome yet. Its placeholder is never read: an outcome
        # counts towards a reference forecast only once its window has closed.
        frame[target] = frame[target].fillna(0.0)
        frame = frame.join(baselines.reference_forecasts(frame, h))
        frame.loc[as_of, target] = np.nan
        train, _ = evaluate.split(frame, as_of, h, FINAL.train_weeks)
        model = new_estimator(FINAL).fit(design(train, FINAL), train[target].to_numpy())
        simple = best_simple(frame, target, as_of)
        out["horizons"][h] = {
            "model": round(float(model.predict_proba(design(frame.loc[[as_of]], FINAL))[0, 1]), 4),
            "simple": round(float(frame.loc[as_of, simple]), 4),
            "simple_name": simple,
            "train_rows": int(len(train)),
        }
    return out


def minutes_text(minutes: float) -> str:
    hours, rest = divmod(int(round(minutes)), 60)
    return f"{hours} h {rest:02d} min" if hours else f"{rest} min"


def as_text(result: dict) -> str:
    """Plain-text report of one forecast."""
    as_of = pd.Timestamp(result["as_of_utc"]).tz_convert(data.KYIV_TZ)
    lines = [f"{data.REGION} at {as_of:%Y-%m-%d %H:%M} Kyiv time"]
    if result["active"]:
        lines.append(f"An alert has been running for {minutes_text(result['minutes_since_start'])}.")
        lines.append("The chance of a new alert is forecast only while no alert is active.")
        return "\n".join(lines)
    lines.append(f"No alert is active. The last one ended {minutes_text(result['minutes_since_end'])} ago.")
    lines += ["", f"{'New alert within':<18}{'model':>7}{'simple':>9}  best simple forecast"]
    for h, row in result["horizons"].items():
        label = "1 hour" if h == 1 else f"{h} hours"
        lines.append(f"{label:<18}{row['model']:>7.0%}{row['simple']:>9.0%}  {SIMPLE_NAMES[row['simple_name']]}")
    rows = next(iter(result["horizons"].values()))["train_rows"]
    lines += [
        "",
        f"Model: logistic regression fitted on {rows:,} quiet hours from the last {FINAL.train_weeks} weeks.",
        f"Best simple forecast: the simple rule with the lowest Brier score over the last {SELECTION_WEEKS} weeks.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--no-download", action="store_true", help="reuse files already in --raw-dir")
    parser.add_argument("--as-of", help="UTC instant to forecast from, e.g. 2026-10-12T18:00:00Z")
    args = parser.parse_args()
    if not args.no_download:
        data.download(args.raw_dir)
    as_of = pd.Timestamp(args.as_of).tz_convert("UTC") if args.as_of else None
    events, info = data.load_events(args.raw_dir, as_of)
    print(as_text(forecast(events, pd.Timestamp(info["as_of_utc"]))))


if __name__ == "__main__":
    main()
