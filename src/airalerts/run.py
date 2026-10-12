"""Single entry point: download -> clean -> describe -> backtest -> outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from airalerts import data, describe, evaluate, grid
from airalerts.model import ModelSpec, coefficients
from airalerts.report import plot_reliability, plot_skill

HORIZONS = (1, 3, 6)
MODELS = [
    # First attempt: plain logistic regression on every earlier origin.
    ModelSpec("logit_all_history", train_weeks=None),
    # Final: recent data only, with the trailing climatology as an input so the base rate can move.
    ModelSpec("logit_16w", train_weeks=16, reference_logits=("hour_28d",)),
    # Challenger as first run: boosted trees on the same inputs inside a sigmoid calibration wrapper.
    ModelSpec("boosting_16w", train_weeks=16, reference_logits=("hour_28d",), estimator="boosting"),
    # The same trees without the wrapper, added after a review traced most of the failure to it.
    ModelSpec("boosting_16w_uncalibrated", 16, ("hour_28d",), estimator="boosting_uncalibrated"),
]
FINAL = "logit_16w"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-download", action="store_true", help="reuse files already in --raw-dir")
    parser.add_argument("--as-of", help="UTC cutoff, e.g. 2026-10-12T18:00:00Z; nothing after it is used")
    parser.add_argument("--sensitivity", action="store_true", help="also vary the training window and inputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_download:
        data.download(args.raw_dir)
    as_of = pd.Timestamp(args.as_of).tz_convert("UTC") if args.as_of else None
    events, info = data.load_events(args.raw_dir, as_of)
    info["published_file_check"] = data.compare_with_open_data(events, args.raw_dir)
    print(json.dumps(info, indent=2))
    as_of = pd.Timestamp(info["as_of_utc"])

    tables = describe.run(events, as_of, args.out_dir)
    for name in ("regime_summary", "threat_summary"):
        print()
        print(tables[name].to_string(index=False))

    table = grid.make_grid(events, as_of, HORIZONS)
    report, predictions, fitted = evaluate.evaluate(table, HORIZONS, MODELS)
    final_spec = next(spec for spec in MODELS if spec.name == FINAL)
    weights = pd.DataFrame({f"{h}h": coefficients(fitted[h][FINAL], final_spec) for h in HORIZONS})
    weights.to_csv(args.out_dir / "tables" / "final_model_coefficients.csv", index_label="feature")
    report = {"data": info, **report}
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    starts = {"test_period": report["test_start_utc"], "since_reform": data.REFORM.isoformat()}
    for period, start in starts.items():
        print(f"\n{period}: {start} to {report['test_end_utc']}")
        summary = evaluate.summary_table(report, period)
        summary.to_csv(args.out_dir / "tables" / f"results_{period}.csv", index=False)
        print(summary.to_string(index=False))
    print()
    print("already-active share:", {h: report["horizons"][str(h)]["already_active_share"] for h in HORIZONS})
    print("final model, last fold, standardised coefficients")
    print(weights.to_string())
    plot_skill(report, args.out_dir / "figures" / "skill.png")
    plot_reliability(predictions, args.out_dir / "figures" / "reliability.png")

    if args.sensitivity:
        # Run after the headline numbers were fixed; reported as sensitivity, not as results.
        specs = [
            ModelSpec(f"logit_{'all' if weeks is None else f'{weeks}w'}{'_clim' if logits else ''}", weeks, logits)
            for weeks in (8, 16, 32, None)
            for logits in ((), ("hour_28d",))
        ]
        sensitivity, _, _ = evaluate.evaluate(table, HORIZONS, specs)
        rows = pd.concat([evaluate.summary_table(sensitivity, period).assign(period=period) for period in starts])
        rows = rows[rows["forecast"].str.startswith("logit_")]
        rows.to_csv(args.out_dir / "tables" / "sensitivity.csv", index=False)
        print()
        print(rows.pivot_table(index="forecast", columns=["period", "horizon_h"], values="vs_best_simple_%", sort=False).to_string())


if __name__ == "__main__":
    main()
