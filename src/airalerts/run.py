"""Single entry point: download -> clean -> describe -> backtest -> outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from airalerts import data, describe, evaluate, grid
from airalerts.model import ModelSpec

HORIZONS = (1, 3, 6)
MODELS = [
    # First attempt: plain logistic regression on every earlier origin.
    ModelSpec("logit_all_history", train_weeks=None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-download", action="store_true", help="reuse files already in --raw-dir")
    parser.add_argument("--as-of", help="UTC cutoff, e.g. 2026-10-12T18:00:00Z; nothing after it is used")
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
    report, _ = evaluate.evaluate(table, HORIZONS, MODELS)
    report = {"data": info, **report}
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for period in ("test_period", "since_reform"):
        print(f"\n{period}: {report['test_start_utc']} to {report['test_end_utc']}")
        print(evaluate.summary_table(report, period).to_string(index=False))


if __name__ == "__main__":
    main()
