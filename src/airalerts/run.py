"""Single entry point: download -> clean -> describe -> backtest -> outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from airalerts import data


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


if __name__ == "__main__":
    main()
