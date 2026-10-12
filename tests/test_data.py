from pathlib import Path

import pandas as pd
import pytest

from airalerts import data

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def utc(*stamps: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(s, tz="UTC") for s in stamps]


def test_local_to_utc_applies_winter_and_summer_offsets():
    local = pd.Series(pd.to_datetime(["2023-01-15 12:00:00", "2023-07-15 12:00:00"]))
    out, stats = data.local_to_utc(local)
    assert list(out) == utc("2023-01-15 10:00:00", "2023-07-15 09:00:00")
    assert stats == {"ambiguous_dst": 0, "nonexistent_dst": 0}


def test_local_to_utc_keeps_the_repeated_autumn_hour_in_order():
    # 03:00-03:59 happened twice on 30 Oct 2022. 03:20 after 03:50 must be the second pass.
    local = pd.Series(pd.to_datetime(["2022-10-30 03:10:00", "2022-10-30 03:50:00", "2022-10-30 03:20:00"]))
    out, stats = data.local_to_utc(local)
    assert list(out) == utc("2022-10-30 00:10:00", "2022-10-30 00:50:00", "2022-10-30 01:20:00")
    assert stats["ambiguous_dst"] == 3


def test_local_to_utc_moves_a_skipped_spring_time_forward():
    local = pd.Series(pd.to_datetime(["2023-03-26 03:30:00"]))
    out, stats = data.local_to_utc(local)
    assert list(out) == utc("2023-03-26 01:00:00")
    assert stats["nonexistent_dst"] == 1


def history(rows: list[tuple[int, list[str], str]]) -> pd.DataFrame:
    return pd.DataFrame({"state": [r[0] for r in rows], "causes": [r[1] for r in rows], "ts": utc(*[r[2] for r in rows])})


def test_pair_events_uses_latest_start_and_drops_repeats():
    events, stats = data.pair_events(
        history(
            [
                (1, [], "2022-03-01 05:00:00"),
                (1, [], "2022-03-01 06:00:00"),
                (0, [], "2022-03-01 07:00:00"),
                (0, [], "2022-03-01 08:00:00"),
                (1, ["drone"], "2026-10-01 09:00:00"),
                (0, [], "2026-10-01 09:30:00"),
                (1, ["drone", "missile"], "2026-10-01 10:00:00"),
                (0, [], "2026-10-01 11:00:00"),
            ]
        )
    )
    assert list(events["start"]) == utc("2022-03-01 06:00:00", "2026-10-01 09:00:00", "2026-10-01 10:00:00")
    assert list(events["end"]) == utc("2022-03-01 07:00:00", "2026-10-01 09:30:00", "2026-10-01 11:00:00")
    assert events["threat"].fillna("none").tolist() == ["none", "drone", "missile"]
    assert stats == {"repeated_starts_dropped": 1, "repeated_clears_dropped": 1}


def test_pair_events_keeps_a_running_alert_open():
    events, _ = data.pair_events(
        history([(1, [], "2026-10-01 09:00:00"), (0, [], "2026-10-01 09:30:00"), (1, ["drone"], "2026-10-01 10:00:00")])
    )
    assert len(events) == 2
    assert pd.isna(events["end"].iloc[1])
    assert pd.isna(events["threat"].iloc[1])


def test_truncate_hides_everything_after_the_cutoff():
    events, _ = data.pair_events(
        history(
            [
                (1, ["drone"], "2026-10-01 09:00:00"),
                (0, [], "2026-10-01 09:30:00"),
                (1, ["missile"], "2026-10-01 10:00:00"),
                (0, [], "2026-10-01 11:00:00"),
                (1, ["drone"], "2026-10-01 12:00:00"),
                (0, [], "2026-10-01 12:30:00"),
            ]
        )
    )
    cut = data.truncate(events, pd.Timestamp("2026-10-01 10:30:00", tz="UTC"))
    assert len(cut) == 2
    assert pd.isna(cut["end"].iloc[1])
    # The running alert's causes may still change, so they are not exposed.
    assert cut["causes"].iloc[1] == "" and pd.isna(cut["threat"].iloc[1])


@pytest.mark.skipif(not (RAW / data.HISTORY_FILE).exists(), reason="raw data not downloaded")
def test_api_events_match_the_published_file():
    events, _ = data.load_events(RAW)
    check = data.compare_with_open_data(events, RAW)
    assert check["matched_start_and_end"] == check["published_records"]
    assert check["published_only"] == 0 and check["api_only"] == 0
