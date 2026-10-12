"""The remaining-time estimate must be built only from alerts that had ended by its instant."""

import numpy as np
import pandas as pd
from test_leakage import synthetic_events

from airalerts import data, remaining

EVENTS = synthetic_events(1500)


def stamp(clock: str) -> pd.Timestamp:
    return pd.Timestamp(f"2026-10-01 {clock}", tz="UTC")


def alerts(spans: list[tuple[str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": data.REGION,
            "start": pd.DatetimeIndex([stamp(a) for a, _ in spans]).as_unit("s"),
            "end": pd.DatetimeIndex([stamp(b) if b else pd.NaT for _, b in spans]).as_unit("s"),
            "causes": "",
            "threat": None,
        }
    )


def test_estimate_on_a_hand_checked_example():
    events = alerts(
        [
            ("00:00", "00:10"),
            ("01:00", "01:20"),
            ("02:00", "02:30"),
            ("03:00", "03:40"),
            ("04:00", "04:50"),
            ("06:00", "07:30"),
        ]
    )
    got = remaining.estimate(events, stamp("06:25"))
    # Past durations 10..50 min; three of them outlasted 25 min, leaving 5, 15 and 25 min.
    assert got["elapsed_min"] == 25 and got["pool_size"] == 3 and got["pool"] == "all history"
    assert (got["low_min"], got["median_min"], got["high_min"]) == (7.0, 15.0, 23.0)
    assert (got["baseline_low_min"], got["baseline_median_min"], got["baseline_high_min"]) == (14.0, 30.0, 46.0)


def test_no_estimate_when_nothing_is_running_or_nothing_has_ended():
    events = alerts([("00:00", "00:10"), ("06:00", "07:30")])
    assert remaining.estimate(events, stamp("05:00")) is None
    assert remaining.estimate(events, stamp("07:30")) is None  # ended exactly now
    assert remaining.estimate(alerts([("06:00", "07:30")]), stamp("06:10")) is None


def test_an_alert_longer_than_any_before_falls_back_to_the_baseline():
    events = alerts([("00:00", "00:10"), ("01:00", "01:20"), ("06:00", "09:00")])
    got = remaining.estimate(events, stamp("07:00"))
    assert got["pool"] == "none lasted this long" and got["pool_size"] == 0
    assert got["median_min"] == got["baseline_median_min"] == 15.0


def test_estimate_does_not_change_when_the_future_is_hidden_or_altered():
    rng = np.random.default_rng(3)
    checked = 0
    for i in rng.integers(200, 1400, 60):
        start, end = EVENTS["start"].iloc[i], EVENTS["end"].iloc[i]
        t = (start + (end - start) * float(rng.random())).floor("s")
        seen = remaining.estimate(EVENTS, t)
        if seen is None:
            continue  # t fell exactly on the start or the end
        assert remaining.estimate(data.truncate(EVENTS, t), t) == seen
        altered = EVENTS.copy()
        altered.loc[i, "end"] = end + pd.Timedelta(hours=3)
        later = altered.index > i
        altered.loc[later, "start"] += pd.Timedelta(hours=5)
        altered.loc[later, "end"] += pd.Timedelta(hours=6)
        assert remaining.estimate(altered, t) == seen
        checked += 1
    assert checked >= 50


def test_backtest_rows_are_origins_with_a_running_alert_and_the_true_remaining_time():
    start, end = EVENTS["start"].iloc[1200].floor("15min"), EVENTS["start"].iloc[1400].floor("15min")
    rows = remaining.backtest(EVENTS, start, end)
    assert len(rows) > 50
    for t, row in rows.iloc[:: max(len(rows) // 25, 1)].iterrows():
        alert = EVENTS[(EVENTS["start"] <= t) & (EVENTS["end"] > t)].iloc[0]
        assert row["alert_start"] == alert["start"]
        assert row["truth_min"] == (alert["end"] - t).total_seconds() / 60


def test_intervals_cover_about_80_percent_on_memoryless_durations():
    """Synthetic durations are exponential, so both the estimate and the baseline should be calibrated."""
    start, end = EVENTS["start"].iloc[700].floor("15min"), EVENTS["start"].iloc[1490].floor("15min")
    report = remaining.score(remaining.backtest(EVENTS, start, end))
    assert report["origins"] > 400
    for name in ("estimate", "baseline"):
        assert 0.72 <= report[name]["coverage_80"] <= 0.88
