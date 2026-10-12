"""Checks that nothing used at forecast time t comes from after t."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airalerts import baselines, data, evaluate, grid

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HORIZONS = (1, 3, 6)


def synthetic_events(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(4 * 3600, n).astype(int) + 60
    durations = rng.exponential(40 * 60, n).astype(int) + 30
    start = np.cumsum(gaps) + np.concatenate([[0], np.cumsum(durations[:-1])])
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "region": data.REGION,
            "start": (t0 + pd.to_timedelta(start, unit="s")).as_unit("s"),
            "end": (t0 + pd.to_timedelta(start + durations, unit="s")).as_unit("s"),
            "causes": "",
            "threat": rng.choice(np.array([None, "drone", "missile"], dtype=object), n),
        }
    )


def sample_origins(events: pd.DataFrame, n: int, seed: int = 2) -> pd.DatetimeIndex:
    """Random instants plus the awkward ones: exactly at a start, exactly at an end."""
    rng = np.random.default_rng(seed)
    first, last = events["start"].iloc[5], events["end"].dropna().iloc[-1]
    span = int((last - first).total_seconds())
    random = [first + pd.Timedelta(seconds=int(s)) for s in rng.integers(0, span, n)]
    exact = list(events["start"].iloc[10:20]) + list(events["end"].iloc[10:20])
    return pd.DatetimeIndex(sorted(random + exact)).unique().as_unit("s")


def lookahead_columns(build, events: pd.DataFrame, origins: pd.DatetimeIndex) -> set[str]:
    """Feature columns whose value at an origin changes when everything after it is hidden."""
    full = build(events, origins)
    changed = set()
    for t in origins:
        hidden = build(data.truncate(events, t), pd.DatetimeIndex([t]))
        a = full.loc[t].to_numpy(dtype=float)
        b = hidden.iloc[0].to_numpy(dtype=float)
        changed.update(full.columns[~np.isclose(a, b, equal_nan=True)])
    return changed


def peeking_features(events: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """The real features plus one that looks ahead: hours until the next alert starts."""
    out = grid.build_features(events, origins)
    start_s, t_s = grid.epoch_seconds(events["start"]), grid.epoch_seconds(origins)
    nxt = np.searchsorted(start_s, t_s, side="right")
    out["next_start_h"] = np.where(nxt < len(start_s), (start_s[np.minimum(nxt, len(start_s) - 1)] - t_s) / 3600, np.nan)
    return out


def test_features_do_not_change_when_the_future_is_hidden():
    events = synthetic_events()
    assert lookahead_columns(grid.build_features, events, sample_origins(events, 200)) == set()


def test_a_feature_that_peeks_ahead_is_caught():
    events = synthetic_events()
    assert lookahead_columns(peeking_features, events, sample_origins(events, 200)) == {"next_start_h"}


@pytest.mark.skipif(not (RAW / data.HISTORY_FILE).exists(), reason="raw data not downloaded")
def test_real_features_do_not_change_when_the_future_is_hidden():
    events, _ = data.load_events(RAW)
    recent = events[events["start"] >= events["start"].max() - pd.Timedelta(days=200)].reset_index(drop=True)
    assert lookahead_columns(grid.build_features, recent, sample_origins(recent, 300)) == set()


def test_feature_values_on_a_hand_checked_example():
    stamp = lambda s: pd.Timestamp(f"2026-10-01 {s}", tz="UTC")
    events = pd.DataFrame(
        {
            "region": data.REGION,
            "start": [stamp("10:00"), stamp("12:00")],
            "end": [stamp("10:30"), pd.NaT],
            "causes": ["missile", ""],
            "threat": ["missile", None],
        }
    )
    row = grid.build_features(events, pd.DatetimeIndex([stamp("12:15")])).iloc[0]
    assert row["active"]
    assert row["starts_1h"] == 1 and row["starts_3h"] == 2
    assert row["alert_min_24h"] == 45
    assert row["since_start_h"] == 0.25 and row["since_end_h"] == 1.75
    assert row["last_dur_min"] == 30 and row["last_missile"] == 1


def test_covered_seconds_matches_brute_force():
    events = synthetic_events(60)
    start_s, end_s = grid.epoch_seconds(events["start"]), grid.epoch_seconds(events["end"])
    times = grid.epoch_seconds(sample_origins(events, 100))
    brute = [int(np.clip(np.minimum(end_s, t) - start_s, 0, None).sum()) for t in times]
    assert grid.covered_seconds(start_s, end_s, times).tolist() == brute


def test_labels_match_brute_force_and_stop_at_the_cutoff():
    events = synthetic_events()
    as_of = events["start"].iloc[300]
    origins = sample_origins(events, 200)
    labels = grid.build_labels(data.truncate(events, as_of), origins, HORIZONS, as_of)
    for h in HORIZONS:
        window = pd.Timedelta(hours=h)
        for t, value in labels[f"onset_{h}h"].items():
            if t + window > as_of:
                assert np.isnan(value)
            else:
                assert value == float(((events["start"] > t) & (events["start"] <= t + window)).any())


def test_training_labels_close_before_each_test_fold():
    events = synthetic_events()
    as_of = events["end"].iloc[-1]
    table = grid.make_grid(events, as_of, HORIZONS)
    start, end = evaluate.latest_window(table, HORIZONS, test_weeks=3)
    for h in HORIZONS:
        frame = table[~table["active"] & table[f"onset_{h}h"].notna()]
        for fold_start in evaluate.fold_starts(start, end):
            for train_weeks in (None, 4):
                train, test = evaluate.split(frame, fold_start, h, train_weeks)
                assert len(train) and len(test)
                assert train.index.max() + pd.Timedelta(hours=h) <= fold_start
                assert test.index.min() >= fold_start


def test_reference_forecasts_ignore_labels_that_have_not_closed():
    events = synthetic_events()
    table = grid.make_grid(events, events["end"].iloc[-1], HORIZONS)
    for h in HORIZONS:
        target = f"onset_{h}h"
        frame = table[~table["active"] & table[target].notna()]
        before = baselines.reference_forecasts(frame, h)
        row = len(frame) // 2
        t = frame.index[row]
        tampered = frame.copy()
        still_open = tampered.index + pd.Timedelta(hours=h) > t
        tampered.loc[still_open, target] = 1 - tampered.loc[still_open, target]
        after = baselines.reference_forecasts(tampered, h)
        assert np.allclose(before.iloc[row], after.iloc[row])
        assert not np.allclose(before.iloc[row + 500 :], after.iloc[row + 500 :])
