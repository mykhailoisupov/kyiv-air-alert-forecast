"""The live forecast must depend only on what was known at its instant."""

import numpy as np
import pandas as pd
from test_leakage import synthetic_events

from airalerts import data, evaluate, grid, now
from airalerts.model import FINAL, HORIZONS

EVENTS = synthetic_events(1500)  # about 290 days


def quiet_instant(events: pd.DataFrame, position: int) -> pd.Timestamp:
    """Ten minutes after an alert ends; the synthetic gaps are longer than that only mostly, so check."""
    t = events["end"].iloc[position] + pd.Timedelta(minutes=10)
    assert t < events["start"].iloc[position + 1]
    return t


def test_live_forecast_does_not_change_when_the_future_is_hidden_or_altered():
    as_of = quiet_instant(EVENTS, 1300)
    seen = now.forecast(EVENTS, as_of)
    assert not seen["active"] and set(seen["horizons"]) == set(HORIZONS)
    assert now.forecast(data.truncate(EVENTS, as_of), as_of) == seen

    altered = EVENTS.copy()
    later = altered["start"] > as_of
    altered.loc[later, "start"] += pd.Timedelta(hours=2)
    altered.loc[later, "end"] += pd.Timedelta(hours=2)
    assert now.forecast(altered, as_of) == seen


def test_live_forecast_hides_the_end_of_a_running_alert():
    start, end = EVENTS["start"].iloc[1300], EVENTS["end"].iloc[1300]
    as_of = start + (end - start) / 2
    seen = now.forecast(EVENTS, as_of.floor("s"))
    assert seen["active"] and seen["horizons"] == {}
    lengthened = EVENTS.copy()
    lengthened.loc[1300, "end"] = end + pd.Timedelta(minutes=1)
    assert now.forecast(lengthened, as_of.floor("s")) == seen


def test_live_forecast_equals_the_backtest_forecast_for_the_same_instant():
    """The backtest saw the whole history; the live forecast sees only the past. They must agree."""
    table = grid.make_grid(EVENTS, EVENTS["end"].iloc[-1], HORIZONS)
    start, end = evaluate.latest_window(table, HORIZONS, test_weeks=4)
    compared = 0
    for h in HORIZONS:
        predictions, _, _ = evaluate.backtest(table, h, [FINAL], start, end)
        for fold_start in evaluate.fold_starts(start, end):
            if fold_start not in predictions.index:
                continue  # an alert was running at that instant
            live = now.forecast(EVENTS, fold_start)["horizons"][h]
            assert np.isclose(live["model"], predictions.loc[fold_start, FINAL.name], atol=1e-4)
            assert np.isclose(live["simple"], predictions.loc[fold_start, live["simple_name"]], atol=1e-4)
            compared += 1
    assert compared >= 3
