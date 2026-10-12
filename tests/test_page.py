"""The page must be self-contained and must not show anything from after its instant."""

import pandas as pd
from test_leakage import synthetic_events

from airalerts import data, evaluate, grid, now, page
from airalerts.model import FINAL, HORIZONS

EVENTS = synthetic_events(1500)


def altered_future(events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Same past, different future: later alerts move, and a running alert ends later."""
    out = events.copy()
    later = out["start"] > as_of
    out.loc[later, "start"] += pd.Timedelta(hours=2)
    out.loc[later, "end"] += pd.Timedelta(hours=2)
    running = (out["start"] <= as_of) & (out["end"] > as_of)
    out.loc[running, "end"] += pd.Timedelta(minutes=30)
    return out


def test_strip_and_table_show_nothing_from_after_the_instant():
    quiet = EVENTS["end"].iloc[1300] + pd.Timedelta(minutes=5)
    start, end = EVENTS["start"].iloc[1301], EVENTS["end"].iloc[1301]
    during = (start + (end - start) / 2).floor("s")
    for as_of in (quiet, during):
        for block in (page.strip_block, page.strip_table):
            seen = block(EVENTS, as_of)
            assert block(data.truncate(EVENTS, as_of), as_of) == seen
            assert block(altered_future(EVENTS, as_of), as_of) == seen
    assert "still running" in page.strip_table(EVENTS, during)
    assert "still running" not in page.strip_table(EVENTS, quiet)


def test_page_renders_and_loads_nothing_from_the_network():
    as_of = EVENTS["end"].iloc[-1]
    table = grid.make_grid(EVENTS, as_of, HORIZONS)
    report, predictions, _ = evaluate.evaluate(table, HORIZONS, [FINAL])
    html = page.render(EVENTS, as_of, now.forecast(EVENTS, as_of), report, predictions)
    assert html.startswith("<!doctype html>")
    for fragment in ("http://", "https://", "<link", "@import", "src="):
        assert fragment not in html
    for h in HORIZONS:
        assert f'data-horizon="{h}"' in html
    assert html.count('class="tile"') == len(HORIZONS)
    assert max(map(len, html.splitlines())) < 120
