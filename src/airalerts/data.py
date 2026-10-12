"""Download and clean the Kyiv City air raid alert history.

Source of truth is the Kyiv Digital open API, which returns every alert start
(state 1) and all-clear (state 0) since February 2022 and is current to the
minute. The weekly open-data file from the city portal is only used as a
cross-check.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

HISTORY_URL = "https://kyiv.digital/open-api/air-alert/history"
OPEN_DATA_URL = (
    "https://data.kyivcity.gov.ua/dataset/bb0f27d3-00e2-414a-9d6e-a8b6633f8a2a/"
    "resource/5e4fb8a8-f0c8-4a12-885f-192d1f0dba75/download/airalerts.json"
)
HISTORY_FILE = "kyiv_digital_history.json"
OPEN_DATA_FILE = "airalerts.json"
META_FILE = "meta.json"

KYIV_TZ = "Europe/Kyiv"
REGION = "Kyiv City"
# The published file starts here; earlier API items are starts with no all-clear.
SERIES_START = pd.Timestamp("2022-02-28", tz=KYIV_TZ).tz_convert("UTC")
# First Kyiv alert that carries a yellow/red level.
REFORM = pd.Timestamp("2026-09-07 17:39:50", tz=KYIV_TZ).tz_convert("UTC")
# Causes that correspond to the red level; "drone" alone is yellow.
RED_CAUSES = frozenset({"missile", "ballistic", "massive-drone"})


def parse_history(body: bytes) -> list[dict]:
    """Items of the history response. The API answers in JSON or in XML regardless of the Accept header."""
    text = body.decode("utf-8-sig").lstrip()
    if not text.startswith("<"):
        return json.loads(text)["items"]
    return [
        {
            "state": int(item.findtext("state")),
            "causes": [cause.text for cause in item.find("causes")],
            "created_at": item.findtext("created_at"),
        }
        for item in ElementTree.fromstring(text).find("items")
    ]


def download(raw_dir: Path) -> dict:
    """Fetch both sources into `raw_dir` and record when the fetch started."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetched = datetime.now(timezone.utc).replace(microsecond=0)
    for name, url in ((HISTORY_FILE, HISTORY_URL), (OPEN_DATA_FILE, OPEN_DATA_URL)):
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=60)
        response.raise_for_status()
        body = response.content
        if name == HISTORY_FILE:
            body = json.dumps({"items": parse_history(body)}, ensure_ascii=False).encode("utf-8")
        (raw_dir / name).write_bytes(body)
    meta = {"fetched_at_utc": fetched.isoformat()}
    (raw_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def local_to_utc(local: pd.Series) -> tuple[pd.Series, dict]:
    """Convert Kyiv wall-clock times without an offset to UTC.

    `local` must be in chronological order. An autumn wall-clock hour happens
    twice; the first pass is assumed unless that would make time run backwards.
    A spring wall-clock hour that does not exist is moved to the next valid instant.
    """
    local = local.reset_index(drop=True)
    n = len(local)
    first = local.dt.tz_localize(KYIV_TZ, ambiguous=np.ones(n, bool), nonexistent="shift_forward")
    second = local.dt.tz_localize(KYIV_TZ, ambiguous=np.zeros(n, bool), nonexistent="shift_forward")
    strict = local.dt.tz_localize(KYIV_TZ, ambiguous=np.ones(n, bool), nonexistent="NaT")
    utc = first.dt.tz_convert("UTC")
    alt = second.dt.tz_convert("UTC")
    ambiguous = np.flatnonzero((utc != alt).to_numpy())
    for i in ambiguous:
        if i > 0 and utc.iloc[i] < utc.iloc[i - 1]:
            utc.iloc[i] = alt.iloc[i]
    stats = {"ambiguous_dst": int(len(ambiguous)), "nonexistent_dst": int(strict.isna().sum())}
    return utc, stats


def _threat(causes: list[str]) -> str | None:
    if not causes:
        return None
    return "missile" if RED_CAUSES.intersection(causes) else "drone"


def pair_events(history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Turn the start / all-clear stream into one row per alert.

    `history` has columns `state`, `causes`, `ts` (UTC) in chronological order.
    An all-clear closes the start immediately before it, which is the rule the
    published file follows. Repeated starts and repeated all-clears (2022 only)
    are dropped and counted.
    """
    state = history["state"].to_numpy()
    closes = np.flatnonzero((state[1:] == 0) & (state[:-1] == 1)) + 1
    starts = history.iloc[closes - 1]
    events = pd.DataFrame(
        {
            "region": REGION,
            "start": starts["ts"].to_numpy(),
            "end": history["ts"].to_numpy()[closes],
            "causes": [",".join(c) for c in starts["causes"]],
            "threat": [_threat(c) for c in starts["causes"]],
        }
    )
    if len(state) and state[-1] == 1:
        # Alert still running at fetch time. Its cause list is not final, so it is left blank.
        open_alert = {"region": REGION, "start": history["ts"].iloc[-1], "end": pd.NaT, "causes": "", "threat": None}
        events = pd.concat([events, pd.DataFrame([open_alert])], ignore_index=True)
    for col in ("start", "end"):
        events[col] = pd.to_datetime(events[col], utc=True).dt.as_unit("s")
    stats = {
        "repeated_starts_dropped": int(((state[1:] == 1) & (state[:-1] == 1)).sum()),
        "repeated_clears_dropped": int(((state[1:] == 0) & (state[:-1] == 0)).sum()),
    }
    return events, stats


def load_history(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    items = json.loads((raw_dir / HISTORY_FILE).read_text(encoding="utf-8"))["items"]
    history = pd.DataFrame(items)
    history["local"] = pd.to_datetime(history["created_at"])
    history = history.sort_values("local", kind="stable").reset_index(drop=True)
    history["ts"], stats = local_to_utc(history["local"])
    return history[["state", "causes", "ts"]], stats


def fetched_at(raw_dir: Path) -> pd.Timestamp:
    meta = json.loads((raw_dir / META_FILE).read_text(encoding="utf-8"))
    return pd.Timestamp(meta["fetched_at_utc"]).tz_convert("UTC")


def truncate(events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Keep only what was known at `as_of`: later alerts go, later ends become open."""
    events = events[events["start"] <= as_of].copy()
    unfinished = events["end"] > as_of
    events.loc[unfinished, "end"] = pd.NaT
    events.loc[unfinished, ["causes", "threat"]] = ["", None]
    return events.reset_index(drop=True)


def load_events(raw_dir: Path, as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean alert table as known at `as_of` (default: the fetch time)."""
    history, info = load_history(raw_dir)
    events, pair_stats = pair_events(history)
    info.update(pair_stats)
    events = events[events["start"] >= SERIES_START]
    as_of = fetched_at(raw_dir) if as_of is None else as_of
    events = truncate(events, as_of)
    validate(events)
    info.update(
        {
            "as_of_utc": as_of.isoformat(),
            "alerts": int(len(events)),
            "first_start_utc": events["start"].min().isoformat(),
            "last_start_utc": events["start"].max().isoformat(),
            "open_alert": bool(events["end"].isna().any()),
        }
    )
    return events, info


def validate(events: pd.DataFrame) -> None:
    start, end = events["start"], events["end"]
    assert start.is_monotonic_increasing and start.is_unique, "starts must be sorted and unique"
    closed = end.notna()
    assert (end[closed] > start[closed]).all(), "every closed alert must end after it starts"
    assert not closed.iloc[:-1].eq(False).any(), "only the latest alert may be open"
    assert (start.iloc[1:].to_numpy() >= end.iloc[:-1].to_numpy()).all(), "alerts must not overlap"


def compare_with_open_data(events: pd.DataFrame, raw_dir: Path) -> dict:
    """Check the API-derived table against the weekly published file."""
    published = pd.DataFrame(json.loads((raw_dir / OPEN_DATA_FILE).read_text(encoding="utf-8")))
    start, _ = local_to_utc(pd.to_datetime(published["dateTimeStart"]))
    end, _ = local_to_utc(pd.to_datetime(published["dateTimeEnd"]))
    published = pd.DataFrame({"start": start.dt.as_unit("s"), "end_published": end.dt.as_unit("s")})
    # Compare only the period both tables cover (the events may be cut by --as-of).
    cutoff = min(published["start"].max(), events["start"].max())
    published = published[published["start"] <= cutoff]
    overlap = events[events["start"] <= cutoff]
    merged = published.merge(overlap[["start", "end"]], on="start", how="outer", indicator=True)
    both = merged[merged["_merge"] == "both"]
    return {
        "published_records": int(len(published)),
        "matched_start": int(len(both)),
        "matched_start_and_end": int((both["end"] == both["end_published"]).sum()),
        "published_only": int((merged["_merge"] == "left_only").sum()),
        "api_only": int((merged["_merge"] == "right_only").sum()),
    }
