"""As-of features and labels on a time grid.

Every feature at origin t is a function of alert starts at or before t and
alert ends at or before t, and nothing else. An alert that is still running at
t contributes its start and its elapsed time, never its end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airalerts.data import KYIV_TZ

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
HOUR = 3600
DAY = 24 * HOUR
# Stand-in end for an alert that has not finished; far beyond any origin.
OPEN_END = np.iinfo(np.int64).max // 4

START_WINDOWS = {"starts_6h": 6 * HOUR, "starts_24h": DAY, "starts_72h": 3 * DAY, "starts_7d": 7 * DAY, "starts_28d": 28 * DAY}
MINUTE_WINDOWS = {"alert_min_24h": DAY, "alert_min_7d": 7 * DAY}


def epoch_seconds(times) -> np.ndarray:
    """UTC timestamps -> int64 seconds; missing values become OPEN_END."""
    values = pd.Series(pd.DatetimeIndex(times))
    seconds = (values - EPOCH) // pd.Timedelta(seconds=1)
    return seconds.fillna(OPEN_END).to_numpy(dtype=np.int64)


def covered_seconds(start_s: np.ndarray, end_s: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Seconds spent under alert before each time in `t_s`.

    Alerts are sorted and do not overlap, so every alert before the latest one
    started at or before t has fully ended by t; the latest one is cut at t.
    """
    full = np.concatenate([[0], np.cumsum(np.minimum(end_s, OPEN_END // 2) - start_s)])
    k = np.searchsorted(start_s, t_s, side="right")
    last = np.maximum(k - 1, 0)
    partial = np.where(k > 0, np.minimum(end_s[last], t_s) - start_s[last], 0) if len(start_s) else np.zeros_like(t_s)
    return full[last] * (k > 0) + partial


def build_features(events: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """One row of features per origin, using only what was known at that origin."""
    start_s = epoch_seconds(events["start"])
    end_s = epoch_seconds(events["end"])
    missile = events["threat"].eq("missile").fillna(False).to_numpy()
    t_s = epoch_seconds(origins)

    started = np.searchsorted(start_s, t_s, side="right")  # alerts started at or before t
    ended = np.searchsorted(end_s, t_s, side="right")  # alerts ended at or before t
    last_started = np.maximum(started - 1, 0)
    last_ended = np.maximum(ended - 1, 0)
    never = np.float64(28 * 24)  # cap, also used before the first alert

    out = pd.DataFrame(index=origins)
    out["active"] = started > ended
    for name, window in START_WINDOWS.items():
        out[name] = started - np.searchsorted(start_s, t_s - window, side="right")
    covered = covered_seconds(start_s, end_s, t_s)
    for name, window in MINUTE_WINDOWS.items():
        out[name] = (covered - covered_seconds(start_s, end_s, t_s - window)) / 60
    out["since_start_h"] = np.where(started > 0, np.minimum((t_s - start_s[last_started]) / HOUR, never), never)
    out["since_end_h"] = np.where(ended > 0, np.minimum((t_s - end_s[last_ended]) / HOUR, never), never)
    # Duration and threat type are taken from the last alert that has already ended:
    # the API's cause list can grow while an alert is running.
    out["last_dur_min"] = np.where(ended > 0, (end_s[last_ended] - start_s[last_ended]) / 60, 0.0)
    out["last_missile"] = np.where(ended > 0, missile[last_ended], False).astype(float)

    local = origins.tz_convert(KYIV_TZ)
    out["local_hour"] = local.hour
    angle = 2 * np.pi * (local.hour + local.minute / 60) / 24
    for harmonic in (1, 2):
        out[f"hour_sin{harmonic}"] = np.sin(harmonic * angle)
        out[f"hour_cos{harmonic}"] = np.cos(harmonic * angle)
    return out


def build_labels(events: pd.DataFrame, origins: pd.DatetimeIndex, horizons: tuple[int, ...], as_of: pd.Timestamp) -> pd.DataFrame:
    """`onset_{H}h` = a new alert starts in (t, t+H]. Missing where t+H is past `as_of`."""
    start_s = epoch_seconds(events["start"])
    t_s = epoch_seconds(origins)
    as_of_s = epoch_seconds([as_of])[0]
    after_now = np.searchsorted(start_s, t_s, side="right")
    out = pd.DataFrame(index=origins)
    for h in horizons:
        onset = (np.searchsorted(start_s, t_s + h * HOUR, side="right") - after_now > 0).astype(float)
        onset[t_s + h * HOUR > as_of_s] = np.nan
        out[f"onset_{h}h"] = onset
    return out


def make_grid(events: pd.DataFrame, as_of: pd.Timestamp, horizons: tuple[int, ...] = (1, 3, 6), freq: str = "1h") -> pd.DataFrame:
    """Features and labels for every origin from the first alert to `as_of`."""
    origins = pd.date_range(events["start"].min().ceil(freq), as_of.floor(freq), freq=freq, unit="s")
    return build_features(events, origins).join(build_labels(events, origins, horizons, as_of))
