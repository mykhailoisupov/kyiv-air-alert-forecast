"""Reference forecasts built only from labels that had closed before the forecast time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from airalerts.grid import DAY, HOUR, epoch_seconds

WINDOW = 28 * DAY
REFERENCE = "hour_28d"  # the forecast every skill score is measured against


def trailing_rate(
    t_s: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    horizon_s: int,
    window_s: int | None,
    prior: np.ndarray | float = 0.5,
    strength: float = 1.0,
) -> np.ndarray:
    """Share of positives among earlier origins of the same group, shrunk towards `prior`.

    An origin s counts towards the forecast at t only if its label window had
    closed by then (s + horizon <= t) and, with a window, closed within it.
    """
    prior = np.broadcast_to(np.asarray(prior, dtype=float), t_s.shape)
    out = np.empty(len(t_s))
    for value in np.unique(group):
        rows = np.flatnonzero(group == value)
        closed_at = t_s[rows] + horizon_s
        positives = np.concatenate([[0.0], np.cumsum(y[rows])])
        hi = np.searchsorted(closed_at, t_s[rows], side="right")
        lo = np.searchsorted(closed_at, t_s[rows] - window_s, side="right") if window_s else np.zeros_like(hi)
        out[rows] = (positives[hi] - positives[lo] + strength * prior[rows]) / (hi - lo + strength)
    return out


def reference_forecasts(frame: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Reference probabilities for `onset_{h}h` on quiet origins, in time order."""
    t_s = epoch_seconds(frame.index)
    y = frame[f"onset_{horizon_h}h"].to_numpy()
    horizon_s = horizon_h * HOUR
    everything = np.zeros(len(frame), dtype=int)
    hour = frame["local_hour"].to_numpy()
    recent_onset = (frame[f"starts_{horizon_h}h"] > 0).to_numpy().astype(int)

    out = pd.DataFrame(index=frame.index)
    out["const_28d"] = trailing_rate(t_s, y, everything, horizon_s, WINDOW)
    out["hour_28d"] = trailing_rate(t_s, y, hour, horizon_s, WINDOW, prior=out["const_28d"].to_numpy(), strength=10)
    out["hour_all"] = trailing_rate(t_s, y, hour, horizon_s, None)
    out["markov_28d"] = trailing_rate(t_s, y, recent_onset, horizon_s, WINDOW, prior=out["const_28d"].to_numpy(), strength=10)
    return out
