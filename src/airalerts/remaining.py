"""How much longer will a running alert last?

Design, fixed before any score on the test weeks was computed:

- Estimate at instant t for an alert that has run e minutes: take the alerts that had
  ended by t and started in the previous 16 weeks, keep those that lasted longer than e,
  and read the 10th, 50th and 90th percentiles of (duration - e). The middle value is
  the forecast; the outer two are the 80% interval.
- If fewer than 30 of those alerts lasted longer than e, use every alert that had ended
  by t. If none ever lasted that long, fall back to the baseline below.
- Baseline: ignore the minutes elapsed and treat the alert as if it had just started,
  that is the same percentiles of the full durations in the 16-week pool.
- Backtest: every 15 minutes of the test period at which an alert was running.
  Scores: share of true remaining times inside the 80% interval, mean absolute error of
  the middle value, mean interval width; 95% intervals from resampling whole Kyiv days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airalerts.data import KYIV_TZ, REFORM
from airalerts.grid import DAY, epoch_seconds

POOL_SECONDS = 16 * 7 * DAY
MIN_POOL = 30
QUANTILES = (0.1, 0.5, 0.9)
BOOTSTRAP_SAMPLES = 1000
FREQ = "15min"


def _quantiles(minutes: np.ndarray) -> tuple[float, float, float]:
    low, median, high = np.quantile(minutes, QUANTILES)
    return float(low), float(median), float(high)


def estimate(events: pd.DataFrame, t: pd.Timestamp) -> dict | None:
    """Remaining-time forecast at t, or None when no alert is running then.

    Only alerts that started at or before t are read, and only the ends that fall at or
    before t. The running alert contributes its start and nothing else.
    """
    t_s = int(epoch_seconds([t])[0])
    start_s = epoch_seconds(events["start"])
    end_s = epoch_seconds(events["end"])
    running = int(np.searchsorted(start_s, t_s, side="right")) - 1
    if running < 0 or end_s[running] <= t_s:
        return None
    elapsed = (t_s - start_s[running]) / 60
    # Alerts do not overlap, so every alert before the running one had ended by t.
    durations = (end_s[:running] - start_s[:running]) / 60
    recent = durations[start_s[:running] >= t_s - POOL_SECONDS]
    if len(recent) == 0:
        recent = durations
    if len(recent) == 0:
        return None  # nothing has ever ended: no basis for an estimate
    base_low, base_median, base_high = _quantiles(recent)
    pool, source = recent[recent > elapsed], "last 16 weeks"
    if len(pool) < MIN_POOL:
        pool, source = durations[durations > elapsed], "all history"
    if len(pool) == 0:
        low, median, high, source = base_low, base_median, base_high, "none lasted this long"
    else:
        low, median, high = _quantiles(pool - elapsed)
    return {
        "elapsed_min": round(elapsed, 1),
        "low_min": round(low, 1),
        "median_min": round(median, 1),
        "high_min": round(high, 1),
        "pool": source,
        "pool_size": int(len(pool)),
        "baseline_low_min": round(base_low, 1),
        "baseline_median_min": round(base_median, 1),
        "baseline_high_min": round(base_high, 1),
    }


def backtest(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One row per 15-minute origin in [start, end) with a running alert whose end is known."""
    start_s = epoch_seconds(events["start"])
    end_known = events["end"].notna().to_numpy()
    end_s = epoch_seconds(events["end"])
    rows = []
    for t in pd.date_range(start, end, freq=FREQ, inclusive="left", unit="s"):
        t_s = int(epoch_seconds([t])[0])
        running = int(np.searchsorted(start_s, t_s, side="right")) - 1
        if running < 0 or end_s[running] <= t_s or not end_known[running]:
            continue
        forecast = estimate(events, t)
        if forecast is not None:
            truth = (end_s[running] - t_s) / 60
            rows.append({"t": t, "alert_start": events["start"].iloc[running], "truth_min": truth, **forecast})
    return pd.DataFrame(rows).set_index("t")


def _day_sums(values: np.ndarray, day_index: np.ndarray) -> np.ndarray:
    return np.bincount(day_index, weights=values)


def score(rows: pd.DataFrame, seed: int = 0) -> dict:
    """Coverage, median error and width for the estimate and the baseline, with intervals from resampled days."""
    truth = rows["truth_min"].to_numpy()
    days = rows.index.tz_convert(KYIV_TZ).strftime("%Y-%m-%d").to_numpy()
    _, day_index = np.unique(days, return_inverse=True)
    counts = _day_sums(np.ones(len(rows)), day_index)
    draws = np.random.default_rng(seed).integers(0, len(counts), size=(BOOTSTRAP_SAMPLES, len(counts)))
    n = counts[draws].sum(axis=1)

    def mean_with_interval(values: np.ndarray) -> tuple[float, list[float], np.ndarray]:
        resampled = _day_sums(values, day_index)[draws].sum(axis=1) / n
        return float(values.mean()), [float(np.quantile(resampled, q)) for q in (0.025, 0.975)], resampled

    out = {"origins": int(len(rows)), "alerts": int(rows["alert_start"].nunique()), "days": int(len(counts))}
    errors, resampled_errors = {}, {}
    for name, prefix in (("estimate", ""), ("baseline", "baseline_")):
        low, median, high = (rows[f"{prefix}{part}_min"].to_numpy() for part in ("low", "median", "high"))
        coverage, coverage_ci, _ = mean_with_interval(((truth >= low) & (truth <= high)).astype(float))
        error, error_ci, resampled_errors[name] = mean_with_interval(np.abs(median - truth))
        errors[name] = error
        out[name] = {
            "coverage_80": round(coverage, 4),
            "coverage_80_ci95": [round(v, 4) for v in coverage_ci],
            "median_abs_error_min": round(error, 2),
            "median_abs_error_min_ci95": [round(v, 2) for v in error_ci],
            "mean_interval_width_min": round(float((high - low).mean()), 1),
        }
    reduction = 1 - resampled_errors["estimate"] / resampled_errors["baseline"]
    out["error_reduction_vs_baseline"] = round(1 - errors["estimate"] / errors["baseline"], 4)
    out["error_reduction_vs_baseline_ci95"] = [round(float(np.quantile(reduction, q)), 4) for q in (0.025, 0.975)]
    return out


def evaluate(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    rows = backtest(events, start, end)
    report = {
        "design": "15-minute origins with a running alert; pool = alerts ended by t, last 16 weeks",
        "test_start_utc": start.isoformat(),
        "test_end_utc": end.isoformat(),
        "test_period": score(rows),
        "since_reform": score(rows[rows.index >= REFORM]),
    }
    return report, rows


def summary_table(report: dict) -> pd.DataFrame:
    rows = []
    for period in ("test_period", "since_reform"):
        block = report[period]
        for name in ("estimate", "baseline"):
            m = block[name]
            rows.append(
                {
                    "period": period,
                    "forecast": "conditional on minutes elapsed" if name == "estimate" else "ignores minutes elapsed",
                    "origins": block["origins"],
                    "alerts": block["alerts"],
                    "coverage_80_%": round(100 * m["coverage_80"], 1),
                    "coverage_ci95_%": " to ".join(f"{100 * v:.1f}" for v in m["coverage_80_ci95"]),
                    "median_abs_error_min": m["median_abs_error_min"],
                    "error_ci95_min": " to ".join(f"{v:.1f}" for v in m["median_abs_error_min_ci95"]),
                    "mean_width_min": m["mean_interval_width_min"],
                }
            )
    return pd.DataFrame(rows)
