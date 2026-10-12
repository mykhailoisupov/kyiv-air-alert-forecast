"""Walk-forward backtest on the latest weeks, scored against reference forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from airalerts.baselines import REFERENCE, reference_forecasts
from airalerts.data import KYIV_TZ, REFORM
from airalerts.model import ModelSpec, design, new_estimator

WEEK = pd.Timedelta(days=7)
TEST_WEEKS = 8
BOOTSTRAP_SAMPLES = 1000


def latest_window(grid: pd.DataFrame, horizons: tuple[int, ...], test_weeks: int = TEST_WEEKS) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The latest `test_weeks` weeks for which every horizon's label is known. End is exclusive."""
    known = grid[[f"onset_{h}h" for h in horizons]].notna().all(axis=1)
    step = grid.index[1] - grid.index[0]
    end = grid.index[known][-1] + step
    return end - test_weeks * WEEK, end


def fold_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end - pd.Timedelta(seconds=1), freq=WEEK))


def split(frame: pd.DataFrame, fold_start: pd.Timestamp, horizon_h: int, train_weeks: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows are origins whose label window closed by `fold_start`; test rows are the week after."""
    closes = frame.index + pd.Timedelta(hours=horizon_h)
    train = closes <= fold_start
    if train_weeks is not None:
        train &= frame.index >= fold_start - train_weeks * WEEK
    test = (frame.index >= fold_start) & (frame.index < fold_start + WEEK)
    return frame[train], frame[test]


def backtest(grid: pd.DataFrame, horizon_h: int, specs: list[ModelSpec], start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[dict]]:
    """Out-of-sample probabilities for one horizon on quiet origins in [start, end)."""
    target = f"onset_{horizon_h}h"
    frame = grid[~grid["active"] & grid[target].notna()]
    frame = frame.join(reference_forecasts(frame, horizon_h))
    parts, folds = [], []
    for fold_start in fold_starts(start, end):
        fold = {"fold_start_utc": fold_start.isoformat()}
        test_part = None
        for spec in specs:
            train, test = split(frame, fold_start, horizon_h, spec.train_weeks)
            test = test[test.index < end]
            model = new_estimator().fit(design(train, spec), train[target].to_numpy())
            if test_part is None:
                test_part = test[[target, "local_hour", *reference_forecasts_columns(frame)]].rename(columns={target: "y"})
            test_part[spec.name] = model.predict_proba(design(test, spec))[:, 1]
            fold[spec.name] = {
                "train_rows": int(len(train)),
                "last_train_label_closes_utc": (train.index[-1] + pd.Timedelta(hours=horizon_h)).isoformat(),
            }
        fold["test_rows"] = int(len(test_part))
        parts.append(test_part)
        folds.append(fold)
    return pd.concat(parts), folds


def reference_forecasts_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in ("const_28d", "hour_28d", "hour_all", "markov_28d") if c in frame.columns]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def skill_interval(y: np.ndarray, p: np.ndarray, reference: np.ndarray, days: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% interval for Brier skill, resampling whole Kyiv calendar days."""
    _, day_index = np.unique(days, return_inverse=True)
    model_error = np.bincount(day_index, weights=(p - y) ** 2)
    reference_error = np.bincount(day_index, weights=(reference - y) ** 2)
    draws = np.random.default_rng(seed).integers(0, len(model_error), size=(BOOTSTRAP_SAMPLES, len(model_error)))
    skill = 1 - model_error[draws].sum(axis=1) / reference_error[draws].sum(axis=1)
    return float(np.quantile(skill, 0.025)), float(np.quantile(skill, 0.975))


def score(predictions: pd.DataFrame, forecasts: list[str]) -> dict:
    y = predictions["y"].to_numpy()
    reference = predictions[REFERENCE].to_numpy()
    days = predictions.index.tz_convert(KYIV_TZ).strftime("%Y-%m-%d").to_numpy()
    out = {"rows": int(len(y)), "days": int(len(np.unique(days))), "prevalence": round(float(y.mean()), 4), "forecasts": {}}
    for name in forecasts:
        p = predictions[name].to_numpy()
        low, high = skill_interval(y, p, reference, days)
        out["forecasts"][name] = {
            "brier": round(brier(y, p), 5),
            "brier_skill": round(1 - brier(y, p) / brier(y, reference), 4),
            "brier_skill_ci95": [round(low, 4), round(high, 4)],
            "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 5),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
        }
    return out


def reliability(predictions: pd.DataFrame, forecast: str, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(predictions[forecast], edges) - 1, 0, bins - 1)
    table = predictions.assign(bucket=bucket).groupby("bucket").agg(forecast=(forecast, "mean"), observed=("y", "mean"), rows=("y", "size"))
    return table.reset_index(drop=True)


def evaluate(grid: pd.DataFrame, horizons: tuple[int, ...], specs: list[ModelSpec]) -> tuple[dict, dict[int, pd.DataFrame]]:
    start, end = latest_window(grid, horizons)
    report = {"test_start_utc": start.isoformat(), "test_end_utc": end.isoformat(), "reference": REFERENCE, "horizons": {}}
    predictions = {}
    for h in horizons:
        predictions[h], folds = backtest(grid, h, specs, start, end)
        forecasts = reference_forecasts_columns(predictions[h]) + [spec.name for spec in specs]
        report["horizons"][str(h)] = {
            "test_period": score(predictions[h], forecasts),
            "since_reform": score(predictions[h][predictions[h].index >= REFORM], forecasts),
            "folds": folds,
        }
    return report, predictions


def summary_table(report: dict, period: str = "test_period") -> pd.DataFrame:
    rows = []
    for h, block in report["horizons"].items():
        for name, m in block[period]["forecasts"].items():
            rows.append(
                {
                    "horizon_h": int(h),
                    "forecast": name,
                    "brier": m["brier"],
                    "skill_%": round(100 * m["brier_skill"], 1),
                    "ci95_%": f"{100 * m['brier_skill_ci95'][0]:+.1f} to {100 * m['brier_skill_ci95'][1]:+.1f}",
                    "log_loss": m["log_loss"],
                    "pr_auc": m["pr_auc"],
                    "roc_auc": m["roc_auc"],
                    "prevalence": block[period]["prevalence"],
                    "rows": block[period]["rows"],
                }
            )
    return pd.DataFrame(rows)
