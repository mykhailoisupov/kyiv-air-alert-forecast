"""Logistic onset-hazard model: P(a new alert starts within H hours | no alert now)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

LOG_FEATURES = (
    "starts_1h",
    "starts_3h",
    "starts_6h",
    "starts_24h",
    "starts_72h",
    "starts_7d",
    "starts_28d",
    "alert_min_24h",
    "alert_min_7d",
    "since_start_h",
    "since_end_h",
    "last_dur_min",
)
RAW_FEATURES = ("last_missile", "hour_sin1", "hour_cos1", "hour_sin2", "hour_cos2")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    train_weeks: int | None  # None = every earlier origin
    reference_logits: tuple[str, ...] = ()  # reference forecasts fed in as extra features
    estimator: str = "logistic"  # or "boosting", "boosting_uncalibrated"
    extra_log_features: tuple[str, ...] = ()  # further count-like columns, log-transformed


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def design(frame: pd.DataFrame, spec: ModelSpec) -> np.ndarray:
    columns = [np.log1p(frame[name].to_numpy(dtype=float)) for name in LOG_FEATURES + spec.extra_log_features]
    columns += [frame[name].to_numpy(dtype=float) for name in RAW_FEATURES]
    columns += [logit(frame[name].to_numpy(dtype=float)) for name in spec.reference_logits]
    return np.column_stack(columns)


def feature_names(spec: ModelSpec) -> list[str]:
    logged = [f"log1p({n})" for n in LOG_FEATURES + spec.extra_log_features]
    return logged + list(RAW_FEATURES) + [f"logit({n})" for n in spec.reference_logits]


def new_estimator(spec: ModelSpec) -> Pipeline | CalibratedClassifierCV | HistGradientBoostingClassifier:
    if spec.estimator in ("boosting", "boosting_uncalibrated"):
        trees = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)
        if spec.estimator == "boosting_uncalibrated":
            return trees
        # First run, kept so its numbers stay on record. cv=3 fits three copies of the trees, each on
        # two thirds of the window, and a sigmoid calibrator for each on the held-out third. It scored
        # far worse than the same trees without it, so it is not a fair test of boosted trees.
        return CalibratedClassifierCV(trees, method="sigmoid", cv=3)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def coefficients(model: Pipeline, spec: ModelSpec) -> pd.Series:
    """Standardised coefficients of a fitted logistic model."""
    return pd.Series(model[-1].coef_[0], index=feature_names(spec)).round(3)
