"""Evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airalerts import evaluate
from airalerts.describe import AXIS, INK, INK_2, MUTED, SURFACE, titled

PERIODS = (("test_period", "Latest 8 weeks", "#2a78d6"), ("since_reform", "Since the 7 Sep reform", "#eb6834"))
MODELS = (("logit_16w", "Final model", "#2a78d6"), ("logit_all_history", "First attempt", "#eb6834"))
LABELS = {
    "const_28d": "Flat 28-day rate",
    "markov_28d": "Recent-onset rate (28 days)",
    "logit_all_history": "Logistic, all history\n(first attempt)",
    "logit_16w": "Logistic, last 16 weeks\n(final)",
}


def plot_skill(report: dict, path: Path) -> None:
    """Brier skill over the best simple forecast of each period, with 95% intervals, per horizon."""
    horizons = list(report["horizons"])
    names = list(LABELS)
    fig, axes = plt.subplots(1, len(horizons), figsize=(10, 4.4), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.72, bottom=0.14, wspace=0.08)
    subtitle = "Brier skill over whichever simple forecast scored best on that period (0 = no better). "
    subtitle += "Lines are 95% intervals from resampling days."
    titled(fig, "Against the best simple forecast, the model's edge is small", subtitle)
    handles = []
    for ax, h in zip(axes, horizons):
        handles = []
        for offset, (period, _, colour) in zip((0.16, -0.16), PERIODS):
            block = report["horizons"][h][period]["forecasts"]
            y = np.arange(len(names))[::-1] + offset
            skill = np.array([100 * block[n]["skill_vs_best_simple"] for n in names])
            low = np.array([100 * block[n]["skill_vs_best_simple_ci95"][0] for n in names])
            high = np.array([100 * block[n]["skill_vs_best_simple_ci95"][1] for n in names])
            ax.hlines(y, low, high, color=colour, linewidth=2)
            handles.append(ax.scatter(skill, y, s=42, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3))
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_title(f"New alert within {h} h", loc="left", fontsize=10, color=INK_2, pad=8)
        ax.set_yticks(np.arange(len(names))[::-1], [LABELS[n] for n in names])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.xaxis.set_major_formatter("{x:+.0f}%")
    style = dict(ncols=2, fontsize=9, labelcolor=INK_2, columnspacing=1.6, handletextpad=0.3)
    names = [label for _, label, _ in PERIODS]
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.005, 0.875), **style)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reliability(predictions: dict[int, pd.DataFrame], path: Path, min_rows: int = 20) -> None:
    """Forecast probability against observed frequency for the final and first models."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(10, 4.4), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.07, right=0.975, top=0.72, bottom=0.14, wspace=0.16)
    subtitle = f"Each dot is a forecast bin with at least {min_rows} origins; "
    subtitle += "above the diagonal means the forecast was too low."
    titled(fig, "Calibration over the latest 8 weeks is rough for both models", subtitle)
    lines = []
    for ax, (h, frame) in zip(axes, predictions.items()):
        lines = []
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
        for name, _, colour in MODELS:
            table = evaluate.reliability(frame, name)
            table = table[table["rows"] >= min_rows]
            marks = dict(marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
            lines += ax.plot(table["forecast"], table["observed"], color=colour, **marks)
        ax.set_title(f"New alert within {h} h", loc="left", fontsize=10, color=INK_2, pad=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("Forecast probability")
        ax.xaxis.set_major_formatter("{x:.0%}")
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.grid(axis="x", visible=True)
    axes[0].set_ylabel("Observed frequency")
    style = dict(ncols=2, fontsize=9, labelcolor=INK_2, columnspacing=1.6)
    names = [label for _, label, _ in MODELS]
    fig.legend(lines, names, loc="upper left", bbox_to_anchor=(0.005, 0.875), **style)
    fig.savefig(path, dpi=160)
    plt.close(fig)
