"""Descriptive tables and figures for the alert history."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airalerts.data import KYIV_TZ, REFORM
from airalerts.grid import DAY, HOUR, covered_seconds, epoch_seconds

SURFACE, INK, INK_2, MUTED, GRIDLINE, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, BLUE_LIGHT = "#2a78d6", "#9ec5f4"
# Name, first local day, colour. A regime keeps its colour in every figure.
REGIMES = (
    ("2022", "2022-02-28", "#eda100"),
    ("2023–2025", "2023-01-01", "#1baf7a"),
    ("Jan–Jul 2026", "2026-01-01", "#eb6834"),
    ("Aug 2026 onward", "2026-08-01", "#2a78d6"),
)
REGIME_NAMES = [name for name, _, _ in REGIMES]
REGIME_EDGES = pd.DatetimeIndex([pd.Timestamp(day, tz=KYIV_TZ) for _, day, _ in REGIMES]).tz_convert("UTC")
NIGHT_HOURS = (23, 0, 1, 2, 3, 4, 5, 6)

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
    }
)


def regime_of(times: pd.DatetimeIndex) -> np.ndarray:
    index = np.searchsorted(epoch_seconds(REGIME_EDGES), epoch_seconds(times), side="right") - 1
    return np.array(REGIME_NAMES)[np.maximum(index, 0)]


def titled(fig, title: str, subtitle: str) -> None:
    fig.text(0.01, 0.975, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.925, subtitle, fontsize=10, color=INK_2, va="top")


def monthly_table(events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Alert starts and hours under alert per Kyiv calendar month."""
    first = events["start"].min().tz_convert(KYIV_TZ).to_period("M").start_time.tz_localize(KYIV_TZ)
    months = pd.date_range(first, as_of.tz_convert(KYIV_TZ), freq="MS")
    edges = months.tz_convert("UTC").append(pd.DatetimeIndex([as_of]))
    start_s, end_s, edge_s = epoch_seconds(events["start"]), epoch_seconds(events["end"]), epoch_seconds(edges)
    return pd.DataFrame(
        {
            "month": months.tz_localize(None),
            "alerts": np.diff(np.searchsorted(start_s, edge_s, side="left")),
            "hours_under_alert": np.diff(covered_seconds(start_s, end_s, edge_s)) / HOUR,
            "days_covered": np.diff(edge_s) / DAY,
        }
    )


def hour_profile(events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Per regime and Kyiv clock hour: mean alert starts and mean share of the hour under alert."""
    edges = pd.date_range(events["start"].min().floor("h"), as_of.floor("h"), freq="h", unit="s")
    start_s, end_s, edge_s = epoch_seconds(events["start"]), epoch_seconds(events["end"]), epoch_seconds(edges)
    hours = pd.DataFrame(
        {
            "regime": regime_of(edges[:-1]),
            "local_hour": edges[:-1].tz_convert(KYIV_TZ).hour,
            "starts": np.diff(np.searchsorted(start_s, edge_s, side="left")),
            "share_under_alert": np.diff(covered_seconds(start_s, end_s, edge_s)) / HOUR,
        }
    )
    return hours.groupby(["regime", "local_hour"], as_index=False).mean()


def burstiness(intervals: np.ndarray) -> float:
    """Goh-Barabasi coefficient: -1 regular, 0 Poisson-like, towards 1 bursty."""
    mean, std = intervals.mean(), intervals.std()
    return float((std - mean) / (std + mean))


def regime_summary(events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    closed = events[events["end"].notna()].copy()
    closed["regime"] = regime_of(pd.DatetimeIndex(closed["start"]))
    closed["duration_min"] = (closed["end"] - closed["start"]).dt.total_seconds() / 60
    closed["gap_h"] = (closed["start"].shift(-1) - closed["end"]).dt.total_seconds() / HOUR
    closed["between_starts_h"] = closed["start"].diff().dt.total_seconds() / HOUR
    closed["night"] = closed["start"].dt.tz_convert(KYIV_TZ).dt.hour.isin(NIGHT_HOURS)
    start_s, end_s = epoch_seconds(events["start"]), epoch_seconds(events["end"])
    bounds = epoch_seconds(REGIME_EDGES.append(pd.DatetimeIndex([as_of])))
    covered = np.diff(covered_seconds(start_s, end_s, bounds))
    rows = []
    for i, name in enumerate(REGIME_NAMES):
        part = closed[closed["regime"] == name]
        days = (bounds[i + 1] - bounds[i]) / DAY
        rows.append(
            {
                "regime": name,
                "days": round(days),
                "alerts": len(part),
                "alerts_per_day": round(len(part) / days, 2),
                "time_under_alert_pct": round(100 * covered[i] / (bounds[i + 1] - bounds[i]), 1),
                "median_duration_min": round(part["duration_min"].median(), 1),
                "p90_duration_min": round(part["duration_min"].quantile(0.9), 1),
                "median_quiet_gap_h": round(part["gap_h"].median(), 1),
                "starts_23_to_07_pct": round(100 * part["night"].mean(), 1),
                "burstiness": round(burstiness(part["between_starts_h"].dropna().to_numpy()), 2),
            }
        )
    return pd.DataFrame(rows)


def threat_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Durations since the reform, by the causes attached to the finished alert."""
    part = events[(events["start"] >= REFORM) & events["end"].notna()].copy()
    part["duration_min"] = (part["end"] - part["start"]).dt.total_seconds() / 60
    durations = part.groupby("threat")["duration_min"]
    table = durations.agg(alerts="count", median_min="median", p90_min=lambda d: d.quantile(0.9))
    names = {"drone": "drone only (yellow)", "missile": "missile, ballistic or massive drone (red)"}
    return table.round(1).rename(index=names).reset_index()


def plot_monthly(monthly: pd.DataFrame, as_of: pd.Timestamp, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.07, hspace=0.32)
    title = "September 2026 was Kyiv's heaviest month for alerts since March 2022"
    titled(fig, title, "Per calendar month, Kyiv time. The last bar is a partial month.")
    partial = monthly["days_covered"] < 27
    colours = np.where(partial, BLUE_LIGHT, BLUE)
    reform = REFORM.tz_convert(KYIV_TZ).tz_localize(None)
    panels = (("alerts", "Alert starts", "{:.0f}"), ("hours_under_alert", "Hours under alert", "{:.0f} h"))
    for ax, (column, label, fmt) in zip(axes, panels):
        ax.bar(monthly["month"] + pd.Timedelta(days=15), monthly[column], width=22, color=colours, linewidth=0)
        ax.set_title(label, loc="left", fontsize=10, color=INK_2)
        for _, row in monthly.nlargest(2, column).iterrows():
            at = (row["month"] + pd.Timedelta(days=15), row[column])
            note = dict(xytext=(7, 1), textcoords="offset points", ha="left", fontsize=9, color=INK)
            ax.annotate(fmt.format(row[column]), at, **note)
        ax.axvline(reform, color=MUTED, linewidth=0.8)
        ax.set_ylim(0, monthly[column].max() * 1.15)
    note = dict(textcoords="offset points", fontsize=9, color=INK_2)
    at = (reform, axes[0].get_ylim()[1])
    axes[0].annotate("7 Sep 2026: yellow and red levels", at, xytext=(-6, -2), ha="right", va="top", **note)
    last = monthly.iloc[-1]
    partial_label = f"{last['month']:%b}: first {last['days_covered']:.0f} days"
    at = (last["month"] + pd.Timedelta(days=15), last["alerts"] / 2)
    axes[0].annotate(partial_label, at, xytext=(8, 0), ha="left", va="center", **note)
    axes[1].set_xlim(right=monthly["month"].iloc[-1] + pd.Timedelta(days=250))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def regime_legend(fig, lines) -> None:
    style = dict(ncols=4, fontsize=9, labelcolor=INK_2, handlelength=1.6, columnspacing=1.6)
    fig.legend(lines, REGIME_NAMES, loc="upper left", bbox_to_anchor=(0.005, 0.895), **style)


def plot_hour_profile(profile: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.12, wspace=0.16)
    title = "Since August 2026, alerts come around the clock, not mostly at night"
    titled(fig, title, "By Kyiv clock hour, averaged over the days of each period.")
    panels = (
        ("starts", "Alert starts in that clock hour, per day", 1.0, "{x:.2f}"),
        ("share_under_alert", "Share of the hour under alert", 100.0, "{x:.0f}%"),
    )
    lines = []
    for ax, (column, label, scale, fmt) in zip(axes, panels):
        lines = []
        for name, _, colour in REGIMES:
            part = profile[profile["regime"] == name]
            marks = dict(marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1)
            lines += ax.plot(part["local_hour"], part[column] * scale, color=colour, **marks)
        ax.set_title(label, loc="left", fontsize=10, color=INK_2, pad=10)
        ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
        ax.tick_params(axis="x", pad=7)
        ax.set_xlim(-0.5, 23.5)
        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(fmt)
    regime_legend(fig, lines)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def survival_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values)
    return values, 100 * (1 - np.arange(1, len(values) + 1) / len(values))


def plot_durations_and_gaps(events: pd.DataFrame, path: Path) -> None:
    closed = events[events["end"].notna()]
    regime = regime_of(pd.DatetimeIndex(closed["start"]))
    duration_min = ((closed["end"] - closed["start"]).dt.total_seconds() / 60).to_numpy()
    gap_h = ((closed["start"].shift(-1) - closed["end"]).dt.total_seconds() / HOUR).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.74, bottom=0.14, wspace=0.18)
    subtitle = "Share of alerts (left) and of quiet gaps (right) that last longer than the value on the x-axis."
    titled(fig, "Alert lengths changed little; the quiet between alerts collapsed", subtitle)
    panels = (
        (
            duration_min,
            "Alert duration",
            [5, 15, 30, 60, 120, 240, 480],
            ["5 min", "15 min", "30 min", "1 h", "2 h", "4 h", "8 h"],
            (3, 600),
        ),
        (
            gap_h,
            "Quiet gap before the next alert",
            [0.25, 1, 4, 12, 24, 72, 168, 720],
            ["15 min", "1 h", "4 h", "12 h", "1 d", "3 d", "1 wk", "30 d"],
            (0.1, 1000),
        ),
    )
    lines = []
    for ax, (values, label, ticks, tick_labels, limits) in zip(axes, panels):
        lines = []
        for name, _, colour in REGIMES:
            x, y = survival_xy(values[(regime == name) & ~np.isnan(values)])
            lines += ax.step(x, y, where="post", color=colour)
        ax.set_xscale("log")
        ax.set_xticks(ticks, tick_labels)
        ax.minorticks_off()
        ax.set_xlim(*limits)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter("{x:.0f}%")
        ax.set_title(label, loc="left", fontsize=10, color=INK_2, pad=12)
        ax.tick_params(axis="x", pad=7)
    regime_legend(fig, lines)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(events: pd.DataFrame, as_of: pd.Timestamp, out_dir: Path) -> dict[str, pd.DataFrame]:
    figures, tables = out_dir / "figures", out_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    results = {
        "monthly": monthly_table(events, as_of),
        "hour_profile": hour_profile(events, as_of),
        "regime_summary": regime_summary(events, as_of),
        "threat_summary": threat_summary(events),
    }
    for name, table in results.items():
        table.to_csv(tables / f"{name}.csv", index=False)
    plot_monthly(results["monthly"], as_of, figures / "monthly.png")
    plot_hour_profile(results["hour_profile"], figures / "hour_profile.png")
    plot_durations_and_gaps(events, figures / "durations_gaps.png")
    return results
