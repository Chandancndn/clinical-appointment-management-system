"""Figure helpers for the research experiments: one style, the validated reference palette (light surface).

Categorical slots in fixed order: 1 blue, 2 orange, 3 aqua. Aqua is 2.7:1 on the surface (a validator WARN), so
the relief rule applies: every figure has a legend and its numbers are in the matching results/*.csv table.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898881", "#e6e5e1"
MODEL_COLORS = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834", "hist_gradient_boosting": "#1baf7a"}
MODEL_LABELS = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
                "hist_gradient_boosting": "Gradient boosting"}
VERSION_COLORS = {"uncalibrated": "#eb6834", "calibrated": "#2a78d6"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8, linestyle="-")  # solid hairlines, never dashed
    ax.set_axisbelow(True)


def line_figure(path: Path, title: str, subtitle: str, note: str, series: list[dict], xlabel: str, ylabel: str,
                reference: dict | None = None, limits=((0, 1), (0, 1))) -> None:
    """Lines (and optional markers with whiskers) on a percent scale. series: dicts with label, x, y, color,
    and optionally markers=True and yerr=(lower, upper) distances."""
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200, facecolor=SURFACE)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.76, bottom=0.17)
    fig.text(0.13, 0.955, title, fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.13, 0.905, subtitle, fontsize=9, color=INK_SECONDARY, ha="left", va="top", linespacing=1.5)
    fig.text(0.13, 0.025, note, fontsize=7.5, color=INK_MUTED, ha="left", va="bottom", linespacing=1.5)
    _style(ax)
    if reference:
        ax.plot(reference["x"], reference["y"], color=INK_MUTED, linewidth=1.0, zorder=1)
    for s in series:
        ax.plot(s["x"], s["y"], color=s["color"], linewidth=1.6, solid_capstyle="round", zorder=3,
                marker="o" if s.get("markers") else None, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        if s.get("yerr") is not None:
            ax.errorbar(s["x"], s["y"], yerr=s["yerr"], fmt="none", ecolor=s["color"], elinewidth=0.9, capsize=2, zorder=2)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    handles = [Line2D([0], [0], color=s["color"], linewidth=2, label=s["label"]) for s in series]
    if reference and reference.get("label"):
        handles.append(Line2D([0], [0], color=INK_MUTED, linewidth=1, label=reference["label"]))
    fig.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.125, 0.835), ncol=2, fontsize=8.5,
               labelcolor=INK_SECONDARY, handlelength=1.6, columnspacing=1.6)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
