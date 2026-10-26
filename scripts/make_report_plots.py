#!/usr/bin/env python3
"""Make the two figures for high-level-report.md with the SHAREing `assessmenttemplate` package.

  images/intranode.png  intra-node strong-scaling efficiency with the 80% / 60% critical points
                        (the package's categorical plot, so 1, 2, 4 ... 128 cores are evenly spaced)
  images/summary.png    spiderweb of the rubric scores, radial axis fixed to 0-1

Inputs (CSV, no header):
  results/times_pinned.csv   "cores, median training time in s"
  results/rubric_scores.csv  "rubric, score"

Requirements: the package from https://github.com/SHAREing-DRI/assessment-template
(`pip install .` in that repository) and Python >= 3.12. Version 1.2 of the package uses
f-string syntax that only Python 3.12+ accepts, although its metadata says >= 3.8.

Why a script rather than `high-level-plots.py summary`: the package's spiderweb does not fix the
radial axis, so it autoscales to the largest score (here 0.42), which makes poor scores look large.
The template's placeholder figure uses 0-1, so this script calls the package's own
`summary_to_spiderweb` and only sets the axis to 0-1.

Usage (from the repository root):  python3 scripts/make_report_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
from assessmenttemplate.scaling import (scaling_compute_metrics, scaling_times_crit_80_60,
                                        scaling_times_to_graph, scaling_times_to_markdown)
from assessmenttemplate.summary import summary_to_spiderweb

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"


def intranode() -> None:
    table = pd.read_csv(ROOT / "results" / "times_pinned.csv", header=None, names=["Cores", "Time"])
    scaling_compute_metrics(table)
    fig = scaling_times_to_graph(table, cat_plot=True, critical_points=True)
    fig.savefig(IMAGES / "intranode.png")
    c80, c60 = scaling_times_crit_80_60(table, prop=True)
    print(scaling_times_to_markdown(table))
    print(f"\nCritical points as a proportion of the node: 80% -> {c80}, 60% -> {c60}")


def summary() -> None:
    table = pd.read_csv(ROOT / "results" / "rubric_scores.csv", header=None, names=["Rubric", "Score"],
                        skipinitialspace=True)
    fig = summary_to_spiderweb(table)
    ax = fig.axes[0]
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    fig.savefig(IMAGES / "summary.png")
    print(table.to_string(index=False))


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    intranode()
    summary()
    print(f"\nWrote {IMAGES / 'intranode.png'} and {IMAGES / 'summary.png'}")
