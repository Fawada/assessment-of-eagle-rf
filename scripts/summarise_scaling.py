#!/usr/bin/env python3
"""Summarise a thread-scaling sweep from rf_benchmark.py result files.

Writes, in the sweep directory:
  times.csv    "threads, median fit time (s)" - the input format of the SHAREing template's
               scripts/intranode_times_to_graph.py
  summary.md   table with min/median/max fit time, speed-up, parallel efficiency, peak memory,
               and a correctness check (forest hash identical for every run)

Usage: python summarise_scaling.py <sweep_dir>
Then:  cat <sweep_dir>/times.csv | ./scripts/intranode_times_to_graph.py -gmd --markdown-file=stdout
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sweep = Path(sys.argv[1])
    runs = [json.loads(p.read_text()) for p in sorted(sweep.glob("result_*.json"))]
    if not runs:
        raise SystemExit(f"No result_*.json files in {sweep}")

    fit = defaultdict(list)
    rss = defaultdict(list)
    hashes = set()
    r2s = set()
    for r in runs:
        nj = r["settings"]["n_jobs"]
        fit[nj].append(r["timings_s"]["fit"])
        rss[nj].append(r["memory"]["peak_rss_bytes"])
        hashes.add(r["correctness"]["forest_hash"])
        r2s.add(round(r["correctness"]["test_r2_transformed_space"], 10))

    threads = sorted(fit)
    med = {nj: statistics.median(fit[nj]) for nj in threads}
    base_nj = threads[0]
    base = med[base_nj] * base_nj  # serial-equivalent time if the first point is not 1 thread

    (sweep / "times.csv").write_text("".join(f"{nj}, {med[nj]:.4f}\n" for nj in threads))

    s = runs[0]["settings"]
    d = runs[0]["data"]
    lines = [
        f"# Thread scaling: {s['simulation']} z={s['z']}",
        "",
        f"n_estimators={s['n_estimators']}, max_depth={s['max_depth']}, "
        f"n_train={d['n_train']}, target bins={d['target_bins']}, host={runs[0]['host']}",
        "",
        "| Threads | Runs | Fit min (s) | Fit median (s) | Fit max (s) | Speed-up | Efficiency | Peak RSS (GB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for nj in threads:
        speedup = base / med[nj]
        lines.append(
            f"| {nj} | {len(fit[nj])} | {min(fit[nj]):.2f} | {med[nj]:.2f} | {max(fit[nj]):.2f} "
            f"| {speedup:.2f} | {speedup / nj:.2f} | {max(rss[nj]) / 1e9:.2f} |"
        )
    ok = len(hashes) == 1
    lines += [
        "",
        f"Correctness: forest hash identical across all {len(runs)} runs: **{'yes' if ok else 'NO'}** "
        f"({', '.join(sorted(hashes))}); distinct test R2 values (10 d.p.): {sorted(r2s)}",
    ]
    if base_nj != 1:
        lines.append(f"\nNote: no 1-thread run; speed-up is relative to {base_nj} threads x {base_nj}.")
    (sweep / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
