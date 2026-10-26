#!/usr/bin/env python3
"""Summarise a run_highlevel.sh output folder into the guidebook's high-level metrics.

Usage: python3 summarise_highlevel.py highlevel_<jobid> [--unpinned scaling_<jobid>]

Writes into the folder:
  highlevel_summary.md   tables and the three rubric scores with their guidebook classification
  times_pinned.csv       "cores, median fit time" for the template's intranode_times_to_graph.py
  rubric_scores.csv      "rubric, score" for the template's summary.py (spider diagram)

Metric definitions (guidebook Chapter 5):
  Core        C_core  = measured MFLOP/s of a serial run / scalar peak MFLOP/s of one core
  I/O         C_I/O   = 1 - (t_read + t_write) / t_total
  Intra-node  E(p)    = t_serial / (t(p) * p);  p80 = largest p with E >= 0.8, p60 likewise;
              C80 = p80 / cores per node,  C60 = p60 / cores per node
Thresholds: >= 0.8 good, 0.6-0.8 not particularly good, < 0.6 poor (intra-node uses C80 and C60).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

NUM = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"


def load_results(folder: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(folder.glob("result_*.json"))] if folder.is_dir() else []


def median(xs):
    return statistics.median(xs) if xs else None


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def classify(c):
    if c is None:
        return "not measured"
    if c >= 0.8:
        return "good"
    if c >= 0.6:
        return "not particularly good"
    return "poor"


# ------------------------------------------------------------------ core
def read_peak(path: Path):
    if not path.is_file():
        return None
    m = re.findall(r"MFlops/s:\s*" + NUM, path.read_text())
    return float(m[-1]) if m else None


def read_perfctr(path: Path, prefix: str):
    """Return (MFLOP/s, runtime s) from a likwid-perfctr single-core table, or (None, None)."""
    if not path.is_file():
        return None, None
    text = path.read_text()
    rate = re.search(r"^\|\s*" + prefix + r" \[MFLOP/s\]\s*\|\s*" + NUM, text, re.M)
    if rate is None:  # metric names differ between processors; fall back to the first MFLOP/s row
        rate = re.search(r"^\|[^|\n]*\[MFLOP/s\]\s*\|\s*" + NUM, text, re.M)
    rt = re.search(r"^\|\s*Runtime \(RDTSC\) \[s\]\s*\|\s*" + NUM, text, re.M)
    return (float(rate.group(1)) if rate else None, float(rt.group(1)) if rt else None)


def core_section(out: Path) -> tuple[list[str], float | None]:
    d = out / "core"
    peak = read_peak(d / "peak_scalar.txt")
    peak_avx = read_peak(d / "peak_avx.txt")
    peak_fma = read_peak(d / "peak_avx_fma.txt")
    dp, dp_rt = read_perfctr(d / "perfctr_FLOPS_DP.txt", "DP")
    sp, sp_rt = read_perfctr(d / "perfctr_FLOPS_SP.txt", "SP")
    lines = ["## Core rubric", "",
             "| Quantity | Value |", "|---|---|",
             f"| Scalar DP peak, one core (`likwid-bench -t peakflops`) | {fmt(peak, 1)} MFLOP/s |",
             f"| AVX peak, one core (context only) | {fmt(peak_avx, 1)} MFLOP/s |",
             f"| AVX FMA peak, one core (context only) | {fmt(peak_fma, 1)} MFLOP/s |",
             f"| Code, serial, FLOPS_DP | {fmt(dp, 2)} MFLOP/s over {fmt(dp_rt, 1)} s |",
             f"| Code, serial, FLOPS_SP | {fmt(sp, 2)} MFLOP/s over {fmt(sp_rt, 1)} s |"]
    c_core = None
    note = ""
    if peak and dp is not None:
        total = dp
        if sp is not None:
            # Compare TOTAL operation counts (rate x runtime), not rates: two runs of the same
            # program differ slightly in runtime, so their rates differ even when counts match.
            dp_ops = dp * dp_rt if dp_rt else None
            sp_ops = sp * sp_rt if sp_rt else None
            if dp_ops and sp_ops and abs(sp_ops - dp_ops) / max(dp_ops, sp_ops) < 0.01:
                note = (f"FLOPS_DP and FLOPS_SP count the same total ({dp_ops / 1e3:,.1f} vs "
                        f"{sp_ops / 1e3:,.1f} GFLOP, within 1%): on this processor both LIKWID groups "
                        "read the same all-precision counter, so they are NOT added. C_core uses the "
                        "FLOPS_DP run.")
            else:
                total = dp + sp
                note = ("FLOPS_DP and FLOPS_SP count different totals, so they are added "
                        "(each from its own serial run).")
        c_core = total / peak
        lines.append(f"| **C_core** = {fmt(total, 2)} / {fmt(peak, 1)} | **{c_core:.4f}** ({classify(c_core)}) |")
    else:
        lines.append("| **C_core** | not computed (missing peak or counter data; see core/*.txt) |")
    lines += ["", note] if note else []
    return lines, c_core


# ------------------------------------------------------------------ intra-node
def intra_section(out: Path, ncores: int | None, unpinned: Path | None):
    runs = load_results(out / "scaling_pinned")
    fit = defaultdict(list)
    for r in runs:
        fit[r["settings"]["n_jobs"]].append(r["timings_s"]["fit"])
    lines = ["## Intra-node rubric (threads pinned to physical cores, compact)", ""]
    if 1 not in fit:
        return lines + ["No serial (1-thread) run found; cannot compute E(p)."], None, None, runs
    t1 = median(fit[1])
    ps = sorted(fit)
    ncores = ncores or max(ps)
    unp = {}
    if unpinned:
        for r in load_results(unpinned):
            unp.setdefault(r["settings"]["n_jobs"], []).append(r["timings_s"]["fit"])
    lines += ["| Cores | Runs | Fit median (s) | min–max (s) | Speed-up | E(p) | Unpinned E(p) |",
              "|---|---|---|---|---|---|---|"]
    eff = {}
    for p in ps:
        m = median(fit[p])
        eff[p] = t1 / (m * p)
        ue = ""
        if unp.get(p) and unp.get(1):
            ue = f"{median(unp[1]) / (median(unp[p]) * p):.2f}"
        lines.append(f"| {p} | {len(fit[p])} | {m:.2f} | {min(fit[p]):.2f}–{max(fit[p]):.2f} "
                     f"| {t1 / m:.2f} | {eff[p]:.2f} | {ue} |")
    p80 = max([p for p in ps if eff[p] >= 0.8], default=0)
    p60 = max([p for p in ps if eff[p] >= 0.6], default=0)
    c80, c60 = p80 / ncores, p60 / ncores
    if c80 >= 0.8:
        verdict = "good"
    elif c60 >= 0.6:
        verdict = "not particularly good"
    else:
        verdict = "poor"
    lines += ["",
              f"p80 = {p80} cores, p60 = {p60} cores, cores per node = {ncores} "
              f"(only measured core counts can qualify; the true thresholds lie between measured points).",
              "",
              f"**C80_intra = {c80:.3f}, C60_intra = {c60:.3f} → shared-memory scaling is {verdict}.**"]
    smt = load_results(out / "scaling_pinned" / "smt")
    if smt:
        m = median([r["timings_s"]["fit"] for r in smt])
        n = smt[0]["settings"]["n_jobs"]
        lines += ["", f"All {n} hardware threads (SMT, unpinned): fit median {m:.2f} s, "
                      f"speed-up {t1 / m:.2f} (= {t1 / m / ncores:.2f} per physical core)."]
    (out / "times_pinned.csv").write_text("".join(f"{p}, {median(fit[p]):.4f}\n" for p in ps))
    return lines, c80, c60, runs + smt


def numa_section(out: Path):
    runs = load_results(out / "numa_check")
    by = defaultdict(list)
    for r in runs:
        by[r["tag"].split("16_")[0]].append(r["timings_s"]["fit"])
    if not by:
        return [], runs
    lines = ["## NUMA check, 16 threads (exploratory, not a rubric)", "",
             "| Placement | Runs | Fit median (s) |", "|---|---|---|"]
    for k in ("compact", "spread"):
        if by.get(k):
            lines.append(f"| {k} ({'one NUMA domain' if k == 'compact' else 'all NUMA domains'}) "
                         f"| {len(by[k])} | {median(by[k]):.2f} |")
    return lines, runs


# ------------------------------------------------------------------ I/O
def parse_darshan_total(path: Path) -> dict[str, float]:
    vals = {}
    if path.is_file():
        for m in re.finditer(r"^total_(\w+):\s*" + NUM, path.read_text(), re.M):
            vals[m.group(1)] = float(m.group(2))
    return vals


def io_section(out: Path):
    d = out / "io"
    runs = load_results(d)
    by = defaultdict(list)
    for r in runs:
        tag = r["tag"]
        if tag.startswith("io_") and tag != "io_darshan":
            by[tag.split("_")[1]].append(r)
    nj = runs[0]["settings"]["n_jobs"] if runs else "?"
    lines = [f"## I/O rubric (owner's full configuration: model + arrays + plot, {nj} threads)", ""]
    c_io = None
    if by:
        lines += ["| Output location | Runs | read (s) | save (s) | plot (s) | total (s) | Model GB | "
                  "C_I/O from phases |", "|---|---|---|---|---|---|---|---|"]
        med = {}
        for fs in ("shm", "lustre", "vast"):
            rs = by.get(fs)
            if not rs:
                continue
            g = {k: median([r["timings_s"][k] for r in rs]) for k in ("read", "save", "plot", "total")}
            gb = max(r["outputs_bytes"].get(next((n for n in r["outputs_bytes"] if n.endswith(".pkl")), ""), 0)
                     for r in rs) / 1e9
            c = 1 - (g["read"] + g["save"]) / g["total"]
            med[fs] = (g, c)
            label = {"shm": "RAM (/dev/shm): pickling only", "lustre": "Lustre (/cosma8)",
                     "vast": "VAST (/cosma/apps)"}[fs]
            lines.append(f"| {label} | {len(rs)} | {g['read']:.2f} | {g['save']:.2f} | {g['plot']:.2f} "
                         f"| {g['total']:.2f} | {gb:.2f} | {c:.3f} |")
        if "shm" in med:
            pick = med["shm"][0]["save"]
            lines += ["", f"Saving to RAM takes {pick:.1f} s: this is the cost of converting the model to bytes "
                          "(pickling), with almost no filesystem involved."]
            for fs in ("lustre", "vast"):
                if fs in med:
                    g = med[fs][0]
                    write_only = max(0.0, g["save"] - pick)
                    c_w = 1 - (g["read"] + write_only) / g["total"]
                    if fs == "lustre":
                        c_io = c_w  # headline: filesystem time only (closest to Darshan's t_write)
                    lines.append(f"- {fs}: save {g['save']:.1f} s = pickling {pick:.1f} s + filesystem "
                                 f"{write_only:.1f} s; C_I/O counting only filesystem time = {c_w:.3f}")
        if "lustre" in med and c_io is None:
            c_io = med["lustre"][1]
    dv = parse_darshan_total(d / "darshan_total.txt")
    if dv:
        rd = dv.get("POSIX_F_READ_TIME", 0) + dv.get("STDIO_F_READ_TIME", 0)
        wr = dv.get("POSIX_F_WRITE_TIME", 0) + dv.get("STDIO_F_WRITE_TIME", 0)
        meta = dv.get("POSIX_F_META_TIME", 0) + dv.get("STDIO_F_META_TIME", 0)
        written = dv.get("POSIX_BYTES_WRITTEN", 0) + dv.get("STDIO_BYTES_WRITTEN", 0)
        dj = d / "result_io_darshan.json"
        total = json.loads(dj.read_text())["timings_s"]["total"] if dj.is_file() else None
        lines += ["", "### Darshan (one run, Lustre)", "",
                  "| Counter (POSIX + STDIO) | Value |", "|---|---|",
                  f"| Read time | {rd:.2f} s |", f"| Write time | {wr:.2f} s |",
                  f"| Metadata time (open/close/stat; not in the metric) | {meta:.2f} s |",
                  f"| Bytes written | {written / 1e9:.2f} GB |",
                  f"| Script total (harness) | {fmt(total)} s |"]
        if total:
            c_d = 1 - (rd + wr) / total
            lines.append(f"| **C_I/O (Darshan)** | **{c_d:.3f}** ({classify(c_d)}) |")
            c_io = c_d
        lines += ["", "Darshan's write time counts only time inside write() calls, i.e. excluding pickling, "
                      "so it is the closest match to the guidebook's t_write."]
    elif (d / "module_darshan.txt").is_file():
        lines += ["", "Darshan: no counters parsed (see io/module_darshan.txt and io/darshan_run.log)."]
    return lines, c_io, runs


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--unpinned", type=Path, default=None, help="Earlier unpinned sweep folder, for comparison")
    a = ap.parse_args()
    out = a.folder
    layout = (out / "topology" / "layout.txt")
    m = re.search(r"physical cores:\s*(\d+)", layout.read_text()) if layout.is_file() else None
    ncores = int(m.group(1)) if m else None

    core_lines, c_core = core_section(out)
    intra_lines, c80, c60, intra_runs = intra_section(out, ncores, a.unpinned)
    numa_lines, numa_runs = numa_section(out)
    io_lines, c_io, io_runs = io_section(out)
    core_runs = load_results(out / "core")

    allruns = core_runs + intra_runs + numa_runs + io_runs
    hashes = {r["correctness"]["forest_hash"] for r in allruns}
    ok = len(hashes) == 1

    head = ["# High-level assessment summary", "",
            f"Folder: `{out.name}`; physical cores per node: {ncores}", "",
            "| Rubric | Metric | Value | Guidebook classification |", "|---|---|---|---|",
            f"| Core | C_core | {fmt(c_core, 4)} | {classify(c_core)} |",
            f"| I/O | C_I/O | {fmt(c_io, 3)} | {classify(c_io)} |",
            f"| Intra-node | C80 / C60 | {fmt(c80, 3)} / {fmt(c60, 3)} | "
            f"{'not measured' if c80 is None else ('good' if c80 >= 0.8 else 'not particularly good' if c60 >= 0.6 else 'poor')} |",
            "| GPU | – | – | not applicable (CPU-only code) |",
            "| Inter-node | – | – | not applicable (no MPI) |", "",
            f"Correctness: forest hash identical across all {len(allruns)} runs: **{'yes' if ok else 'NO'}** "
            f"({', '.join(sorted(hashes))})", ""]
    text = "\n".join(head + core_lines + [""] + intra_lines + [""] + numa_lines + [""] + io_lines) + "\n"
    (out / "highlevel_summary.md").write_text(text)
    scores = [("Core", c_core), ("I/O", c_io), ("Intra-node", c80)]
    (out / "rubric_scores.csv").write_text("".join(f"{k}, {v:.3f}\n" for k, v in scores if v is not None))
    print(text)


if __name__ == "__main__":
    main()
