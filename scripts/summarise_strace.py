#!/usr/bin/env python3
"""Summarise a run_strace.sh folder: per-repeat I/O times and C_I/O from `strace -f -c -w`.

Usage: python3 summarise_strace.py strace_<jobid> [--model-gb 7.3]

t_read  = wall time in read, pread64, readv  (all threads)
t_write = wall time in write, pwrite64, writev
C_I/O   = 1 - (t_read + t_write) / t_total, with t_total the script's own total under strace.
openat/close/fsync/lseek are reported but, as in the guidebook, not counted in the metric.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

READS = ("read", "pread64", "readv")
WRITES = ("write", "pwrite64", "writev")


def parse(path: Path) -> dict[str, tuple[float, int]]:
    """Rows of a `strace -c` table -> {syscall: (seconds, calls)}."""
    calls: dict[str, tuple[float, int]] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5 and re.match(r"^[0-9.]+$", parts[0]) and parts[-1] != "total":
            calls[parts[-1]] = (float(parts[1]), int(parts[3]))
    return calls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--model-gb", type=float, default=None,
                    help="Model size in GB for the write-speed column (default: read from result JSON)")
    a = ap.parse_args()
    out = a.folder

    rows = []
    for sf in sorted(out.glob("strace_summary_r*.txt")):
        r = sf.stem.split("_r")[-1]
        tj, pj = out / f"result_io_strace_r{r}.json", out / f"result_io_plain_r{r}.json"
        if not (tj.is_file() and pj.is_file()):
            continue
        t, p = json.loads(tj.read_text()), json.loads(pj.read_text())
        calls = parse(sf)
        gb = a.model_gb or max([v for k, v in t.get("outputs_bytes", {}).items() if k.endswith(".pkl")] or [0]) / 1e9
        rows.append(dict(
            r=r, calls=calls, gb=gb,
            rd=sum(calls.get(c, (0, 0))[0] for c in READS),
            wr=sum(calls.get(c, (0, 0))[0] for c in WRITES),
            nw=sum(calls.get(c, (0, 0))[1] for c in WRITES),
            meta=sum(calls.get(c, (0, 0))[0] for c in ("openat", "close", "fsync", "lseek")),
            tot=t["timings_s"]["total"], save=t["timings_s"]["save"],
            ptot=p["timings_s"]["total"], psave=p["timings_s"]["save"],
            hashes={t["correctness"]["forest_hash"], p["correctness"]["forest_hash"]},
        ))
    if not rows:
        raise SystemExit(f"No complete repeats found in {out}")

    med = statistics.median
    print("# strace I/O cross-check (Lustre, owner's full configuration)\n")
    print("| Repeat | Plain: save / total (s) | strace: save / total (s) | t_read (s) | t_write (s) "
          "| write calls | Write speed | C_I/O (strace) |")
    print("|---|---|---|---|---|---|---|---|")
    cs = []
    for x in rows:
        c = 1 - (x["rd"] + x["wr"]) / x["tot"]
        cs.append(c)
        speed = f"{x['gb'] * 1e3 / x['wr']:.0f} MB/s" if x["wr"] > 0 and x["gb"] else "n/a"
        print(f"| {x['r']} | {x['psave']:.2f} / {x['ptot']:.2f} | {x['save']:.2f} / {x['tot']:.2f} "
              f"| {x['rd']:.2f} | {x['wr']:.2f} | {x['nw']} | {speed} | {c:.3f} |")

    rng = lambda key: f"{med([x[key] for x in rows]):.2f} s ({min(x[key] for x in rows):.2f}–{max(x[key] for x in rows):.2f})"
    print("\n| Summary over repeats | Value |\n|---|---|")
    print(f"| Plain save time: median (min–max) | {rng('psave')} |")
    print(f"| strace save time: median (min–max) | {rng('save')} |")
    print(f"| Plain total: median (min–max) | {rng('ptot')} |")
    print(f"| strace total: median (min–max) | {rng('tot')} |")
    print(f"| Save time not spent in write() (pickling etc.), median | "
          f"{med([x['save'] - x['wr'] for x in rows]):.2f} s |")
    print(f"| open/close/fsync/lseek (not in the metric), median | {med([x['meta'] for x in rows]):.2f} s |")
    print(f"| **C_I/O (strace): median (min–max)** | **{med(cs):.3f} ({min(cs):.3f}–{max(cs):.3f})** |")
    ok = len(set().union(*[x["hashes"] for x in rows])) == 1
    print(f"\nForest hash identical across all {2 * len(rows)} runs: **{'yes' if ok else 'NO'}**")
    print("Note: strace -c -w sums wall time over all threads; the save phase is single-threaded.")


if __name__ == "__main__":
    main()
