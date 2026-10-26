#!/usr/bin/env python3
"""Print a comma-separated list of logical CPU IDs for pinning, built from the node's real layout.

Reads `lscpu -p=CPU,CORE,SOCKET,NODE` on the node where it runs (compute and login nodes can
number their CPUs differently, so never prepare these lists on a login node).

Modes
  compact N   N physical cores, one hardware thread each, filling NUMA domain 0 first, then 1, ...
  spread N    N physical cores spread evenly over all NUMA domains (round-robin)
  all         every logical CPU, including SMT siblings
  info        print the layout summary

Examples
  taskset -c "$(python3 cpulist.py compact 16)" python3 rf_benchmark.py ...
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict


def read_layout() -> list[tuple[int, int, int, int]]:
    out = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
    rows = []
    for line in out.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu, core, socket, node = (int(x) if x != "" else 0 for x in line.split(","))
        rows.append((cpu, core, socket, node))
    return rows


def physical_cores(rows) -> list[tuple[int, int, int]]:
    """One entry per physical core: (node, core, first logical CPU of that core)."""
    first: dict[tuple[int, int], tuple[int, int, int]] = {}
    for cpu, core, socket, node in sorted(rows):
        key = (socket, core)
        if key not in first:
            first[key] = (node, core, cpu)
    return sorted(first.values())


def compact(rows, n: int) -> list[int]:
    cores = physical_cores(rows)
    if n > len(cores):
        raise SystemExit(f"Requested {n} cores but the node has {len(cores)} physical cores")
    return [cpu for _, _, cpu in cores[:n]]


def spread(rows, n: int) -> list[int]:
    by_node: dict[int, list[int]] = defaultdict(list)
    for node, _, cpu in physical_cores(rows):
        by_node[node].append(cpu)
    nodes = sorted(by_node)
    picked: list[int] = []
    i = 0
    while len(picked) < n:
        node = nodes[i % len(nodes)]
        idx = i // len(nodes)
        if idx < len(by_node[node]):
            picked.append(by_node[node][idx])
        i += 1
        if i > 10 * (n + len(rows)):
            raise SystemExit(f"Cannot spread {n} cores over this node")
    return sorted(picked)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = read_layout()
    mode = sys.argv[1]
    if mode == "info":
        cores = physical_cores(rows)
        nodes = sorted({n for n, _, _ in cores})
        print(f"logical CPUs: {len(rows)}  physical cores: {len(cores)}  "
              f"SMT: {len(rows) // max(1, len(cores))}  NUMA domains: {len(nodes)}")
        for node in nodes:
            cpus = [c for n, _, c in cores if n == node]
            print(f"  NUMA {node}: {len(cpus)} cores, first CPUs {cpus[:4]}...")
        return
    if mode == "all":
        ids = sorted(r[0] for r in rows)
    elif mode in ("compact", "spread") and len(sys.argv) == 3:
        n = int(sys.argv[2])
        ids = compact(rows, n) if mode == "compact" else spread(rows, n)
    else:
        raise SystemExit(__doc__)
    print(",".join(str(i) for i in ids))


if __name__ == "__main__":
    main()
