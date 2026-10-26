# High-level assessment summary

Folder: `highlevel_12187670`; physical cores per node: 128

| Rubric | Metric | Value | Guidebook classification |
|---|---|---|---|
| Core | C_core | 0.0918 | poor |
| I/O | C_I/O | 0.252 | poor |
| Intra-node | C80 / C60 | 0.062 / 0.500 | poor |
| GPU | – | – | not applicable (CPU-only code) |
| Inter-node | – | – | not applicable (no MPI) |

Correctness: forest hash identical across all 41 runs: **yes** (5590010f3e7cacbb)

## Core rubric

| Quantity | Value |
|---|---|
| Scalar DP peak, one core (`likwid-bench -t peakflops`) | 9012.9 MFLOP/s |
| AVX peak, one core (context only) | 51430.3 MFLOP/s |
| AVX FMA peak, one core (context only) | 55967.0 MFLOP/s |
| Code, serial, FLOPS_DP | 827.13 MFLOP/s over 118.4 s |
| Code, serial, FLOPS_SP | 847.81 MFLOP/s over 115.5 s |
| **C_core** = 827.13 / 9012.9 | **0.0918** (poor) |

FLOPS_DP and FLOPS_SP count the same total (97.9 vs 97.9 GFLOP, within 1%): on this processor both LIKWID groups read the same all-precision counter, so they are NOT added. C_core uses the FLOPS_DP run.

## Intra-node rubric (threads pinned to physical cores, compact)

| Cores | Runs | Fit median (s) | min–max (s) | Speed-up | E(p) | Unpinned E(p) |
|---|---|---|---|---|---|---|
| 1 | 3 | 106.45 | 106.12–106.56 | 1.00 | 1.00 | 1.00 |
| 2 | 3 | 53.92 | 53.84–53.95 | 1.97 | 0.99 | 0.99 |
| 4 | 3 | 27.26 | 27.23–27.86 | 3.90 | 0.98 | 0.98 |
| 8 | 3 | 14.42 | 14.40–14.52 | 7.38 | 0.92 | 0.94 |
| 16 | 3 | 8.40 | 8.39–8.40 | 12.67 | 0.79 | 0.87 |
| 32 | 3 | 4.65 | 4.62–4.67 | 22.87 | 0.71 | 0.74 |
| 64 | 3 | 2.73 | 2.72–2.81 | 38.94 | 0.61 | 0.63 |
| 128 | 3 | 2.56 | 2.48–2.68 | 41.62 | 0.33 | 0.37 |

p80 = 8 cores, p60 = 64 cores, cores per node = 128 (only measured core counts can qualify; the true thresholds lie between measured points).

**C80_intra = 0.062, C60_intra = 0.500 → shared-memory scaling is poor.**

All 256 hardware threads (SMT, unpinned): fit median 2.58 s, speed-up 41.24 (= 0.32 per physical core).

## NUMA check, 16 threads (exploratory, not a rubric)

| Placement | Runs | Fit median (s) |
|---|---|---|
| compact (one NUMA domain) | 3 | 8.41 |
| spread (all NUMA domains) | 3 | 8.69 |

## I/O rubric (owner's full configuration: model + arrays + plot, 16 threads)

| Output location | Runs | read (s) | save (s) | plot (s) | total (s) | Model GB | C_I/O from phases |
|---|---|---|---|---|---|---|---|
| RAM (/dev/shm): pickling only | 2 | 0.01 | 3.18 | 2.54 | 15.82 | 7.30 | 0.799 |
| Lustre (/cosma8) | 2 | 0.01 | 36.98 | 2.21 | 49.43 | 7.30 | 0.252 |
| VAST (/cosma/apps) | 2 | 0.01 | 67.01 | 2.22 | 79.78 | 7.30 | 0.160 |

Saving to RAM takes 3.2 s: this is the cost of converting the model to bytes (pickling), with almost no filesystem involved.
- lustre: save 37.0 s = pickling 3.2 s + filesystem 33.8 s; C_I/O counting only filesystem time = 0.316
- vast: save 67.0 s = pickling 3.2 s + filesystem 63.8 s; C_I/O counting only filesystem time = 0.200

Darshan: no counters parsed (see io/module_darshan.txt and io/darshan_run.log).

---
Notes added afterwards:
- Raw output of summarise_highlevel.py after jobs 12187670 (scaling, NUMA, I/O), 12188095 (core)
  and 12191857 (Darshan attempt). The I/O headline above (0.252) is the version that counts
  pickling as I/O; the report uses the filesystem-only value, 0.316, which the current
  summarise_highlevel.py reports as the headline.
- Darshan failed: `python3: symbol lookup error: .../libdarshan.so: undefined symbol: H5get_libversion`
  when the benchmark opens its HDF5 input with h5py. See high-level-report.md, Section 6.
- Counter detail from core/perfctr_FLOPS_*.txt (same event in both groups):
  FLOPS_DP: RETIRED_INSTRUCTIONS 699099741360, CPU_CLOCKS_UNHALTED 393071446691, RETIRED_SSE_AVX_FLOPS_ALL 97911365675
  FLOPS_SP: RETIRED_INSTRUCTIONS 699073926273, CPU_CLOCKS_UNHALTED 393368711123, RETIRED_SSE_AVX_FLOPS_ALL 97911365533
