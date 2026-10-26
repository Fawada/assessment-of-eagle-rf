---
License: Copyright © 2026 Durham University, SHAREing Project, MIT Licensed
Creator: Fawada Qaiser
Contributors: Fawada Qaiser
Summary: High-level performance assessment of the EAGLE gas density profile Random Forest code
Template: SHAREing-DRI/assessment-template, high-level-report.md (Chapter 5 of the guidebook)
---

# High-level assessment: EAGLE gas density profile Random Forest

> **How to use this report as an example.** This is a completed high-level assessment of a
> small, CPU-only Python machine-learning code on COSMA8. It follows the eight sections of the
> SHAREing high-level template. Every number comes from a Slurm job whose scripts are in
> [`scripts/`](scripts/) and whose raw output is in [`results/`](results/). "Method" notes explain
> choices that are not obvious from the guidebook, and Section 6 and the "Tooling notes" record
> what did not work on COSMA8 and how it was worked around, so the next assessor does not lose
> the same time.

Assessor: Fawada Qaiser (DiRAC / Durham University), SHAREing project.
Measurements: 18–20 October 2026, COSMA8. Pre-assessment: [`pre-assessment-report.md`](pre-assessment-report.md).

| Performance dimension | Applies? |
|---|---|
| Core | Yes |
| Intra-node | Yes |
| Inter-node | **No**: no MPI or other multi-node parallelism |
| GPU | **No**: CPU-only code |
| I/O | Yes |

## Summary of results

| Rubric | Metric | Value | Guidebook classification |
|---|---|---|---|
| Core | C_core | **0.092** | poor |
| Intra-node | C⁸⁰_intra / C⁶⁰_intra | **0.063 / 0.50** | poor |
| I/O (Lustre) | C_I/O | **0.42** (median; 0.28–0.55 over 9 runs) | poor |
| Inter-node | – | – | not applicable |
| GPU | – | – | not applicable |

All 41 benchmark runs used for these results produced a bit-for-bit identical trained forest
(hash `5590010f3e7cacbb`), so every timing compares the same computation.

**In one paragraph:** the owner's script spends most of its time writing a 7.3 GB model file
(I/O poor); its training scales efficiently only up to about 8–16 cores and stops getting faster
beyond 64 (intra-node poor); and its floating-point rate is 9% of one core's scalar peak (core
poor). The core result mainly reflects the algorithm: building decision trees is dominated by
comparisons and data movement, not arithmetic, and the processor is busy (1.78 instructions per
cycle) rather than idle. The I/O result is the one the owner can most easily act on.

## 1: Setup details

### Programming model

Python 3.12.4 with scikit-learn 1.9.1 (`RandomForestRegressor`). Training runs as one process
with shared-memory threads managed by joblib; the tree-building code is compiled (Cython) and
releases Python's global interpreter lock, so threads build trees genuinely in parallel. There
is no MPI and no GPU code. Full library versions are in the pre-assessment report.

### Benchmark

The harness [`scripts/rf_benchmark.py`](scripts/rf_benchmark.py) performs the same computation
as the owner's script (see pre-assessment Section 1) with configurable paths and per-phase
timing.

| Setting | Value |
|---|---|
| Simulation | EAGLE `L0100N1504`, z = 0 (`eagle_z0.hdf5`, 85 MB) |
| Haloes | 48,031 training, 12,008 test; 6 features; 9 target bins |
| Forest | 1000 trees, `max_depth=32`, `min_samples_split=2`, `max_features='sqrt'`, `random_state=1` |
| Timed region for core and intra-node | Training (`fit`) only, unless stated |
| I/O configuration | Owner's full behaviour: save model + 4 arrays + PDF plot, 16 threads |

### SLURM configuration

| Item | Value |
|---|---|
| Partition / account | `cosma8` / the assessor's project account |
| Nodes | 1, `--exclusive` (whole node, no other jobs) |
| Node type | 2 × AMD EPYC 7H12 (64 cores each): 128 physical cores, SMT 2 (256 hardware threads), 8 NUMA domains of 16 cores, 1006 GiB RAM |
| Environment | Python venv; `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` |
| Modules | `gnu_comp/14.1.0` + `likwid/5.5.1` (core); `gnu_comp/14.1.0` + `openmpi/5.0.3` + `darshan-runtime/3.4.6` + `darshan-util/3.4.6` (Darshan attempt) |

| Slurm job | Script | Node | What it measured |
|---|---|---|---|
| 12187670 | [`run_highlevel.sh`](scripts/run_highlevel.sh) | m8374 | Topology, pinned scaling sweep, SMT run, NUMA check, I/O on RAM / Lustre / VAST |
| 12188095 | [`run_followup.sh`](scripts/run_followup.sh) | m8415 | Core rubric (LIKWID) |
| 12191857 | `run_followup.sh` with `RUN_CORE=0` | m8145 | Darshan attempt (failed; see Section 6) |
| 12186032 + 12187354 | [`run_scaling.sh`](scripts/run_scaling.sh) | m8138 | Earlier **unpinned** sweep, used for comparison only |

Raw output of the combined jobs: [`results/highlevel_12187670_summary.md`](results/highlevel_12187670_summary.md).

> **Method note: build CPU lists on the compute node.** COSMA8 login nodes number their CPUs
> differently from compute nodes (alternating sockets, no SMT), so pinning lists prepared on a
> login node would be wrong. [`scripts/cpulist.py`](scripts/cpulist.py) reads
> `lscpu -p` on the node where the job runs and returns one hardware thread per physical core.

## 2: Core-level

### Measurement

1. Scalar double-precision peak of one core, as the guidebook prescribes (working set = half
   the 32 kB L1 data cache), on the compute node:
   `likwid-bench -t peakflops -w S0:16kB:1`
2. The benchmark run on **one thread pinned to one core**, whole program, with hardware counters:
   `likwid-perfctr -f -C 0 -g FLOPS_DP python3 rf_benchmark.py 0100 1504 0 --n-jobs 1`
   (and the same with `-g FLOPS_SP`).

### Results

| Quantity | Value |
|---|---|
| Scalar DP peak, one core (R_theoretical) | 9,012.9 MFLOP/s |
| AVX peak, one core (context only) | 51,430.3 MFLOP/s |
| AVX FMA peak, one core (context only) | 55,967.0 MFLOP/s |
| Code, serial (R_observed) | 827.1 MFLOP/s over 118.4 s |
| Total floating-point operations | 97.9 × 10⁹ |
| **C_core = R_observed / R_theoretical** | **0.092 → poor** |
| Against the AVX FMA peak (context only) | 0.015 |

Training was 94% of the measured run (106.9 s of 113.8 s inside the script); the rest is
start-up, reading, prediction and process exit, which the guidebook's whole-program measurement
includes.

> **Method note: do not add FLOPS_DP and FLOPS_SP on this processor.** On AMD Zen 2, LIKWID's
> FLOPS_DP and FLOPS_SP groups both read the same event, `RETIRED_SSE_AVX_FLOPS_ALL`, which
> counts every floating-point operation regardless of precision. The two runs counted
> 97,911,365,675 and 97,911,365,533 operations (a difference of 142). Their *rates* differ by
> 2.5% only because the runs took slightly different times. Adding them would double C_core to
> 0.186. Compare total counts (rate × runtime), not rates; `summarise_highlevel.py` does this.

### Interpretation

The same counter run recorded the instruction and cycle counts, which explain the low score:

| Quantity (serial run) | Value |
|---|---|
| Instructions retired | 699.1 × 10⁹ |
| Cycles | 393.1 × 10⁹ (≈ 3.3 GHz over 118 s; the core was boosting) |
| **Instructions per cycle** | **1.78** |
| Floating-point operations per instruction | 0.14 |
| Floating-point operations per cycle | 0.25 |

The core is busy (1.78 instructions per cycle), but only about one instruction in seven is a
floating-point operation. Building a decision tree means sorting candidate thresholds, comparing
values, branching and moving data; the arithmetic is mainly the running sums used to score each
candidate split. A FLOP-based metric will rate such a code as poor however well it is
implemented. **This is a limitation of the core metric for tree-based machine learning, and it
should be read that way**: the classification is "poor", but it does not on its own indicate a
flaw that the owner could fix. Whether the code is limited by branches, cache misses or memory
latency is a low-level question (Section 8).

## 3: Intra-node

### Measurement

Strong scaling on the fixed benchmark. For each core count p, `n_jobs = p` and the process was
restricted with `taskset` to p physical cores, filling one NUMA domain before the next ("compact"),
never using both hardware threads of a core. Three runs per point; the median training time is
used. Efficiency is calibrated against the 1-thread run, as the guidebook specifies:
E(p) = t(1) / (t(p) × p).

### Results

| Cores | MPI ranks | Threads | Training time, median (s) | min–max (s) | Speed-up | E(p) | E(p), earlier unpinned sweep |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 106.45 | 106.12–106.56 | 1.00 | 1.00 | 1.00 |
| 2 | 1 | 2 | 53.92 | 53.84–53.95 | 1.97 | 0.99 | 0.99 |
| 4 | 1 | 4 | 27.26 | 27.23–27.86 | 3.90 | 0.98 | 0.98 |
| 8 | 1 | 8 | 14.42 | 14.40–14.52 | 7.38 | 0.92 | 0.94 |
| 16 | 1 | 16 | 8.40 | 8.39–8.40 | 12.67 | 0.79 | 0.87 |
| 32 | 1 | 32 | 4.65 | 4.62–4.67 | 22.87 | 0.71 | 0.74 |
| 64 | 1 | 64 | 2.73 | 2.72–2.81 | 38.94 | 0.61 | 0.63 |
| 128 | 1 | 128 | 2.56 | 2.48–2.68 | 41.62 | 0.33 | 0.37 |
| 128 (SMT, 256 threads) | 1 | 256 | 2.58 | | 41.24 | 0.32 per core | |

| Score | Value |
|---|---|
| p⁸⁰ (largest measured p with E ≥ 0.8) | 8 cores |
| p⁶⁰ (largest measured p with E ≥ 0.6) | 64 cores |
| **C⁸⁰_intra = p⁸⁰ / 128** | **0.063** |
| **C⁶⁰_intra = p⁶⁰ / 128** | **0.50** |
| Classification | **poor** (C⁸⁰ < 0.8 and C⁶⁰ < 0.6) |

Graph input for the template's `intranode_times_to_graph.py`:
[`results/times_pinned.csv`](results/times_pinned.csv).

### Interpretation

- **Robustness of the classification.** p⁸⁰ lies between 8 and 16 cores (16 cores gives 0.79,
  just under the threshold) and p⁶⁰ between 64 and 128. Measuring intermediate core counts would
  refine C⁸⁰ and C⁶⁰ but cannot change the classification: "good" would need 80% efficiency on
  at least 103 cores (0.8 × 128 = 102.4), and "not particularly good" 60% efficiency on at least 77.
- **Training stops getting faster at about 2.5 s.** 128 cores is only 7% faster than 64, and
  using all 256 hardware threads gains nothing over 128.
- **Pinning helps.** Absolute training times were 3–12% shorter than in the earlier unpinned
  sweep at every core count (e.g. 106.5 s vs 121.0 s serial, 2.73 s vs 3.01 s on 64 cores). The
  pinned efficiencies look slightly *lower* because the serial reference time improved the most.
  The two sweeps ran on different nodes (m8374 and m8138), so part of the difference may be
  node-to-node variation.
- **NUMA placement is not the main limit.** As an exploratory check, 16 threads placed in one
  NUMA domain took 8.41 s; the same 16 threads spread across all eight domains took 8.69 s, only
  3% slower.

Remaining explanations for the loss of efficiency, **not yet tested** (hypotheses for the
low-level stage):

1. **Load imbalance.** 1000 trees on 128 cores is about 8 trees per core; trees differ in size,
   so some threads finish early and wait.
2. **Fixed overhead.** joblib dispatches trees and gathers results from one Python thread; this
   time does not shrink with more cores and becomes significant when training takes ~2.5 s.

## 4: Inter-node

Not applicable. The code has no MPI or other means of running across nodes. The guidebook
rubric therefore receives no score; this is reported as *not applicable*, not as zero.

## 5: GPU-performance

Not applicable. The code runs only on the CPU.

## 6: I/O

### Measurement

The guidebook's metric is C_I/O = 1 − (t_read + t_write) / t_total, measured with Darshan. On
COSMA8, Darshan could not trace this code (see "Darshan" below), so the I/O time was measured in
two independent ways, each on the owner's full configuration (model + 4 arrays + PDF plot,
16 threads pinned):

1. **Output location comparison** ([`run_highlevel.sh`](scripts/run_highlevel.sh), Part D).
   The same run with outputs written to RAM (`/dev/shm`), Lustre (`/cosma8`, the owner's
   filesystem type) and VAST (`/cosma/apps`), twice each. Saving to RAM removes the filesystem
   from the save time.
2. **`strace`** ([`run_strace.sh`](scripts/run_strace.sh), summarised by
   [`summarise_strace.py`](scripts/summarise_strace.py)). A standard Linux tool that wraps any
   program without changes and reports the time spent inside each system call:
   `strace -f -c -w -e trace=read,write,pread64,pwrite64,readv,writev,openat,close,fsync,lseek python3 ...`.
   t_read and t_write are the wall-clock times inside the read and write calls, summed over all
   threads; this is what Darshan's POSIX counters measure. Runs with and without `strace` were
   alternated three times to separate `strace`'s overhead from run-to-run variation.

> **Method note: use `strace -c -w`.** Without `-w`, `strace -c` reports CPU time inside system
> calls, which leaves out time spent waiting for the filesystem and understates I/O. With `-w`
> it reports wall-clock time, which is what the metric needs.

Each 7.3 GB model file was deleted immediately after it was measured.

### Results

**Output location comparison** (job 12187670, medians of 2 runs):

| Output location | Read (s) | Save (s) | Plot (s) | Total (s) | Model (GB) |
|---|---|---|---|---|---|
| RAM (`/dev/shm`) | 0.01 | 3.18 | 2.54 | 15.82 | 7.30 |
| Lustre (`/cosma8`) | 0.01 | 36.98 | 2.21 | 49.43 | 7.30 |
| VAST (`/cosma/apps`) | 0.01 | 67.01 | 2.22 | 79.78 | 7.30 |

**`strace`, Lustre** (3 alternating repeats; raw output in
[`results/strace_summary.md`](results/strace_summary.md)):

| Repeat | t_read (s) | t_write (s) | write() calls | Write speed | Total under strace (s) | C_I/O |
|---|---|---|---|---|---|---|
| 1 | 0.07 | 30.07 | 2048 | 243 MB/s | 46.93 | 0.358 |
| 2 | 0.07 | 22.28 | 2048 | 328 MB/s | 38.76 | 0.423 |
| 3 | 0.08 | 19.37 | 2048 | 377 MB/s | 35.99 | 0.460 |

The save time measured by the harness minus the write time measured by `strace` is about
**1.1 s** in every repeat: this is the cost of converting the model to bytes (pickling). The 3.2 s
"save" to RAM therefore includes about 2 s of copying the data into memory, not only pickling.

**All 9 measured runs on Lustre** (for runs without `strace`, t_write = save − 1.1 s):

| Run | Write time (s) | Write speed | Total (s) | C_I/O |
|---|---|---|---|---|
| Job 12187670, median of 2 | 35.8 | 204 MB/s | 49.43 | 0.28 |
| `strace` job 1, plain | 35.9 | 203 MB/s | 54.77 | 0.34 |
| `strace` job 1, with `strace` | 13.6 | 537 MB/s | 30.59 | 0.55 |
| `strace` job 2, repeats 1–3, plain | 18.7 / 14.8 / 29.5 | 390 / 493 / 247 MB/s | 33.13 / 28.71 / 43.01 | 0.44 / 0.48 / 0.31 |
| `strace` job 2, repeats 1–3, with `strace` | 30.1 / 22.3 / 19.4 | 243 / 328 / 377 MB/s | 46.93 / 38.76 / 35.99 | 0.36 / 0.42 / 0.46 |

| Score | Value |
|---|---|
| **C_I/O, `strace` (3 runs): median** | **0.42 → poor** (range 0.36–0.46) |
| C_I/O, all 9 Lustre runs | 0.28–0.55, median 0.42; **below 0.6 in every run** |
| On VAST, for comparison (job 12187670) | ≈ 0.20 (write ≈ 66 s, ≈ 110 MB/s) |

### Interpretation

- **The filesystem write dominates, not Python.** Pickling takes about 1.1 s; writing the bytes
  to Lustre took between 13.6 s and 35.9 s. Reading the input takes 0.01–0.08 s.
- **Lustre's write speed varies by a factor of 2.6 between runs** (about 200–540 MB/s), even
  minutes apart on the same node. `/cosma8` is shared by all COSMA8 users, so the write speed
  depends on how busy the storage is at that moment. This is why C_I/O is given as a median
  and range. The classification is unaffected: "poor" in every one of the 9 runs.
- **`strace` is a workable fallback when Darshan cannot be used.** Within each `strace` run, the
  write time it reports accounts for all but ~1.1 s of the save time the harness measured
  independently. Its overhead could not be measured precisely: the median total was 17% higher
  with `strace` (38.8 s vs 33.1 s), but plain runs alone varied by 50% (28.7–43.0 s), and in one
  pair the `strace` run was faster. The overhead is therefore smaller than Lustre's own
  variation for this code, which makes few, large writes (2048 calls of ~3.6 MB).
- **The model is large because the trees are almost fully grown:** 53.7 million nodes for
  48,031 training haloes (see the pre-assessment report).
- **The pre-assessment's figure overstated the owner's I/O cost.** The baseline run wrote to
  VAST (68.8 s of saving); on Lustre, the owner's filesystem type, saving takes 15–37 s.
- **Writing at 200–540 MB/s is modest for Lustre.** A single large file written by one process
  is probably limited by being striped over a single storage target. Testing wider striping
  (`lfs setstripe -c`) is a low-level question; the owner's simplest option is to ask whether
  the full model needs saving on every run.

### Darshan

The guidebook's tool was tried and documented; it could not be used for this code on COSMA8:

| Attempt | Result | Workaround |
|---|---|---|
| `module load darshan-runtime` | Refuses: needs a compiler and an MPI module | Load `gnu_comp/14.1.0` and `openmpi/5.0.3` first |
| `LD_PRELOAD=libdarshan.so python3 ...` | Python crashes: `undefined symbol: H5get_libversion`. This Darshan build has HDF5 support, and pip's `h5py` uses its own bundled HDF5 library | `DARSHAN_MOD_DISABLE=H5F,H5D` lets Python start and write files |
| `DARSHAN_LOGFILE=<file>` | Python's own log is lost: Python starts a helper process (`uname -p`) that inherits Darshan and takes the file name first | Use `DARSHAN_LOGPATH=<dir>` so each process writes its own log |
| `DARSHAN_LOGPATH=<dir>` | "unable to create log file" | Create the dated subfolders (`<dir>/YYYY/M/D`) beforehand; Darshan does not |
| All workarounds, plain Python writing a file | **Works**: 1,000,010 bytes written recorded | |
| All workarounds, Python **opening an HDF5 file** with `h5py` | **Crashes** with the same `H5get_libversion` error | None found |

Making Darshan work would need an HDF5 library compatible with both Darshan and pip's `h5py`.
That is a question for the COSMA support team; `strace` and the output-location comparison above
measure the same quantity without it.

## 7: Summary

| Rubric | Score | Classification |
|---|---|---|
| Core | C_core = 0.092 | poor (expected for tree-based ML; see Section 2) |
| Intra-node | C⁸⁰ = 0.063, C⁶⁰ = 0.50 | poor |
| I/O | C_I/O = 0.42 (0.28–0.55 over 9 runs) | poor in every run |
| Inter-node | – | not applicable |
| GPU | – | not applicable |

Spider-diagram input for the template's `summary.py`:
[`results/rubric_scores.csv`](results/rubric_scores.csv)
(`Core, 0.092` / `I/O, 0.423` / `Intra-node, 0.063`).

## 8: Outcome

All three applicable rubrics are classified as poor, but they are not equally significant or
actionable.

| Rubric | Significance for the owner | Suggested low-level work |
|---|---|---|
| **I/O** | **Highest.** Writing the model takes 44–72% of a run on Lustre, depending on how busy the filesystem is, and is the largest single cost | Ask whether the full model must be saved each run; test Lustre striping (`lfs setstripe`) and writing to node-local memory followed by one copy; measure with Darshan if COSMA can provide a compatible build |
| **Intra-node** | Moderate. Runs are short (8.4 s training at the owner's 16 cores), so better scaling saves seconds per run; it matters more if the owner trains much larger forests | Test the load-imbalance and fixed-overhead hypotheses: time individual trees, profile the dispatch thread (`py-spy`), repeat the sweep with more trees |
| **Core** | Low as a target. The metric undervalues tree-building, and the code is scikit-learn's library code, which the owner would not normally change | If pursued: LIKWID groups for branches, caches and memory to show what limits the core |

### Advice for the code owner (not measured findings)

1. Saving the 7.3 GB model dominates each run. If it is only needed occasionally, saving it
   less often is the largest available speed-up.
2. 16 threads is a sensible choice: efficiency is 0.79 there; 64 threads trains three times
   faster but at 0.61 efficiency.
3. The script crashes on COSMA8 compute nodes at the plotting step (LaTeX is not installed);
   see the pre-assessment report.

### Methodology findings (for the SHAREing team)

1. **The FLOP-based core metric is not informative for tree-based machine learning.** Instructions
   per cycle (here 1.78) and the floating-point share of instructions (0.14) give a fairer picture
   of whether the core is being used.
2. **On some AMD processors, LIKWID's FLOPS_DP and FLOPS_SP read the same counter**, so the
   guidebook's advice to measure both must not lead to adding them. According to LIKWID's own
   group definitions (source checked October 2026), both groups read the all-precision event
   `RETIRED_SSE_AVX_FLOPS_ALL` on **Zen 2, Zen 3, Zen 4 and Zen 4c** (EPYC 7002/7003/9004/97x4,
   including COSMA8's EPYC 7H12). First-generation Zen, Zen 5 and recent Intel Xeons
   (Skylake-SP, Ice Lake, Sapphire Rapids) have separate double- and single-precision events,
   so adding is correct there. A check that works on any machine: compare the event names the
   two groups list (`likwid-perfctr -g FLOPS_DP -H`, or the counter table in the output). The
   same name in both means the same count, so use one.
3. **COSMA8's Darshan cannot trace Python codes that read HDF5 through pip's `h5py`.**
   `strace -f -c -w` is a workable black-box substitute: it measures the time inside read and
   write calls for any program, and agreed with an independent measurement of the save time.
4. **I/O on a shared filesystem needs repeated runs.** Lustre's write speed varied by a factor
   of 2.6 between runs minutes apart, so a single run can misstate C_I/O by ±0.15. Reporting a
   median and range over several runs is advisable.
5. **Thread placement matters for the scaling rubric**: pinning shortened every run by 3–12%,
   and `nproc` reports hardware threads (256) rather than cores (128) on SMT nodes.

## Tooling notes for COSMA8

| Problem | Fix |
|---|---|
| `likwid` module will not load | `module load gnu_comp/14.1.0 likwid/5.5.1` |
| `darshan-runtime` module will not load | `module load gnu_comp/14.1.0 openmpi/5.0.3 darshan-runtime/3.4.6 darshan-util/3.4.6` |
| Darshan crashes Python | `DARSHAN_MOD_DISABLE=H5F,H5D` (still fails if the code opens HDF5 files via `h5py`) |
| Darshan log missing or "unable to create log file" | Use `DARSHAN_LOGPATH=<dir>` and create `<dir>/YYYY/M/D` first |
| Darshan unusable (e.g. Python + `h5py`) | `strace -f -c -w -e trace=read,write,...` around the program (`scripts/run_strace.sh`) |
| `strace -c` shows tiny I/O times | Add `-w` for wall-clock time; plain `-c` reports CPU time only |
| `peakflops` measured on a login node | Not valid: login nodes are a different processor. Measure inside the job |
| Pinning lists | Build on the compute node (`scripts/cpulist.py`) |
| Job scripts must be submitted from the folder containing `rf_benchmark.py` | The scripts locate the harness relative to the submission folder |

## Reproducing this assessment

```bash
cd scripts
source ~/venvs/eagle-rf/bin/activate
mkdir -p /cosma8/data/<project>/<user>/eagle-rf-io /cosma/apps/<site>/<user>/eagle-rf-io
export EAGLE_DATA_ROOT=/path/to/eagle_data      # the owner's data directory (not published)
# edit --account and the LUSTRE_DIR / VAST_DIR defaults in run_highlevel.sh
# (the scripts stop with a message if any CHANGE_ME placeholder is left), then:
sbatch run_highlevel.sh                      # ~25 min: topology, core, scaling, NUMA, I/O, Darshan attempt
python3 summarise_highlevel.py highlevel_<jobid> --unpinned <optional earlier unpinned sweep>
```

To rerun only the LIKWID and Darshan parts into an existing results folder:

```bash
TARGET=highlevel_<jobid> sbatch run_followup.sh            # core + Darshan
RUN_CORE=0 TARGET=highlevel_<jobid> sbatch run_followup.sh # Darshan only
```

The `strace` I/O cross-check (about 6 minutes, 3 alternating repeats):

```bash
sbatch run_strace.sh
python3 summarise_strace.py strace_<jobid>     # also run automatically at the end of the job
```
