# Assessment of the EAGLE gas density profile Random Forest

Performance assessment of a Python/scikit-learn Random Forest code that predicts halo gas density
profiles in the EAGLE simulations from global galaxy and halo properties. Carried out on COSMA8
following the SHAREing performance assessment guidebook and the
[SHAREing assessment template](https://github.com/SHAREing-DRI/assessment-template).

**Status (2026-10-20): pre-assessment and high-level assessment complete.**

**Read first:** [`high-level-report.md`](high-level-report.md) for the results, and
[`pre-assessment-report.md`](pre-assessment-report.md) for the benchmark setup. Both are written to
double as worked examples for assessing other small, CPU-only Python codes.

## High-level scores

| Rubric | Score | Classification |
|---|---|---|
| Core | C_core = 0.092 | poor (expected for tree-based ML: the core is busy, 1.78 instructions/cycle, but only 1 in 7 instructions is floating-point) |
| Intra-node | C80 = 0.063, C60 = 0.50 | poor (efficient up to 8–16 cores; no gain beyond 64) |
| I/O | C_I/O = 0.42 (0.28–0.55 over 9 runs) | poor in every run (writing the 7.3 GB model is 44–72% of a run on Lustre) |
| Inter-node, GPU | – | not applicable |

## Key measurements

| | |
|---|---|
| Benchmark | Training on EAGLE `L0100N1504`, z=0: 48,031 haloes, 6 features, 9 target bins, 1000 trees |
| Owner's full run on COSMA8 | 85.5 s, of which **84% is writing outputs** (a 7.3 GB model file) and 10% is training |
| Training, 16 threads (owner's setting) | 8.7 s |
| Training, threads pinned, 1 → 128 cores | 106.5 s → 2.6 s; parallel efficiency 0.79 at 16 cores, 0.61 at 64, 0.33 at 128 |
| Saving the model | ~1.1 s to convert to bytes + 14–36 s to write on Lustre (200–540 MB/s; varies with filesystem load) |
| Peak memory | 8.7 GB (0.8% of a 1006 GiB COSMA8 node) |
| Original script on COSMA8 compute nodes | Would crash at the plotting step: no LaTeX installed |
| Model size | 7.3 GB: 53.7 million tree nodes |
| Reproducibility | Identical trained forest in every run (hash `5590010f3e7cacbb`) |

## Repository layout

| Path | Contents |
|---|---|
| `pre-assessment-report.md` | Pre-assessment report (guidebook Chapter 4) |
| `high-level-report.md` | High-level assessment report (guidebook Chapter 5) |
| `scripts/rf_benchmark.py` | Benchmark harness: same computation as the owner's script, with configurable paths, per-phase timing, optional outputs, and a correctness check |
| `scripts/run_baseline.sh` | Slurm job: owner's settings with all outputs ("full") and without ("quiet") |
| `scripts/run_scaling.sh` | Slurm job: thread-scaling sweep over physical cores, repeated runs |
| `scripts/summarise_scaling.py` | Turns a sweep into a markdown table and `times.csv` for the template's graphing script |
| `scripts/run_highlevel.sh` | Slurm job: all high-level measurements (topology, core, pinned scaling, NUMA check, I/O on RAM/Lustre/VAST, Darshan attempt) |
| `scripts/run_followup.sh` | Slurm job: reruns only the LIKWID (core) and Darshan parts into an existing results folder |
| `scripts/cpulist.py` | Builds pinning lists from the compute node's real CPU layout |
| `scripts/run_strace.sh` | Slurm job: I/O cross-check with `strace -f -c -w` (a fallback when Darshan cannot be used), alternating with plain runs |
| `scripts/summarise_strace.py` | Turns a `run_strace.sh` folder into per-run I/O times and C_I/O |
| `scripts/summarise_highlevel.py` | Turns a high-level results folder into the rubric scores, tables, `times_pinned.csv` and `rubric_scores.csv` |
| `results/` | Raw results of the COSMA8 runs |

## What is deliberately not here

- **The owner's script.** It was shared privately and has no licence. `rf_benchmark.py`
  reproduces its computation, so **ask the owner's permission before making this repository
  public**.
- **Data.** The EAGLE input files stay on COSMA8.
- **Model files and raw run folders.** A single model is 7.3 GB.

## Quick start (COSMA8)

```bash
python3 -m venv ~/venvs/eagle-rf && source ~/venvs/eagle-rf/bin/activate
pip install numpy h5py scikit-learn joblib scipy matplotlib
cd scripts
export EAGLE_DATA_ROOT=/path/to/eagle_data      # folder containing L0100N1504/ etc. (not published here)
python3 rf_benchmark.py 0100 1504 0 --outdir smoke --n-jobs 4 --n-estimators 10   # smoke test
```

Full instructions, including the Slurm settings that need changing, are in the
"Reproducing this assessment" section of the report.

## Licence

Assessment material in this repository: MIT, following the SHAREing template. This does not
apply to the code under assessment.
