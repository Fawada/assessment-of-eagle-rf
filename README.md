# Assessment of the EAGLE gas density profile Random Forest

Performance assessment of a Python/scikit-learn Random Forest code that predicts halo gas density
profiles in the EAGLE simulations from global galaxy and halo properties. Carried out on COSMA8
following the SHAREing performance assessment guidebook and the
[SHAREing assessment template](https://github.com/SHAREing-DRI/assessment-template).

**Status (2026-10-19): pre-assessment complete; passed.** High-level assessment next.

**Read first:** [`pre-assessment-report.md`](pre-assessment-report.md). It is written to double as a
worked example for assessing other small, CPU-only Python codes.

## Key results so far

| | |
|---|---|
| Benchmark | Training on EAGLE `L0100N1504`, z=0: 48,031 haloes, 6 features, 9 target bins, 1000 trees |
| Owner's full run on COSMA8 | 85.5 s, of which **84% is writing outputs** (a 7.3 GB model file) and 10% is training |
| Training, 16 threads (owner's setting) | 8.7 s |
| Training, 1 → 64 threads | 121.0 s → 3.0 s; parallel efficiency 0.87 at 16 threads, 0.63 at 64 |
| Peak memory | 8.7 GB |
| Reproducibility | Identical trained forest in all 26 runs (hash `5590010f3e7cacbb`) |

## Repository layout

| Path | Contents |
|---|---|
| `pre-assessment-report.md` | Pre-assessment report (guidebook Chapter 4) |
| `scripts/rf_benchmark.py` | Benchmark harness: same computation as the owner's script, with configurable paths, per-phase timing, optional outputs, and a correctness check |
| `scripts/run_baseline.sh` | Slurm job: owner's settings with all outputs ("full") and without ("quiet") |
| `scripts/run_scaling.sh` | Slurm job: thread-scaling sweep over physical cores, repeated runs |
| `scripts/summarise_scaling.py` | Turns a sweep into a markdown table and `times.csv` for the template's graphing script |
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
python3 rf_benchmark.py 0100 1504 0 --outdir smoke --n-jobs 4 --n-estimators 10   # smoke test
```

Full instructions, including the Slurm settings that need changing, are in the
"Reproducing this assessment" section of the report.

## Licence

Assessment material in this repository: MIT, following the SHAREing template. This does not
apply to the code under assessment.
