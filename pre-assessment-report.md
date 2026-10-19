---
License: Copyright © 2026 Durham University, SHAREing Project, MIT Licensed
Creator: Fawada Qaiser
Contributors: Fawada Qaiser
Summary: Pre-assessment report for the EAGLE gas density profile Random Forest code
Template: SHAREing-DRI/assessment-template, pre-assessment-report.md (Chapter 4 of the guidebook)
---

# Pre-assessment: EAGLE gas density profile Random Forest

> **How to use this report as an example.** This is a completed pre-assessment of a small,
> CPU-only Python machine-learning code. Every number in it comes from a run that can be
> repeated with the scripts in [`scripts/`](scripts/); the raw outputs are in
> [`results/`](results/). Short "Method" notes explain *why* each step was done, so the same
> approach can be reused for other codes. Items still open are marked **Pending** together with
> the command that closes them.

## Assessment objective

An internal review of a Random Forest regression code that predicts gas density profiles of
haloes in the EAGLE cosmological simulations from global galaxy and halo properties. The
submission is a single Python script provided by the code owner (a COSMA8 user in project dp004);
there is no repository or version tag. The benchmark is training on the EAGLE `L0100N1504`
simulation at redshift 0. The assessment is performed by Fawada Qaiser (DiRAC / Durham
University) as part of the SHAREing project (UKRI Digital Research Infrastructure, grant
UKRI1801), on COSMA8, between 18 and 19 October 2026.

## Disclaimers

1. This report is not a commentary on code quality, but an indicator of the quality of the
   current SHAREing testing methodology as of October 2026.
2. The pre-assessment is only a preliminary assessment of submission suitability and does not
   guarantee a full assessment.
3. The scientific validity of the model (choice of features, targets, hyperparameters, or the
   accuracy of its predictions) is outside the scope of this assessment. Prediction accuracy is
   recorded only as a check that every run computed the same thing.

## Table of contents

- [x] [1: Benchmark setup](#1-benchmark-setup)
- [x] [2: Description of working environment](#2-description-of-working-environment)
- [x] [3: Compiler setup and optimisations](#3-compiler-setup-and-optimisations)
- [x] [4: Computational complexity and scaling](#4-computational-complexity-and-scaling)
- [x] [5: Memory, storage and I/O](#5-memory-storage-and-io)
- [ ] [6: Additional comments from submitter](#6-additional-comments-from-submitter)
- [x] [7: Pre-assessment outcome](#7-pre-assessment-outcome)

## Summary of findings

| Question (guidebook Chapter 4) | Answer |
|---|---|
| Does the benchmark run as a black box? | Yes, once paths are made configurable (the original hard-codes the owner's directories). Both baseline runs and all 24 scaling runs completed with exit code 0 |
| Is it a single well-defined benchmark? | Yes: one script, one training run, three arguments |
| Is the result reproducible? | Yes: with the fixed random seed, the trained forest is bit-for-bit identical in every run and at every thread count (forest hash `5590010f3e7cacbb`) |
| Best-case runtime under 10 minutes? | Yes: 85.5 s for the owner's full run; 8.7 s of training at 16 threads; 121 s of training on 1 thread |
| Memory footprint | 8.7 GB peak, under 1% of a COSMA8 node: well below the guidebook's 20–80% guideline |
| Parallelism | Shared-memory threads within one process (scikit-learn/joblib). No MPI, no GPU |
| Where does the time go? | **84% writing outputs** (a 7.3 GB model file takes 68.8 s to save); 10% training |
| Intra-node scaling (training) | Parallel efficiency 0.94 at 8 threads, 0.87 at 16, 0.74 at 32, 0.63 at 64 |
| Outcome | **Proceed to the high-level assessment**, with the I/O and scaling caveats below |

## 1: Benchmark setup

### What the code does

The script trains a `RandomForestRegressor` (scikit-learn) to predict a halo's gas density
profile from six properties of the central galaxy and its host halo:

| Feature | HDF5 dataset | Transform |
|---|---|---|
| Gas mass of the central galaxy | `CentralGalaxyMass_Gas` | log10 |
| Stellar mass | `CentralGalaxyMass_Stellar` | log10 |
| Star formation rate | `CentralGalaxyStarFormationRate` | modified arsinh |
| Black hole mass | `CentralGalaxyMass_BH` | modified arsinh |
| Black hole accretion rate | `CentralGalaxyBHAccretionRate` | modified arsinh |
| Halo mass M200c | `MCrit200` | log10 |

The target is the gas density profile (`DensityProfiles/Gas`), restricted to radial bins between
0.05 and 1 R200c and transformed with a modified arsinh scaled by the critical density. Haloes
with zero gas or stellar mass, or M200c ≤ 10^10 M☉, are removed. The data are split 80/20 at
random (`random_state=1`), a forest is trained, and the script then saves the model and the
train/test arrays and draws a four-panel figure comparing true and predicted mean profiles in
four halo-mass bins.

Steps in order, with the phase names used in this report:

| Phase | What happens | Parallel? |
|---|---|---|
| startup | Python starts and imports numpy, h5py, scikit-learn | No |
| read | One HDF5 file is read into memory | No |
| preprocess | Cuts and transforms with numpy | No |
| split | `train_test_split`, 80/20 | No |
| **fit** | **Train 1000 trees (the main computation)** | **Yes, threads** |
| predict | Predict the test set (added by the harness as a correctness check) | Yes, threads |
| save | `joblib.dump` of the model, `np.save` of four arrays | No |
| plot | Matplotlib figure; predicts each mass bin again | Prediction only |

### Fetch and build program

There is no build step and no repository. The submission is one Python script run with three
positional arguments:

```bash
python <script>.py <Lbox> <Npart> <z>      # e.g. python <script>.py 0100 1504 0
```

The arguments select the input file `<data>/L<Lbox>N<Npart>/eagle_z<z>.hdf5`. Note that `<z>` must
match the file name exactly (`0`, not `0.0`).

Hyperparameters fixed in the script:

| Parameter | Value |
|---|---|
| `n_estimators` (trees) | 1000 |
| `max_depth` | 32 |
| `min_samples_split` | 2 |
| `max_features` | `'sqrt'` |
| `n_jobs` (threads) | 16 |
| `random_state` | 1 |

Dependencies (from the imports): numpy, h5py, joblib, matplotlib, scipy, scikit-learn. The owner
has not provided versions.

### Issues preventing use as a black box

These were found by reading the script before running it.

| Issue | Effect | How it was handled |
|---|---|---|
| All paths hard-coded to the owner's directories (`/cosma8/data/../../...`), including where outputs are written | Another user cannot run it; if they had write access, it would **overwrite the owner's saved model** | The original script was **not run**. A benchmark harness with configurable paths was used instead (below) |
| `plt.rc('text', usetex=True)` | Needs LaTeX on the compute node; if missing, the script crashes at the very end, after training | Harness uses LaTeX only when it is installed (`--usetex` is passed automatically by `run_baseline.sh` if `latex` is found) |
| `test = True` prints arrays; the plotting section prints full prediction arrays | Terminal output during the run (an "assessment crime" in the guidebook) | Printing off by default in the harness (`--verbose` restores it) |
| `n_jobs = 16` fixed | Cannot scale with the node | Made a harness argument |
| `from scipy.stats import *` | Unused import; no performance effect | None needed |
| `r2_score` imported but never used; no accuracy metric is computed | None for performance | Harness computes test-set R² as a correctness check |

> **Method note.** Never run a submitted script that writes into someone else's directories.
> Read it first, list every hard-coded path, and make them arguments in a copy.

### Fetch and run benchmark

The benchmark uses a harness, [`scripts/rf_benchmark.py`](scripts/rf_benchmark.py), that performs
the **same computation** as the owner's script (same datasets, cuts, transforms, split and forest
settings). The differences are limited to:

- paths and run parameters are command-line arguments;
- each phase is timed separately with `time.perf_counter()`;
- saving and plotting are off unless `--save-model --save-arrays --plot` are given;
- `n_jobs`, `n_estimators` and an optional halo subsample (`--subsample`) can be changed;
- after training, the test set is predicted and the forest is hashed (correctness check);
- each run writes a JSON file with timings, peak memory, data sizes, model size, library
  versions and the correctness values.

The benchmark run (owner's settings, outputs off):

```bash
python3 scripts/rf_benchmark.py 0100 1504 0 --outdir bench_out --n-jobs 16 --n-estimators 1000
```

Expected console output: a single line at the end, for example

```text
[quiet] fit=8.70s predict=0.40s total=10.99s peakRSS=8.70GB r2=0.7419863541 forest=5590010f3e7cacbb -> .../result_quiet.json
```

Any run of `L0100N1504`, `z=0` with the default forest settings must report
`forest=5590010f3e7cacbb` and `r2=0.7419863541`, whatever the thread count. A different value
means the computation changed (different data, library version or settings).

> **Method note: choosing the correctness check.** With a fixed `random_state`, scikit-learn
> draws every tree's random seed before training starts, so the trees do not depend on how many
> threads build them. The harness therefore hashes the trees themselves (split features,
> thresholds, children, leaf values). It does **not** hash the predictions: these differ by
> about 1e-15 between thread counts, because threads add up the 1000 tree outputs in different
> orders and floating-point addition is not associative. This was checked directly (identical
> trees, maximum prediction difference 8.9e-16) before relying on the hash.

Slurm jobs used (see [Reproducing this assessment](#reproducing-this-assessment)):

| Job script | Purpose | Slurm job IDs |
|---|---|---|
| [`scripts/run_baseline.sh`](scripts/run_baseline.sh) | Owner's settings, twice: "full" (all outputs, as the original script) and "quiet" (no outputs) | 12186031 |
| [`scripts/run_scaling.sh`](scripts/run_scaling.sh) | Training at 1–64 and 256 threads, 3 repeats each | 12186032 |
| `run_scaling.sh` with `JOBS_LIST="128"` | Missing 128-thread point | 12187354 (**Pending**) |

### Reference architecture

The owner runs the code on COSMA8 (the paths are on `/cosma8/data`). No reference runtimes were
provided. The same system was used for this assessment, so there is no difference in
architecture to account for.

## 2: Description of working environment

### Hardware information

The COSMA8 system at Durham University (DiRAC Memory Intensive service) was used. All runs were
on a single exclusive compute node (`m8138`) in the `cosma8` partition, submitted with the
`dr004` project account. Processor details were recorded with `lscpu` inside the job:

| Specification | Per node |
|---|---|
| Processors | 2 × AMD EPYC 7H12 64-Core Processor |
| Hardware accelerators | None used |
| Sockets | 2 |
| Cores | 128 physical (64 per socket) |
| Hardware threads | 256 (SMT: 2 threads per core) |
| NUMA domains | 8 (16 cores each) |
| Clock speed per CPU | **Pending**: `grep MHz baseline_12186031/environment.txt` |
| RAM | **Pending**: `free -g` section of `baseline_12186031/environment.txt` (COSMA8 documentation states 1 TB) |
| Output filesystem used | `/cosma/apps`: VAST, NFS mount (958 TB free at the time) |
| Input filesystem | `/cosma8/data` (Lustre; to confirm with `df -T /cosma8/data`) |

Two hardware facts matter for interpreting the results:

- **SMT.** Each physical core can run two hardware threads. Two threads on one core share that
  core's execution units, so they do not double the work done. `nproc` reports 256, which is
  why the first scaling job ran a 256-thread point instead of 128. Scaling is measured against
  the **128 physical cores**.
- **NUMA.** The 128 cores form 8 groups of 16, each with its own directly attached memory.
  Accessing another group's memory is slower. Runs with more than 16 threads necessarily span
  more than one group.

### Libraries and modules

A Python virtual environment was created on the login node and activated inside every job:

```bash
python3 -m venv ~/venvs/eagle-rf
source ~/venvs/eagle-rf/bin/activate
pip install numpy h5py scikit-learn joblib scipy matplotlib
```

No environment modules were loaded. Exact versions are recorded in every result JSON (`versions`)
and in `environment.txt` (`pip freeze`).

| Package | Version |
|---|---|
| Python, numpy, h5py, scikit-learn, joblib, scipy, matplotlib | **Pending**: `python3 -c "import json; print(json.load(open('<run>/quiet/result_quiet.json'))['versions'])"` |
| LaTeX available on compute node | **Pending**: `latex` line of `environment.txt` |

The owner's library versions are unknown; they should be asked for `pip freeze` to confirm the
assessment environment matches theirs.

`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` were set to 1 in all jobs, so
numpy's linear-algebra library cannot start its own threads on top of scikit-learn's.

### Assessment tools

1. Pre-assessment (used):
   - Python `time.perf_counter()` inside the harness, for phase timings
   - `resource.getrusage` inside the harness, for peak memory
   - `/usr/bin/time -v` around each run, for total wall time and peak memory as seen by the OS
   - `lscpu`, `free`, `pip freeze`, recorded in each job
2. High-level assessment (planned):
   - The harness itself, for the intra-node scaling rubric (already run)
   - LIKWID (`likwid-perfctr`), for achieved FLOP rate against the core's peak (core rubric)
   - Darshan, for the I/O rubric (needs to be enabled for a non-MPI Python process,
     `DARSHAN_ENABLE_NONMPI=1`; availability on COSMA8 to be checked)
3. Low-level assessment (if pursued):
   - `py-spy` or the Python profiler, for time spent in Python versus compiled code
   - Thread pinning (`numactl`, `taskset`) to test the NUMA explanation in Section 4

## 3: Compiler setup and optimisations

The code is Python and is not compiled. The training itself runs in scikit-learn's compiled
(Cython) tree-building code, which releases Python's global interpreter lock, so threads can
build trees genuinely in parallel. There are no compiler flags to report. The settings that
play the same role are:

| Setting | Value | Effect |
|---|---|---|
| Parallel backend | joblib threads (scikit-learn's choice for forests) | Trees are built in parallel inside one process, sharing memory |
| `n_jobs` | 16 in the original | Number of threads |
| BLAS threads | Forced to 1 in this assessment | Avoids oversubscription |
| Python build / scikit-learn wheel | pip wheels, generic | Not optimised for AMD EPYC specifically |

No build or compatibility problems were encountered.

## 4: Computational complexity and scaling

### Problem size

| Quantity | Value |
|---|---|
| Input file | `L0100N1504/eagle_z0.hdf5`, 85,293,168 bytes |
| Haloes in file | 73,518 |
| Radial bins in file | 19 |
| Haloes after cuts | 60,039 |
| Target bins kept (0.05–1 R200c) | 9 |
| Training / test haloes | 48,031 / 12,008 |
| Features | 6 |

### Size of the trained model

With `max_depth=32` and `min_samples_split=2`, trees grow until almost every leaf holds one
halo. A 10-tree test run produced 522,608 nodes (about 52,000 per tree) and reached the
maximum depth of 32. Every node stores a 9-value profile, so the 10-tree forest's stored
profiles alone took 37.6 MB.

Scaled to 1000 trees this predicts about 52 million nodes, 3.8 GB of stored profiles plus a
similar amount of tree structure: about 7 GB. The saved model was **7.3 GB**, consistent
with this estimate. Measured node count for 1000 trees: **Pending**,
`python3 -c "import json; print(json.load(open('<run>/full/result_full.json'))['model'])"`.

### How cost scales (from the algorithm)

- **Number of trees:** training time and model size grow in proportion. This is the simplest
  way to make the benchmark longer or shorter without changing the work per tree.
- **Number of haloes (n):** each tree's training costs roughly n log n, and its size grows with
  n. `--subsample` changes n within one simulation.
- **Number of target bins:** node size grows in proportion; tree building time increases
  modestly.
- **Simulation box:** four EAGLE boxes are available (`L0025N0376`, `L0050N0752`, `L0100N0752`,
  `L0100N1504`). They give natural problem sizes for weak scaling. `L0025N0376` is too small to
  be useful (training takes 0.04 s for 10 trees).

### Strong scaling within a node

Fixed problem (L0100N1504, z=0, 1000 trees); thread count varied; 3 runs per point; outputs
off. Raw output: [`results/scaling_12186032_summary.md`](results/scaling_12186032_summary.md).

| Threads | Fit time, median (s) | Spread (min–max, s) | Speed-up | Parallel efficiency |
|---|---|---|---|---|
| 1 | 121.04 | 120.92–121.18 | 1.00 | 1.00 |
| 2 | 61.09 | 60.99–61.30 | 1.98 | 0.99 |
| 4 | 30.92 | 30.82–35.90 | 3.91 | 0.98 |
| 8 | 16.16 | 15.71–16.87 | 7.49 | 0.94 |
| 16 | 8.66 | 8.61–8.87 | 13.98 | 0.87 |
| 32 | 5.10 | 4.97–5.23 | 23.73 | 0.74 |
| 64 | 3.01 | 2.88–3.07 | 40.23 | 0.63 |
| 128 | **Pending** (job 12187354) | | | |
| 256 (SMT) | 2.67 | 2.65–2.72 | 45.41 | 0.35 per physical core |

*Speed-up = time on 1 thread ÷ time on N threads. Parallel efficiency = speed-up ÷ N (1.0 is
perfect). The 256-thread efficiency is given per physical core (45.41 ÷ 128).*

Observations:

- Scaling is near-perfect up to 8 threads (efficiency ≥ 0.94).
- Efficiency falls below the guidebook's 80% reference point between 16 and 32 threads, and
  remains above 60% at 64 threads.
- The owner's choice of 16 threads (efficiency 0.87) is close to the most efficient setting.
- Using both SMT threads of every core (256) is only 11% faster than 64 threads.
- Run-to-run variation is small (one 4-thread run at 35.9 s is the only outlier), so three
  repeats were sufficient.

Possible causes of the fall in efficiency above 16 threads. These are **hypotheses**, not
measured findings:

1. **Load imbalance.** 1000 trees of varying size divided among many threads: at 64 threads
   each builds about 16 trees, so uneven tree sizes leave some threads idle at the end.
2. **Fixed serial work.** Python and joblib dispatch the trees and collect results; this time
   does not shrink as threads are added, and becomes a larger fraction when training takes 3 s.
3. **NUMA.** The fall begins exactly where runs start to span more than one 16-core NUMA domain.
   Pinning threads to one domain would test this.

Data for the template's graphing script: [`results/times.csv`](results/times.csv) (1–64 threads).

```bash
cat results/times.csv | ./scripts/intranode_times_to_graph.py -gmd --markdown-file=stdout
```

(`intranode_times_to_graph.py` is in the SHAREing `assessment-template` repository.)

### Weak scaling

Not yet measured. Planned: train on the four EAGLE boxes (or on `--subsample` sizes of
`L0100N1504`) with the thread count increased in proportion to the problem size. Because cost
grows roughly as n log n rather than n, the expected time per halo will rise slightly with size
even with perfect parallelism; this must be stated when interpreting the results.

## 5: Memory, storage and I/O

### Memory

| Run | Peak memory (harness) | Peak memory (`/usr/bin/time`) |
|---|---|---|
| Baseline, full | 8.74 GB | 8,531,436 kB (8.74 GB) |
| Baseline, quiet | 8.70 GB | 8,503,436 kB (8.71 GB) |
| Scaling, 1 → 256 threads | 8.23 → 9.47 GB | |

Almost all of the memory is the trained model; the input data occupy well under 0.1 GB. Peak
memory rises slightly with thread count, probably because each thread holds its own working
buffers while building a tree.

At about 9 GB, the benchmark uses under 1% of a COSMA8 node's memory (**Pending**: confirm node
RAM). The guidebook prefers benchmarks using 20–80% of node memory. Reaching that would need a
much larger problem than the code's real use case, so the benchmark is kept at the owner's size
and the low memory use is recorded here as a characteristic of the code.

### Storage

| File written by the original script | Size |
|---|---|
| `model_eagle_L0100N1504_z0_red.pkl` (the trained forest, via `joblib.dump`) | **7.3 GB** |
| Four `.npy` arrays (train/test features and targets) | < 0.01 GB each |
| `RF_dens_prof_eagle_L0100N1504_z0_rev.pdf` | < 0.01 GB |

Each run of the original script writes a new 7.3 GB model file, overwriting the previous one
of the same name. **This is too large for a typical COSMA home directory:** writing outputs
to `$HOME` risks filling the quota and failing the job. In this assessment outputs were written
to `/cosma/apps` (set with `OUT_ROOT` in `run_baseline.sh`).

### Where the time goes (owner's full configuration)

Phase timings of the "full" run, which behaves like the original script (job 12186031):

| Phase | Time (s) | Share of run |
|---|---|---|
| startup (imports) | 3.91 | 4.6% |
| read input | 0.11 | 0.1% |
| preprocess + split | 0.07 | 0.1% |
| **fit (training)** | **8.89** | **10.4%** |
| predict (test set) | 0.38 | 0.4% |
| **save (model + arrays)** | **68.82** | **80.5%** |
| plot | 3.28 | 3.8% |
| **Total (inside the script)** | **85.47** | |
| Wall time measured by `/usr/bin/time` | 90.60 | |

**Writing outputs takes 84% of the owner's runtime** (save 80.5% + plot 3.8%). Saving the
model takes 7.7 times as long as training it. Reading the input takes 0.1%. The effective
save rate is about 106 MB/s (7.3 GB in 68.8 s). This combines two things this test cannot
separate: Python converting 52 million nodes into bytes (pickling, single-threaded), and
writing to the VAST/NFS filesystem.

With outputs off ("quiet" run), the script total is 11.0 s, of which training is 8.7 s (79%).

Two further observations:

- **About 5 seconds after the script finishes.** In both baseline runs the wall time measured
  by `/usr/bin/time` exceeds the script's own total by about 5 s (5.1 s full, 5.3 s quiet).
  The most likely cause is Python freeing the 52-million-node model at exit. This is a
  hypothesis.
- **Filesystem.** Outputs went to `/cosma/apps` (VAST/NFS); the owner writes to `/cosma8/data`
  (Lustre, to confirm). Write speeds differ between filesystems, so the 68.8 s save is specific to
  `/cosma/apps`. The I/O rubric should be measured on the owner's filesystem type.

> **Method note.** Time each phase separately inside the run. A single wall-clock time would
> show 85 s and suggest a compute-bound code; the breakdown shows the opposite.

### Console output

The original script prints shapes, the first ten rows of the features and targets, and full
prediction arrays during plotting. The harness prints one summary line per run; the benchmark
runs used this quiet mode.

## 6: Additional comments from submitter

None received yet. Questions for the owner:

1. Is `L0100N1504` at `z=0` with these settings representative of real use? Which redshifts and
   boxes do you normally train on?
2. Which data variant does the script really use? The data directory also contains `10bins`,
   `20bins` and `40bins` folders and top-level `eagle_z*.hdf5` files; the script reads
   `L<Lbox>N<Npart>/eagle_z<z>.hdf5`, which has 19 radial bins.
3. Do you need to save the full 7.3 GB model on every run? Saving takes 80% of the runtime.
4. Your Python and library versions (`pip freeze`).
5. Is 16 threads a deliberate choice? Measurements show it is close to the most efficient
   setting; 64 threads trains 2.9× faster (3.0 s against 8.7 s) at lower efficiency.

## 7: Pre-assessment outcome

**The code passes the pre-assessment and proceeds to the high-level assessment.**

It runs as a black box once paths are configurable, completes well within 10 minutes, gives
bit-for-bit identical trained models in every run, and has clear problem-size controls.

Points to carry forward into the high-level assessment:

| Rubric | Applies? | Notes |
|---|---|---|
| Core | Yes | Measure with LIKWID. Tree building is mostly comparisons, sorting and memory access, not floating-point arithmetic, so a FLOP-based score will be low however good the code is. Interpret it as a limitation of the metric for this kind of algorithm |
| Intra-node | Yes | Strong scaling done (above); complete the 128-thread point; weak scaling to do |
| I/O | Yes, and dominant | 84% of runtime. Measure on `/cosma8/data` (owner's filesystem), ideally with Darshan; separate pickling time from write time |
| Inter-node | No | No MPI or other multi-node parallelism |
| GPU | No | CPU-only code |

Open items before the high-level stage:

- 128-thread point (job 12187354)
- Hardware, memory and software-version details marked **Pending** above
- Answers to the owner questions in Section 6

## Reproducing this assessment

On COSMA8 (replace the account and output path with your own):

```bash
# 1. Environment (login node)
python3 -m venv ~/venvs/eagle-rf
source ~/venvs/eagle-rf/bin/activate
pip install numpy h5py scikit-learn joblib scipy matplotlib

# 2. Smoke test (login node, seconds; not a measurement)
cd scripts
python3 rf_benchmark.py 0100 1504 0 --outdir smoke --n-jobs 4 --n-estimators 10

# 3. Find an account that can use the cosma8 partition
for a in $(sacctmgr -n show assoc user=$USER format=account); do
  echo -n "$a: "; sbatch --test-only -A $a -p cosma8 -N1 -t 00:05:00 --wrap=hostname 2>&1 | tail -1
done

# 4. Set --account, LBOX=0100, NPART=1504, Z=0 and the venv line in both job scripts, then:
OUT_ROOT=/path/with/space sbatch run_baseline.sh            # ~2 min; writes a 7.3 GB model
N_ESTIMATORS=1000 REPEATS=3 sbatch run_scaling.sh           # ~16 min

# 5. Summarise the sweep
python3 summarise_scaling.py scaling_<jobid>
```

Practical problems met while doing this, and their fixes:

| Problem | Fix |
|---|---|
| Files copied from Windows had Windows line endings (`$'\r': command not found`) | `sed -i 's/\r$//' *.sh *.py`, or clone with Git using `.gitattributes` (included) |
| Folder copied from Windows was read-only (`sed: couldn't open temporary file: Permission denied`) | `chmod u+w .` |
| Not every project account may use the `cosma8` partition | `sbatch --test-only` loop in step 3 |
| `nproc` reports 256 on a 128-core node (SMT) | `run_scaling.sh` now counts physical cores from `lscpu` |
| A 7.3 GB model would fill the home directory | `OUT_ROOT` variable in `run_baseline.sh` |
