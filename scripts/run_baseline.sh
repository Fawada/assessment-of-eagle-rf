#!/bin/bash -l
# Baseline job: the owner's configuration on one COSMA8 node (pre-assessment, workbook Chapter 4).
#
#   Run 1 "full":  every output the original script writes (model .pkl, 4 .npy arrays, PDF plot)
#                  -> checks it runs as a black box, and measures the I/O cost of the outputs.
#   Run 2 "quiet": no saving, no plotting -> the benchmark configuration for core/intra-node work.
#
# Both runs use the owner's settings: n_jobs=16, n_estimators=1000, max_depth=32.
#
# Before submitting, set:
#   --account   your COSMA project
#   LBOX/NPART/Z  to an existing simulation:  ls /cosma8/data/dp004/dc-sori1/data/eagle_data/
#                 (a folder L0100N1504 containing eagle_z0.0.hdf5 means LBOX=0100 NPART=1504 Z=0.0)
#   ENV SETUP   Python environment with numpy, h5py, scikit-learn, joblib, scipy, matplotlib
#
# The "full" run writes a ~7 GB model file. Point OUT_ROOT at a large filesystem, not $HOME:
#   OUT_ROOT=/path/with/space sbatch run_baseline.sh
# (COSMA8 example used in this assessment: account dr004, OUT_ROOT=/cosma/apps/durham/<user>/eagle-rf-runs)

#SBATCH --job-name=rf_baseline
#SBATCH --partition=cosma8
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=rf_baseline_%j.out

# ---------------- ENV SETUP (edit) ----------------
# module purge
# module load python/3.x
# source ~/venvs/eagle-rf/bin/activate
# --------------------------------------------------

LBOX=CHANGE_ME
NPART=CHANGE_ME
Z=CHANGE_ME

set -u
HERE="$(pwd)"
OUT="${OUT_ROOT:-$HERE}/baseline_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

# Keep numpy/BLAS from spawning their own threads on top of joblib's
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Record the node and environment
{
  echo "== host";    hostname
  echo "== lscpu";   lscpu
  echo "== memory";  free -g
  echo "== modules"; module list 2>&1
  echo "== python";  which python3; python3 --version
  echo "== pip";     python3 -m pip freeze 2>/dev/null
  echo "== latex";   command -v latex || echo "latex NOT FOUND (original script's usetex=True would fail here)"
  echo "== input";   ls -l /cosma8/data/dp004/dc-sori1/data/eagle_data/L${LBOX}N${NPART}/eagle_z${Z}.hdf5
} > "$OUT/environment.txt" 2>&1

# usetex only if LaTeX exists on the node, so the full run still completes
USETEX=""
command -v latex >/dev/null 2>&1 && USETEX="--usetex"

echo "== Run 1: full outputs"
/usr/bin/time -v python3 "$HERE/rf_benchmark.py" "$LBOX" "$NPART" "$Z" \
    --outdir "$OUT/full" --tag full --n-jobs 16 --n-estimators 1000 \
    --save-model --save-arrays --plot $USETEX \
    > "$OUT/full.log" 2> "$OUT/full.time"
echo "exit status: $?"

echo "== Run 2: quiet benchmark"
/usr/bin/time -v python3 "$HERE/rf_benchmark.py" "$LBOX" "$NPART" "$Z" \
    --outdir "$OUT/quiet" --tag quiet --n-jobs 16 --n-estimators 1000 \
    > "$OUT/quiet.log" 2> "$OUT/quiet.time"
echo "exit status: $?"

echo
cat "$OUT/full.log" "$OUT/quiet.log"
grep -h -E "Elapsed|Maximum resident" "$OUT"/*.time
echo "Results in $OUT/"
