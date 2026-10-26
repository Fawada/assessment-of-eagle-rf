#!/bin/bash -l
# Intra-node (thread) scaling sweep for the Random Forest fit (workbook, intra-node rubric).
#
# Fixed problem size (strong scaling); n_jobs swept over powers of two up to the full node.
# Each point is repeated REPEATS times. All runs are "quiet" (no saving, no plotting).
#
# On COSMA8 (L0100N1504, z=0) the full 1000 trees take ~2 minutes on 1 thread, so the
# default below is the owner's value. Otherwise, pick N_ESTIMATORS from the baseline job: the n_jobs=1 run takes roughly
#   (baseline fit time) x 16 x (N_ESTIMATORS / 1000)
# so reduce N_ESTIMATORS until the serial run is a few minutes. Fit cost is linear in the
# number of trees, so this does not change the per-tree work being measured.
#
# Before submitting, set --account, LBOX/NPART/Z and ENV SETUP as in run_baseline.sh.
# Override at submit time, e.g.:  N_ESTIMATORS=200 REPEATS=3 sbatch run_scaling.sh

#SBATCH --job-name=rf_scaling
#SBATCH --partition=cosma8
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
# Data: export EAGLE_DATA_ROOT=/path/to/eagle_data before sbatch (sbatch passes it to the job).
#SBATCH --output=rf_scaling_%j.out

# ---------------- ENV SETUP (edit) ----------------
# module purge
# module load python/3.x
# source ~/venvs/eagle-rf/bin/activate
# --------------------------------------------------

LBOX=CHANGE_ME
NPART=CHANGE_ME
Z=CHANGE_ME
N_ESTIMATORS=${N_ESTIMATORS:-1000}
REPEATS=${REPEATS:-3}
# Physical cores, not hardware threads: COSMA8 nodes have SMT (2 threads per core), so
# nproc reports 256 on a 128-core node. Scaling is measured against physical cores.
SOCKETS=$(lscpu | awk -F: '/^Socket\(s\)/ {gsub(/ /,"",$2); print $2}')
CORES_PER_SOCKET=$(lscpu | awk -F: '/^Core\(s\) per socket/ {gsub(/ /,"",$2); print $2}')
NCORES=$(( SOCKETS * CORES_PER_SOCKET ))
JOBS_LIST=${JOBS_LIST:-"1 2 4 8 16 32 64 $NCORES"}

set -u
if [ -z "${DATA_ROOT:-}" ] && { [ -z "${EAGLE_DATA_ROOT:-}" ] || [ "${EAGLE_DATA_ROOT}" = CHANGE_ME ]; }; then
  echo "Set EAGLE_DATA_ROOT to the EAGLE data directory (export it before sbatch)"; exit 1
fi
HERE="$(pwd)"
OUT="$HERE/scaling_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

{ hostname; lscpu; } > "$OUT/node.txt" 2>&1
echo "n_estimators=$N_ESTIMATORS repeats=$REPEATS cores=$NCORES jobs=[$JOBS_LIST]" | tee "$OUT/settings.txt"

for NJ in $JOBS_LIST; do
  for R in $(seq 1 "$REPEATS"); do
    python3 "$HERE/rf_benchmark.py" "$LBOX" "$NPART" "$Z" \
        --outdir "$OUT" --tag "nj${NJ}_r${R}" \
        --n-jobs "$NJ" --n-estimators "$N_ESTIMATORS" 2>> "$OUT/errors.log"
  done
done

python3 "$HERE/summarise_scaling.py" "$OUT"
