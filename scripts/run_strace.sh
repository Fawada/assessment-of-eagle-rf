#!/bin/bash -l
# Cross-check of the I/O rubric with strace, a standard Linux tool that works on any program
# without changes (no Darshan/HDF5 conflict).
#
# Runs the owner's full I/O configuration (model + arrays + plot, 16 pinned threads) on Lustre,
# alternating REPEATS times between:
#   plain               -> total runtime without strace (to measure strace's overhead)
#   strace -f -c -w     -> wall-clock time spent inside read/write system calls, all threads
# Alternating matters: Lustre write speed varied by more than 2x between runs minutes apart
# (it is a shared filesystem), so a single pair cannot separate strace's overhead from that.
# Computes C_I/O = 1 - (t_read + t_write) / t_total for each strace run.
#
# -w matters: without it, strace -c reports CPU time inside system calls, which leaves out the
# time spent waiting for the filesystem and would understate I/O.
#
# Usage (from the scripts folder):  sbatch run_strace.sh      (~6 minutes with REPEATS=3)

#SBATCH --job-name=rf_strace
#SBATCH --partition=cosma8
#SBATCH --account=CHANGE_ME          # your COSMA project account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
# Data: export EAGLE_DATA_ROOT=/path/to/eagle_data before sbatch (sbatch passes it to the job).
#SBATCH --output=rf_strace_%j.out

source ~/venvs/eagle-rf/bin/activate
LBOX=0100
NPART=1504
Z=0
IO_CORES=${IO_CORES:-16}
REPEATS=${REPEATS:-3}
N_EST=${N_EST:-1000}
DATA_ROOT=${DATA_ROOT:-}
LUSTRE_DIR=${LUSTRE_DIR:-/cosma8/data/CHANGE_ME_PROJECT/$USER/eagle-rf-io}

set -u
if [ -z "${DATA_ROOT:-}" ] && { [ -z "${EAGLE_DATA_ROOT:-}" ] || [ "${EAGLE_DATA_ROOT}" = CHANGE_ME ]; }; then
  echo "Set EAGLE_DATA_ROOT to the EAGLE data directory (export it before sbatch)"; exit 1
fi
case "${LUSTRE_DIR:-} ${VAST_DIR:-}" in *CHANGE_ME*)
  echo "Set LUSTRE_DIR / VAST_DIR (writable output folders) and --account before submitting"; exit 1;;
esac
HERE="$(pwd)"
BENCH="$HERE/rf_benchmark.py"
CPULIST="$HERE/cpulist.py"
SUMMARISE="$HERE/summarise_strace.py"
OUT="$HERE/strace_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" "$LUSTRE_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS=("$LBOX" "$NPART" "$Z")
[ -n "$DATA_ROOT" ] && ARGS+=(--data-root "$DATA_ROOT")
CPUS=$(python3 "$CPULIST" compact "$IO_CORES")
log() { echo "[$(date +%H:%M:%S)] $*"; }
strace -V > "$OUT/strace_version.txt" 2>&1
clean() { rm -f "$LUSTRE_DIR"/*.pkl "$LUSTRE_DIR"/*.npy "$LUSTRE_DIR"/*.pdf "$LUSTRE_DIR"/result_*.json; }

for R in $(seq 1 "$REPEATS"); do
  log "Repeat $R: plain"
  taskset -c "$CPUS" python3 "$BENCH" "${ARGS[@]}" --outdir "$LUSTRE_DIR" --tag "io_plain_r$R" \
      --n-jobs "$IO_CORES" --n-estimators "$N_EST" --save-model --save-arrays --plot \
      > "$OUT/plain_r$R.log" 2>&1 || log "  plain run $R failed"
  cp "$LUSTRE_DIR/result_io_plain_r$R.json" "$OUT/" 2>/dev/null; clean

  log "Repeat $R: strace -f -c -w"
  taskset -c "$CPUS" strace -f -c -w -o "$OUT/strace_summary_r$R.txt" \
      -e trace=read,write,pread64,pwrite64,readv,writev,openat,close,fsync,lseek \
      python3 "$BENCH" "${ARGS[@]}" --outdir "$LUSTRE_DIR" --tag "io_strace_r$R" \
      --n-jobs "$IO_CORES" --n-estimators "$N_EST" --save-model --save-arrays --plot \
      > "$OUT/strace_r$R.log" 2>&1 || log "  strace run $R failed"
  cp "$LUSTRE_DIR/result_io_strace_r$R.json" "$OUT/" 2>/dev/null; clean
done

log "Summary"
python3 "$SUMMARISE" "$OUT" | tee "$OUT/strace_io_summary.md"
log "Done: $OUT"
