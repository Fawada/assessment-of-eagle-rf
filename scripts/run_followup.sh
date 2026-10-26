#!/bin/bash -l
# Follow-up to run_highlevel.sh: reruns only the parts that need LIKWID or Darshan
#   Part B   core rubric (likwid-bench scalar peak; likwid-perfctr FLOPS on a pinned 1-thread run)
#   Part D2  one owner-configuration run under Darshan, writing to Lustre
# Results are written INTO an existing highlevel_<jobid> folder, replacing that folder's
# core/ files and Darshan files, so summarise_highlevel.py combines everything.
#
# Usage (from the scripts folder):
#   TARGET=highlevel_<jobid> sbatch run_followup.sh       # default: the newest highlevel_* folder
# Takes about 6-8 minutes.

#SBATCH --job-name=rf_followup
#SBATCH --partition=cosma8
#SBATCH --account=dr004
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=rf_followup_%j.out

source ~/venvs/eagle-rf/bin/activate
LBOX=0100
NPART=1504
Z=0
N_EST=${N_EST:-1000}
RUN_CORE=${RUN_CORE:-1}          # set RUN_CORE=0 to rerun only the Darshan step
IO_CORES=${IO_CORES:-16}
DATA_ROOT=${DATA_ROOT:-}
LUSTRE_DIR=${LUSTRE_DIR:-/cosma8/data/dr004/dc-qais1/eagle-rf-io}
COMPILER_MODULE=gnu_comp/14.1.0
MPI_MODULE=openmpi/5.0.3          # COSMA8: darshan-runtime refuses to load without an MPI module
LIKWID_MODULE=likwid/5.5.1
DARSHAN_RUNTIME_MODULE=darshan-runtime/3.4.6
DARSHAN_UTIL_MODULE=darshan-util/3.4.6

set -u
HERE="$(pwd)"
BENCH="$HERE/rf_benchmark.py"
CPULIST="$HERE/cpulist.py"
TARGET=${TARGET:-$(ls -d "$HERE"/highlevel_* 2>/dev/null | sort | tail -1)}
[ -d "$TARGET" ] || { echo "No target folder found (set TARGET=highlevel_<jobid>)"; exit 1; }
OUT=$(cd "$TARGET" && pwd)
mkdir -p "$OUT/core" "$OUT/io"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS=("$LBOX" "$NPART" "$Z")
[ -n "$DATA_ROOT" ] && ARGS+=(--data-root "$DATA_ROOT")
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "Writing into $OUT"

# ---------------------------------------------------------------- Part B: core rubric
if [ "$RUN_CORE" = "1" ]; then
log "Part B: core rubric"
rm -f "$OUT"/core/*
module load "$COMPILER_MODULE" "$LIKWID_MODULE" > "$OUT/core/module_likwid.txt" 2>&1
if command -v likwid-bench >/dev/null; then
  likwid-topology > "$OUT/core/likwid-topology.txt" 2>&1 || true
  L1D=$(lscpu | awk -F: '/^L1d cache/ {print $2}')
  echo "L1d cache:$L1D" > "$OUT/core/l1d.txt"
  likwid-bench -t peakflops        -w S0:16kB:1 > "$OUT/core/peak_scalar.txt"  2>&1 || log "  peakflops failed"
  likwid-bench -t peakflops_avx     -w S0:16kB:1 > "$OUT/core/peak_avx.txt"     2>&1 || true
  likwid-bench -t peakflops_avx_fma -w S0:16kB:1 > "$OUT/core/peak_avx_fma.txt" 2>&1 || true
  likwid-perfctr -a > "$OUT/core/groups.txt" 2>&1 || true
  FIRST=$(python3 "$CPULIST" compact 1)
  for GROUP in FLOPS_DP FLOPS_SP; do
    log "  likwid-perfctr $GROUP on CPU $FIRST"
    likwid-perfctr -f -C "$FIRST" -g "$GROUP" \
        python3 "$BENCH" "${ARGS[@]}" --outdir "$OUT/core" --tag "perfctr_$GROUP" \
        --n-jobs 1 --n-estimators "$N_EST" > "$OUT/core/perfctr_$GROUP.txt" 2>&1 \
        || log "  likwid-perfctr $GROUP failed (see core/perfctr_$GROUP.txt)"
  done
else
  log "  LIKWID not available after module load (see core/module_likwid.txt)"
fi
else
  log "Part B skipped (RUN_CORE=0); keeping existing core/ results"
fi

# ---------------------------------------------------------------- Part D2: Darshan
log "Part D2: Darshan (one run, Lustre)"
rm -rf "$OUT"/io/darshan_* "$OUT"/io/result_io_darshan.json "$OUT"/io/module_darshan.txt
module load "$COMPILER_MODULE" "$MPI_MODULE" "$DARSHAN_RUNTIME_MODULE" "$DARSHAN_UTIL_MODULE" > "$OUT/io/module_darshan.txt" 2>&1
DARSHAN_LIB=""
for d in $(echo "${LD_LIBRARY_PATH:-}" | tr ':' ' ') $(dirname "$(command -v darshan-config 2>/dev/null || echo /x/x)")/../lib; do
  [ -f "$d/libdarshan.so" ] && DARSHAN_LIB="$d/libdarshan.so" && break
done
echo "libdarshan: ${DARSHAN_LIB:-NOT FOUND}" >> "$OUT/io/module_darshan.txt"
if [ -n "$DARSHAN_LIB" ]; then
  # Three COSMA8-specific settings, each found by testing:
  #  1. DARSHAN_MOD_DISABLE=H5F,H5D  - this build has HDF5 support and otherwise crashes Python
  #     ("undefined symbol: H5get_libversion"), because pip's h5py bundles its own HDF5 library.
  #  2. DARSHAN_LOGPATH (a directory), not DARSHAN_LOGFILE - Python starts a helper process
  #     (uname -p) that inherits Darshan and would claim a fixed log file name first.
  #  3. Darshan files logs under LOGPATH/YYYY/M/D and does not create those folders itself.
  DLOG="$OUT/io/darshan_logs"
  mkdir -p "$DLOG/$(date +%Y/%-m/%-d)" "$DLOG/$(date +%Y/%m/%d)" "$LUSTRE_DIR"
  CPUS=$(python3 "$CPULIST" compact "$IO_CORES")
  DARSHAN_ENABLE_NONMPI=1 DARSHAN_MOD_DISABLE=H5F,H5D DARSHAN_LOGPATH="$DLOG" \
  LD_PRELOAD="$DARSHAN_LIB" taskset -c "$CPUS" python3 "$BENCH" "${ARGS[@]}" --outdir "$LUSTRE_DIR" \
      --tag io_darshan --n-jobs "$IO_CORES" --n-estimators "$N_EST" --save-model --save-arrays --plot \
      >> "$OUT/io/darshan_run.log" 2>&1 || log "  Darshan run failed"
  cp "$LUSTRE_DIR"/result_io_darshan.json "$OUT/io/" 2>/dev/null
  rm -f "$LUSTRE_DIR"/*.pkl "$LUSTRE_DIR"/*.npy "$LUSTRE_DIR"/*.pdf "$LUSTRE_DIR"/result_*.json
  PYLOG=$(find "$DLOG" -name "*python3*.darshan" | head -1)
  find "$DLOG" -name "*.darshan*" > "$OUT/io/darshan_logs.txt"
  if [ -n "$PYLOG" ]; then
    echo "python log: $PYLOG" >> "$OUT/io/darshan_logs.txt"
    darshan-parser --total "$PYLOG" > "$OUT/io/darshan_total.txt" 2>&1
    darshan-parser --perf  "$PYLOG" > "$OUT/io/darshan_perf.txt"  2>&1
  else
    log "  no python3 Darshan log produced (see io/darshan_run.log, io/darshan_logs.txt)"
  fi
else
  log "  libdarshan.so not found (see io/module_darshan.txt)"
fi

log "Summarising"
python3 "$HERE/summarise_highlevel.py" "$OUT" --unpinned "$HERE/scaling_12186032" 2>/dev/null \
  || python3 "$HERE/summarise_highlevel.py" "$OUT" || log "summary failed"
log "Done: $OUT"
