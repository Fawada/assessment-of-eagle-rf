#!/bin/bash -l
# High-level assessment (guidebook Chapter 5) for the EAGLE Random Forest benchmark on one COSMA8 node.
#
# Part A  Node topology                     lscpu, likwid-topology, pinning layout
# Part B  Core rubric                       likwid-bench scalar peak; likwid-perfctr FLOPS on a
#                                           1-thread run pinned to one core
# Part C  Intra-node rubric                 strong-scaling sweep with threads PINNED to physical
#                                           cores (compact), 1..all cores, plus all SMT threads;
#                                           plus one exploratory NUMA comparison at 16 threads
# Part D  I/O rubric                        owner's full run (model + arrays + plot) writing to
#                                           RAM (/dev/shm), Lustre and VAST; one run under Darshan
#
# Each part writes into highlevel_<jobid>/ ; a failure in one part does not stop the others.
# Finish with:  python3 summarise_highlevel.py highlevel_<jobid>
#
# Submit from the scripts folder:  sbatch run_highlevel.sh
# Settings used for this assessment are below; change ACCOUNT-specific lines for other users.

#SBATCH --job-name=rf_highlevel
#SBATCH --partition=cosma8
#SBATCH --account=CHANGE_ME          # your COSMA project account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
# Data: export EAGLE_DATA_ROOT=/path/to/eagle_data before sbatch (sbatch passes it to the job).
#SBATCH --output=rf_highlevel_%j.out

source ~/venvs/eagle-rf/bin/activate
LBOX=0100
NPART=1504
Z=0
REPEATS=${REPEATS:-3}
N_EST=${N_EST:-1000}           # trees; 1000 = owner's setting
IO_CORES=${IO_CORES:-16}       # threads for the I/O runs; 16 = owner's setting
DATA_ROOT=${DATA_ROOT:-}       # empty = harness default (owner's COSMA8 data directory)

# Output locations for the 7.3 GB model in Part D (each file is deleted after it is measured)
LUSTRE_DIR=${LUSTRE_DIR:-/cosma8/data/CHANGE_ME_PROJECT/$USER/eagle-rf-io}
VAST_DIR=${VAST_DIR:-/cosma/apps/CHANGE_ME_GROUP/$USER/eagle-rf-io}
SHM_DIR=/dev/shm/eagle-rf-io-${SLURM_JOB_ID:-local}

COMPILER_MODULE=gnu_comp/14.1.0   # COSMA8: likwid (and possibly darshan) refuse to load without a compiler module
MPI_MODULE=openmpi/5.0.3          # COSMA8: darshan-runtime refuses to load without an MPI module
LIKWID_MODULE=likwid/5.5.1
DARSHAN_RUNTIME_MODULE=darshan-runtime/3.4.6
DARSHAN_UTIL_MODULE=darshan-util/3.4.6

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
OUT="$HERE/highlevel_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"/{topology,core,scaling_pinned,numa_check,io}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS=("$LBOX" "$NPART" "$Z")
[ -n "$DATA_ROOT" ] && ARGS+=(--data-root "$DATA_ROOT")

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- Part A: topology
log "Part A: topology"
{
  hostname; lscpu; echo; lscpu -p=CPU,CORE,SOCKET,NODE
} > "$OUT/topology/lscpu.txt" 2>&1
python3 "$CPULIST" info > "$OUT/topology/layout.txt" 2>&1
cat "$OUT/topology/layout.txt"
NCORES=$(sed -n 's/.*physical cores: \([0-9]*\).*/\1/p' "$OUT/topology/layout.txt")
log "physical cores: $NCORES"

module load "$COMPILER_MODULE" "$LIKWID_MODULE" > "$OUT/topology/module_likwid.txt" 2>&1
command -v likwid-bench >/dev/null || log "LIKWID not available after module load (see topology/module_likwid.txt)"
likwid-topology > "$OUT/topology/likwid-topology.txt" 2>&1 || log "likwid-topology failed (see file)"

# ---------------------------------------------------------------- Part B: core rubric
log "Part B: core rubric"
# Scalar double-precision peak of one core (guidebook 5.2.2). Working set = half the 32 kB L1d.
L1D=$(lscpu | awk -F: '/^L1d cache/ {print $2}')
echo "L1d cache:$L1D" > "$OUT/core/l1d.txt"
likwid-bench -t peakflops        -w S0:16kB:1 > "$OUT/core/peak_scalar.txt"  2>&1 || log "peakflops failed"
# Vector peaks for context only (the guidebook normalises against the scalar peak)
likwid-bench -t peakflops_avx     -w S0:16kB:1 > "$OUT/core/peak_avx.txt"     2>&1 || true
likwid-bench -t peakflops_avx_fma -w S0:16kB:1 > "$OUT/core/peak_avx_fma.txt" 2>&1 || true
likwid-perfctr -a > "$OUT/core/groups.txt" 2>&1 || true

# The code, serial (n_jobs=1), pinned to the first core, whole program measured.
FIRST=$(python3 "$CPULIST" compact 1)
for GROUP in FLOPS_DP FLOPS_SP; do
  log "  likwid-perfctr $GROUP on CPU $FIRST"
  likwid-perfctr -f -C "$FIRST" -g "$GROUP" \
      python3 "$BENCH" "${ARGS[@]}" --outdir "$OUT/core" --tag "perfctr_$GROUP" \
      --n-jobs 1 --n-estimators "$N_EST" > "$OUT/core/perfctr_$GROUP.txt" 2>&1 \
      || log "  likwid-perfctr $GROUP failed (see core/perfctr_$GROUP.txt)"
done

# ---------------------------------------------------------------- Part C: intra-node rubric
log "Part C: pinned strong scaling"
LIST=${LIST:-"1 2 4 8 16 32 64 $NCORES"}
for P in $LIST; do
  CPUS=$(python3 "$CPULIST" compact "$P")
  for R in $(seq 1 "$REPEATS"); do
    taskset -c "$CPUS" python3 "$BENCH" "${ARGS[@]}" --outdir "$OUT/scaling_pinned" \
        --tag "nj${P}_r${R}" --n-jobs "$P" --n-estimators "$N_EST" \
        >> "$OUT/scaling_pinned/runs.log" 2>&1 || log "  run P=$P R=$R failed"
  done
  log "  P=$P done"
done
# All hardware threads (SMT): reported separately, not part of the core-count sweep
NALL=$(python3 "$CPULIST" all | tr ',' '\n' | wc -l)
for R in $(seq 1 "$REPEATS"); do
  python3 "$BENCH" "${ARGS[@]}" --outdir "$OUT/scaling_pinned" --tag "smt${NALL}_r${R}" \
      --n-jobs "$NALL" --n-estimators "$N_EST" >> "$OUT/scaling_pinned/smt.log" 2>&1 || true
done
mkdir -p "$OUT/scaling_pinned/smt" && mv "$OUT"/scaling_pinned/result_smt* "$OUT/scaling_pinned/smt/" 2>/dev/null

log "Part C2: NUMA check at 16 threads (exploratory)"
for MODE in compact spread; do
  CPUS=$(python3 "$CPULIST" "$MODE" 16)
  echo "$MODE: $CPUS" >> "$OUT/numa_check/cpus.txt"
  for R in $(seq 1 "$REPEATS"); do
    taskset -c "$CPUS" python3 "$BENCH" "${ARGS[@]}" --outdir "$OUT/numa_check" \
        --tag "${MODE}16_r${R}" --n-jobs 16 --n-estimators "$N_EST" \
        >> "$OUT/numa_check/runs.log" 2>&1 || true
  done
done

# ---------------------------------------------------------------- Part D: I/O rubric
log "Part D: I/O"
CPUS16=$(python3 "$CPULIST" compact "$IO_CORES")
df -hT /dev/shm "$(dirname "$LUSTRE_DIR")" "$(dirname "$VAST_DIR")" > "$OUT/io/filesystems.txt" 2>&1

io_run() {   # io_run <label> <dir> <repeat>
  local label=$1 dir=$2 r=$3
  mkdir -p "$dir"
  taskset -c "$CPUS16" python3 "$BENCH" "${ARGS[@]}" --outdir "$dir" --tag "io_${label}_r${r}" \
      --n-jobs "$IO_CORES" --n-estimators "$N_EST" --save-model --save-arrays --plot \
      >> "$OUT/io/runs.log" 2>&1 || log "  io run $label r$r failed"
  cp "$dir"/result_io_"${label}"_r"${r}".json "$OUT/io/" 2>/dev/null
  rm -f "$dir"/*.pkl "$dir"/*.npy "$dir"/*.pdf "$dir"/result_*.json
}
for R in 1 2; do
  io_run shm    "$SHM_DIR"    "$R"
  io_run lustre "$LUSTRE_DIR" "$R"
  io_run vast   "$VAST_DIR"   "$R"
done
rm -rf "$SHM_DIR"

log "Part D2: Darshan (one run, Lustre)"
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
python3 "$HERE/summarise_highlevel.py" "$OUT" || log "summary failed"
log "Done: $OUT"
