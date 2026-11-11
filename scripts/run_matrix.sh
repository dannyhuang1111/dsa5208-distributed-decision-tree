#!/usr/bin/env bash
# Experiment driver: every number in the report comes from the files this script writes.
#
#   scripts/run_matrix.sh smoke              # tiny sanity run, do this first
#   scripts/run_matrix.sh strong weak hybrid design [real]
#
# Output (never overwritten; every run has its own RUN_ID):
#   results/raw/<experiment>_<RUN_ID>.csv    one RESULT row per repetition
#   results/logs/*.log                       full stdout/stderr of every launch
#   results/raw/env_<RUN_ID>.txt             machine / MPI / compiler / parameters
#
# Experiments
#   serial_scan  serial (1 core) runtime vs problem size: N (rows) at fixed F,
#                and F (features) at fixed N  (varies BOTH dimensions)
#   strong  fixed problem (N_STRONG x F): pure MPI, threads-only, MPI x HYB_THREADS threads
#   weak    N grows with the core count (N_PER_RANK rows per rank, per core for hybrid)
#   hybrid  fixed core count HYBRID_CORES, vary ranks x threads
#   design  growth node|level  x  omp sample|feature, at two configurations
#   real    strong scaling on a real CSV (set REAL_CSV=/path/instacart_clean.csv)
#
# Every configuration is run REPS times inside one process (after a warm-up tree)
# without profiling, and PROF_REPS times with --profile 1 (barrier before each
# Allreduce, so waiting for the slowest rank and the transfer itself are timed
# separately).  Use LAUNCHES=3 to repeat whole launches as well.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SER=${SER:-$ROOT/build/dt_serial}
HYB=${HYB:-$ROOT/build/dt_hybrid}
RESULTS=${RESULTS:-$ROOT/results}
RAW=$RESULTS/raw; LOGS=$RESULTS/logs
mkdir -p "$RAW" "$LOGS"

MPIRUN=${MPIRUN:-mpirun}
REPS=${REPS:-5}
PROF_REPS=${PROF_REPS:-3}
LAUNCHES=${LAUNCHES:-1}
DEPTH=${DEPTH:-6}
F=${F:-28}
# F_SWEEP: feature counts swept in strong+weak scaling (shows scaling is F-stable).
# Set F_SWEEP="28" to disable the sweep and run a single feature count.
F_SWEEP=${F_SWEEP:-"7 14 28 56"}
N_STRONG=${N_STRONG:-64000000}
N_PER_RANK=${N_PER_RANK:-2000000}
RANK_LIST=${RANK_LIST:-"1 2 4 8 16 32 48 64 96"}
THREAD_LIST=${THREAD_LIST:-"1 2 4 8 16 32 48 96"}
HYB_THREADS=${HYB_THREADS:-4}
HYBRID_CORES=${HYBRID_CORES:-48}
REAL_CSV=${REAL_CSV:-}
# serial_scan: rows swept at fixed F, and features swept at fixed N.
SCAN_N=${SCAN_N:-"1000000 2000000 4000000 8000000 16000000 32000000"}
SCAN_F_FIXED=${SCAN_F_FIXED:-28}
SCAN_F=${SCAN_F:-"7 14 28 56 112 224"}
SCAN_N_FIXED=${SCAN_N_FIXED:-8000000}
TMO=${TMO:-1800}                       # seconds per launch (needs `timeout`)

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
[ -n "${PBS_JOBID:-}" ] && RUN_ID="${RUN_ID}_${PBS_JOBID%%.*}"

if [ -z "${MAX_CORES:-}" ]; then
  if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then MAX_CORES=$(wc -l < "$PBS_NODEFILE")
  else MAX_CORES=$( (nproc 2>/dev/null || sysctl -n hw.ncpu) ); fi
fi

# Cores per node (largest PBS chunk); a rank's threads cannot span nodes.
if [ -z "${CORES_PER_NODE:-}" ]; then
  if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then CORES_PER_NODE=$(sort "$PBS_NODEFILE" | uniq -c | awk 'BEGIN{m=0} $1>m{m=$1} END{print m}')
  else CORES_PER_NODE=$MAX_CORES; fi
fi

# Rank binding (Open MPI).  On Vanda compute nodes, "--map-by slot:PE=N" (and
# --cpus-per-proc / ppr:..:PE=N) HANG under the PBS cpuset, so we avoid PE=:
#   threads == 1 : --map-by core --bind-to core      (one core per rank, pinned)
#   threads  > 1 : --map-by ppr:<CORES_PER_NODE/threads>:node --bind-to none,
#                  OMP_PROC_BIND=false.  ppr fills one node before using the next
#                  (plain --bind-to none puts ALL ranks on the first node: with
#                  64 ranks x 4 threads that is 256 threads on 64 cores).  Threads
#                  are placed by the kernel inside the PBS cpuset; pinning every
#                  rank's threads to the same cores (OMP_PROC_BIND=close) was ~20x slower.
# Set MPI_BIND yourself to override (%T = thread count), e.g. on a machine where
# PE= works:  MPI_BIND='--map-by slot:PE=%T --bind-to core'
if [ -z "${MPI_BIND+x}" ]; then
  if $MPIRUN --version 2>&1 | grep -qi "open mpi"; then MPI_BIND=AUTO; else MPI_BIND=''; fi
fi
TIMEOUT=""; command -v timeout >/dev/null 2>&1 && TIMEOUT="timeout $TMO"
export KMP_DUPLICATE_LIB_OK=TRUE       # macOS only (conda + Homebrew both ship a libomp); harmless elsewhere

ENVFILE=$RAW/env_$RUN_ID.txt
{
  echo "run_id=$RUN_ID"; echo "date=$(date -u +%FT%TZ)"; echo "host=$(hostname)"
  echo "max_cores=$MAX_CORES"; echo "mpirun=$($MPIRUN --version 2>&1 | head -2 | tr '\n' ' ')"
  echo "mpi_bind=$MPI_BIND (AUTO: thr=1 -> --map-by core --bind-to core; thr>1 -> --map-by ppr:CORES_PER_NODE/thr:node --bind-to none, OMP_PROC_BIND=false) cores_per_node=$CORES_PER_NODE"; echo "cxx=$(${CXX:-c++} --version 2>&1 | head -1)"
  echo "lscpu=$(lscpu 2>/dev/null | grep -E 'Model name|Socket|Core|Thread' | tr '\n' ';')"
  echo "pbs_jobid=${PBS_JOBID:-}"; echo "pbs_nodefile_lines=${MAX_CORES}"
  echo "params: REPS=$REPS PROF_REPS=$PROF_REPS LAUNCHES=$LAUNCHES DEPTH=$DEPTH F=$F N_STRONG=$N_STRONG N_PER_RANK=$N_PER_RANK"
  echo "params: RANK_LIST=$RANK_LIST THREAD_LIST=$THREAD_LIST HYB_THREADS=$HYB_THREADS HYBRID_CORES=$HYBRID_CORES REAL_CSV=$REAL_CSV"
} > "$ENVFILE"
echo "[run_matrix] RUN_ID=$RUN_ID  MAX_CORES=$MAX_CORES  results -> $RESULTS"

CSV=""
begin() {                               # experiment name
  CSV=$RAW/${1}_$RUN_ID.csv
  "$SER" --header > "$CSV"
  echo "[run_matrix] === $1 -> $CSV"
}

# launch <tag> <ranks> <threads> <profile> <growth> <omp> <data args...>
launch() {
  local tag=$1 ranks=$2 thr=$3 prof=$4 growth=$5 omp=$6; shift 6
  local cores=$((ranks * thr))
  if [ "$cores" -gt "$MAX_CORES" ]; then echo "  [skip] $tag ranks=$ranks threads=$thr needs $cores > $MAX_CORES cores"; return; fi
  if [ "$thr" -gt "$CORES_PER_NODE" ]; then echo "  [skip] $tag ranks=$ranks threads=$thr threads per rank > $CORES_PER_NODE cores per node"; return; fi
  local reps=$REPS; [ "$prof" = "1" ] && reps=$PROF_REPS
  local flags omp_bind=close
  if [ "$MPI_BIND" = "AUTO" ]; then
    if [ "$thr" -eq 1 ]; then flags='--map-by core --bind-to core'
    else flags="--map-by ppr:$((CORES_PER_NODE / thr)):node --bind-to none"; omp_bind=false; fi
  else flags=${MPI_BIND//%T/$thr}; fi
  local l
  for ((l = 1; l <= LAUNCHES; l++)); do
    local log="$LOGS/${tag}_r${ranks}_t${thr}_p${prof}_${growth}_${omp}_L${l}_$RUN_ID.log"
    OMP_NUM_THREADS=$thr OMP_PROC_BIND=$omp_bind OMP_PLACES=cores \
      $TIMEOUT $MPIRUN -np "$ranks" $flags "$HYB" --tag "$tag" --growth "$growth" --omp "$omp" \
        --profile "$prof" --reps "$reps" --depth "$DEPTH" "$@" > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then echo "  [FAIL rc=$rc] $tag ranks=$ranks threads=$thr -> $log"; continue; fi
    grep '^RESULT,' "$log" >> "$CSV"
    echo "  ok  $tag ranks=$ranks threads=$thr profile=$prof $growth/$omp ($(grep -c '^RESULT,' "$log") rows)"
  done
}

# launch_serial <tag> <data args...>   (plain build: no MPI, no OpenMP)
launch_serial() {
  local tag=$1; shift
  local l
  for ((l = 1; l <= LAUNCHES; l++)); do
    local log="$LOGS/${tag}_serial_L${l}_$RUN_ID.log"
    $TIMEOUT "$SER" --tag "$tag" --reps "${SERIAL_REPS:-3}" --depth "$DEPTH" "$@" > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then echo "  [FAIL rc=$rc] $tag serial -> $log"; continue; fi
    grep '^RESULT,' "$log" >> "$CSV"
    echo "  ok  $tag serial"
  done
}

exp_smoke() {
  begin smoke
  local D=(--synthetic 2000000 --features 7)
  launch_serial smoke_serial "${D[@]}"
  for r in 1 2 4; do launch smoke_mpi "$r" 1 0 level sample "${D[@]}"; done
  launch smoke_hyb 2 2 1 level sample "${D[@]}"
}

# serial_scan  serial (1 core) runtime vs problem size, varying BOTH dimensions:
#   N (rows) at fixed F, and F (features) at fixed N.  Cost per node is
#   O(n_rows x F), so each sweep should be ~linear.  Serial build only.
exp_serial_scan() {
  begin serial_scan
  local n f
  for n in $SCAN_N; do launch_serial serial_scan --synthetic "$n" --features "$SCAN_F_FIXED"; done
  for f in $SCAN_F; do launch_serial serial_scan --synthetic "$SCAN_N_FIXED" --features "$f"; done
}


exp_strong() {
  begin strong
  local r t fv
  # F sweep: repeat the whole strong-scaling study at each feature count.  The
  # F column in each RESULT row lets plot_results.py separate the curves.
  for fv in $F_SWEEP; do
    local D=(--synthetic "$N_STRONG" --features "$fv")
    launch_serial strong_serial "${D[@]}"
    for r in $RANK_LIST; do launch strong_mpi "$r" 1 0 level sample "${D[@]}"; done
    for r in $RANK_LIST; do launch strong_mpi "$r" 1 1 level sample "${D[@]}"; done
    for t in $THREAD_LIST; do launch strong_thr 1 "$t" 0 level sample "${D[@]}"; done
    for r in $RANK_LIST; do launch "strong_hyb$HYB_THREADS" "$r" "$HYB_THREADS" 0 level sample "${D[@]}"; done
    for r in $RANK_LIST; do launch "strong_hyb$HYB_THREADS" "$r" "$HYB_THREADS" 1 level sample "${D[@]}"; done
  done
}

exp_weak() {
  begin weak
  local r fv
  for fv in $F_SWEEP; do
    for r in $RANK_LIST; do
      launch weak_mpi "$r" 1 0 level sample --synthetic $((N_PER_RANK * r)) --features "$fv"
      launch weak_mpi "$r" 1 1 level sample --synthetic $((N_PER_RANK * r)) --features "$fv"
    done
    for r in $RANK_LIST; do
      launch "weak_hyb$HYB_THREADS" "$r" "$HYB_THREADS" 0 level sample --synthetic $((N_PER_RANK * r * HYB_THREADS)) --features "$fv"
    done
  done
}

exp_hybrid() {
  begin hybrid
  local D=(--synthetic "$N_STRONG" --features "$F")
  local t
  for t in 1 2 4 8 16 32 64 128; do
    if [ "$t" -le "$HYBRID_CORES" ] && [ $((HYBRID_CORES % t)) -eq 0 ]; then
      launch hybrid_fixed $((HYBRID_CORES / t)) "$t" 0 level sample "${D[@]}"
      launch hybrid_fixed $((HYBRID_CORES / t)) "$t" 1 level sample "${D[@]}"
    fi
  done
}

exp_design() {
  begin design
  local D=(--synthetic "$N_STRONG" --features "$F")
  local cfg r t g o p
  for cfg in "$HYBRID_CORES:1" "$((HYBRID_CORES / 4)):4"; do
    r=${cfg%%:*}; t=${cfg##*:}
    for g in node level; do for o in sample feature; do for p in 0 1; do
      launch design "$r" "$t" "$p" "$g" "$o" "${D[@]}"
    done; done; done
  done
}

exp_real() {
  if [ -z "$REAL_CSV" ] || [ ! -f "$REAL_CSV" ]; then echo "[run_matrix] real: set REAL_CSV to an existing CSV, skipping"; return; fi
  begin real
  local D=(--csv "$REAL_CSV")
  launch_serial real_serial "${D[@]}"
  local r
  for r in $RANK_LIST; do launch real_mpi "$r" 1 0 level sample "${D[@]}"; done
  for r in $RANK_LIST; do launch real_mpi "$r" 1 1 level sample "${D[@]}"; done
}

[ $# -eq 0 ] && { sed -n '2,24p' "$0"; exit 1; }
for e in "$@"; do
  case "$e" in
    smoke) exp_smoke;; serial_scan) exp_serial_scan;; strong) exp_strong;; weak) exp_weak;; hybrid) exp_hybrid;; design) exp_design;; real) exp_real;;
    all) exp_smoke; exp_serial_scan; exp_strong; exp_weak; exp_hybrid; exp_design; exp_real;;
    *) echo "unknown experiment '$e'"; exit 2;;
  esac
done
echo "[run_matrix] done. Next: python3 analysis/plot_results.py --raw $RAW"
