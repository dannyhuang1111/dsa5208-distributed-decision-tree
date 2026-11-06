#!/usr/bin/env bash
# Check that every build / rank count / thread count / growth mode / OpenMP
# strategy produces the IDENTICAL tree (same FNV-1a hash of the whole tree).
#
#   scripts/check_equivalence.sh                 # defaults: N=100003 (prime -> uneven row blocks)
#   N=2000000 RANKS="1 2 4 8" THREADS="1 2 4" scripts/check_equivalence.sh
#
# Exit status 0 only if every configuration matches the serial reference.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SER=${SER:-$ROOT/build/dt_serial}
HYB=${HYB:-$ROOT/build/dt_hybrid}
MPIRUN=${MPIRUN:-mpirun}
N=${N:-100003}
F=${F:-7}
DEPTH=${DEPTH:-6}
RANKS=${RANKS:-"1 2 3 4 7"}
THREADS=${THREADS:-"1 2 3"}
export KMP_DUPLICATE_LIB_OK=TRUE     # macOS only: conda and Homebrew both ship a libomp

hash_of() { grep '^RESULT,' | head -1 | cut -d, -f22; }

ref=$("$SER" --synthetic "$N" --features "$F" --depth "$DEPTH" --reps 1 --warmup 0 --quiet --growth level 2>&1 | hash_of)
if [ -z "$ref" ]; then echo "serial reference failed"; exit 2; fi
echo "reference (serial, level-wise): tree hash $ref   N=$N F=$F depth=$DEPTH"
echo

total=0; bad=0
check() {  # label hash
    total=$((total + 1))
    if [ "$2" = "$ref" ]; then printf "  PASS  %s\n" "$1"; else printf "  FAIL  %s  (got '%s')\n" "$1" "$2"; bad=$((bad + 1)); fi
}

echo "serial build, node-wise growth"
h=$("$SER" --synthetic "$N" --features "$F" --depth "$DEPTH" --reps 1 --warmup 0 --quiet --growth node 2>&1 | hash_of)
check "serial growth=node" "$h"

echo "MPI + OpenMP build"
for r in $RANKS; do
  for t in $THREADS; do
    for g in level node; do
      for o in sample feature; do
        h=$(OMP_NUM_THREADS=$t "$MPIRUN" -np "$r" "$HYB" --synthetic "$N" --features "$F" --depth "$DEPTH" \
              --reps 1 --warmup 0 --quiet --check 1 --growth "$g" --omp "$o" 2>&1 | hash_of)
        check "ranks=$r threads=$t growth=$g omp=$o" "$h"
      done
    done
  done
done

echo "CSV input path (same data written to CSV and read back)"
tmpd=$(mktemp -d)
"$SER" --synthetic "$N" --features "$F" --depth "$DEPTH" --reps 1 --warmup 0 --quiet --export-csv "$tmpd/data.csv" >/dev/null 2>&1
for r in 1 3; do
  h=$(OMP_NUM_THREADS=2 "$MPIRUN" -np "$r" "$HYB" --csv "$tmpd/data.csv" --depth "$DEPTH" --reps 1 --warmup 0 --quiet --check 1 2>&1 | hash_of)
  check "csv ranks=$r threads=2" "$h"
done
h=$("$SER" --csv "$tmpd/data.csv" --depth "$DEPTH" --reps 1 --warmup 0 --quiet 2>&1 | hash_of)
check "csv serial" "$h"
rm -rf "$tmpd"

echo
echo "$((total - bad)) / $total configurations identical to the reference"
[ "$bad" -eq 0 ]
