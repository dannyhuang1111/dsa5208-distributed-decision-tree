# Histogram-based distributed decision tree (MPI + OpenMP)

`src/dt.cpp` is one source file that builds three ways (serial, MPI, MPI+OpenMP).
It replaces the split search of the old `tree_mpi_omp.cpp`; see "What was wrong" below.

## Build

```bash
make -f Makefile.dt            # build/dt_serial  build/dt_hybrid
```

* Linux / Atlas: needs `g++` and an MPI compiler wrapper (`module load OpenMPI ...`).
* macOS: Apple clang has no OpenMP, so the Makefile uses Homebrew's libomp (`brew install libomp`)
  and sets `MPICH_CXX=clang++` (the conda MPICH wrapper points at a compiler that is not installed).
  If you see `OMP: Error #15 ... libomp already initialized`, run with `KMP_DUPLICATE_LIB_OK=TRUE`.

## Run

```bash
# serial baseline (no MPI, no OpenMP) on 5M synthetic rows, 28 features
build/dt_serial --synthetic 5000000 --features 28

# 8 ranks x 4 threads, level-wise growth, profiled (barrier before each Allreduce)
OMP_NUM_THREADS=4 mpirun -np 8 build/dt_hybrid --synthetic 5000000 --features 28 --profile 1

# real data: CSV with the label in column 1 and a header row (preprocess.py output)
OMP_NUM_THREADS=2 mpirun -np 4 build/dt_hybrid --csv instacart_clean.csv
```

Options: `--growth node|level`, `--omp sample|feature`, `--depth 6`, `--min-split 20`,
`--reps 3`, `--warmup 1`, `--profile 0|1`, `--check 0|1`, `--dump-tree FILE`, `--tag NAME`.
`build/dt_serial --header` prints the column names of the `RESULT,...` lines.

Synthetic data is a deterministic function of (seed, row number), so the dataset is the same for every rank
count, and nothing has to be read from disk.  `--export-csv FILE` (with `--synthetic`, up to 5M rows) writes
the same data as CSV for scikit-learn.

## Validation

```bash
scripts/check_equivalence.sh          # 64 configurations vs the serial reference tree
build/dt_serial --csv data.csv --dump-tree tree.txt --reps 1 --warmup 0
python3 scripts/compare_sklearn.py data.csv tree.txt     # whole-tree comparison with scikit-learn
```

`check_equivalence.sh` runs rank counts 1, 2, 3, 4, 7, thread counts 1, 2, 3, both growth modes, both OpenMP
strategies, on N = 100,003 rows (a prime, so the row blocks are uneven) and through the CSV input path, and
requires the FNV-1a hash of the entire tree to equal the serial reference.

Run `compare_sklearn.py` on the **real Instacart CSV** as well; it has not been run on it yet (the real data was
not on this machine).  Differences can only come from exact ties between equally good splits.

## What was wrong with the old MPI split search

Rows are sharded across ranks, so the left / right class counts of a threshold are sums over ranks.  The old code
computed the gain from rank-local left counts combined with global totals, then took `MPI_MAXLOC` over those
local gains.  That is not the global Gini gain.

Measured on the same 200,000 synthetic rows (old `tree_mpi_omp.cpp`, 1 thread per rank):

| ranks | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| training accuracy, old MPI | 68.46 % | 68.45 % | 68.44 % | 68.42 % |
| training accuracy, new MPI | 68.46 % | 68.46 % | 68.46 % | 68.46 % |

The old accuracy drifts with the rank count, so the trees differ.  The root split was still correct in every
case (feature 4, threshold 3.5), so checking only the root and first-level splits cannot detect the problem.

## How it works

For every node: each rank counts (rows, positives) per feature bin of its own rows; one
`MPI_Allreduce(SUM)` of that integer histogram gives the global histogram; every rank scans the same global
histogram with the same code and picks the same split.  The sum of integers does not depend on the order of
the reduction, so the tree is bit-for-bit identical for any number of ranks and threads.

* `--growth node`: depth first, one Allreduce per internal node (latency bound).
* `--growth level`: breadth first, one Allreduce per level for all nodes of that level (fewer, larger messages).
* `--omp sample`: threads split the rows, private histograms, then a parallel reduce.
* `--omp feature`: threads split the features, shared histogram (parallelism is limited by F).

Features are binned by their distinct values.  This is exact (same candidate thresholds as sorting) when every
feature has at most 256 distinct values, which holds for the Instacart features.  Otherwise the program uses
quantile-like bins and prints `APPROXIMATE bins`; the trees are still identical across configurations but may
differ from scikit-learn.

The tree stops at `--depth`, when a node has at most `--min-split` rows, when it is pure, or when no split has
positive gain: the same rules as the old serial code.  scikit-learn's equivalent of `--min-split 20` is
`min_samples_split=21`.

## Experiments

```bash
scripts/run_matrix.sh smoke                     # a minute: does everything work?
scripts/run_matrix.sh strong weak hybrid design
python3 analysis/plot_results.py --raw results/raw --out results/report_assets
```

On Atlas use `qsub scripts/job_scaling.pbs` (read the comments in it first: queue name, cores per node, module
names and rank binding are unverified).  Every number in the report must come from `results/raw/*.csv`;
`plot_results.py` reads nothing else and refuses to plot if two configurations built different trees.

What to say in the report about the measurements: which machine (see `results/raw/env_*.txt`), how many
repetitions, that `t_total` is the maximum over ranks from barrier to barrier, and that the baseline for speedup
is the plain serial build on the same data.  The CUDA kernel (`src/tree_cuda.cu`) still uses the old approach and
is not covered by any of this.
