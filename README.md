# Histogram-based distributed decision tree (MPI + OpenMP, CUDA)

DSA5208 final project. A binary CART decision tree (Gini) implemented once in `src/dt.cpp` and compiled three ways from the same source:

| build | parallelism | purpose |
|---|---|---|
| `dt_serial` | none (1 core) | baseline for speedup / efficiency |
| `dt_hybrid` (pure MPI) | MPI only (`OMP_NUM_THREADS=1`) | distributed, data-parallel |
| `dt_hybrid` (MPI+OpenMP) | MPI + OpenMP | hybrid |

A CUDA version (`src/tree_cuda.cu` / `src/dt_gpu.cu`) implements the GPU arm of the MPI+X study on an NVIDIA A40 GPU (see *Status* below).

## Build

```bash
make -f Makefile.dt            # -> build/dt_serial  build/dt_hybrid
make -f Makefile.dt cuda       # -> build/dt_cuda    (GPU; needs nvcc)
```

* Linux / Atlas: needs `g++` (C++17) and an MPI wrapper (`module load foss/2022b` gives GCC 12 + OpenMPI 4.1 on Atlas).
* macOS: Apple clang has no bundled OpenMP, so the Makefile uses Homebrew's libomp (`brew install libomp`). If you see `OMP: Error #15 ... already initialized`, run with `KMP_DUPLICATE_LIB_OK=TRUE`.

## Run

```bash
# serial baseline: 5M synthetic rows, 28 features
build/dt_serial --synthetic 5000000 --features 28

# 8 ranks x 4 threads, level-wise growth, profiled (barrier before each Allreduce)
OMP_NUM_THREADS=4 mpirun -np 8 build/dt_hybrid --synthetic 5000000 --features 28 --profile 1

# CSV input path: label in column 1, header row (preprocess.py output)
OMP_NUM_THREADS=2 mpirun -np 4 build/dt_hybrid --csv instacart_clean.csv
```

Options: `--growth node|level`, `--omp sample|feature`, `--depth 6`, `--min-split 20`, `--reps 3`, `--warmup 1`, `--profile 0|1`, `--check 0|1`, `--dump-tree FILE`, `--export-csv FILE`, `--tag NAME`. `build/dt_serial --header` prints the column names of the `RESULT,...` lines.

Synthetic data is a deterministic function of (seed, row number), so the dataset is identical for every rank count and nothing is read from disk.

## How it works

For every node: each rank counts (rows, positives) per feature bin over its own rows; one `MPI_Allreduce(SUM)` of that integer histogram yields the global histogram; every rank scans the same global histogram and picks the same split. Integer sums are order-independent, so the tree is **bit-for-bit identical** for any number of ranks and threads (verified below).

Two orthogonal parallelization axes are studied:

* `--growth node` — depth first, one Allreduce per internal node (latency bound).
* `--growth level` — breadth first, one Allreduce per level (fewer, larger messages).
* `--omp sample` — threads split rows, private histograms, then parallel reduce.
* `--omp feature` — threads split features, shared histogram (limited by F).

Features are binned by distinct value: exact (same candidate thresholds as a sort) when a feature has <= 256 distinct values, which holds for this dataset; otherwise quantile bins are used and `APPROXIMATE bins` is printed.

## Validation

```bash
scripts/check_equivalence.sh          # many configs vs the serial reference tree
build/dt_serial --csv data.csv --dump-tree tree.txt --reps 1 --warmup 0
python3 scripts/compare_sklearn.py data.csv tree.txt     # whole-tree vs scikit-learn
```

`check_equivalence.sh` runs rank counts {1,2,3,4,7}, thread counts {1,2,3}, both growth modes, both OpenMP strategies, on a prime number of rows (uneven blocks) and through the CSV path, requiring the FNV-1a hash of the entire tree to equal the serial reference.

## Experiments

```bash
scripts/run_matrix.sh smoke                     # ~1 min sanity check, do this first
scripts/run_matrix.sh strong weak hybrid design
python3 analysis/plot_results.py --raw results/raw --out results/report_assets
```

On the cluster: `qsub scripts/job_scaling.pbs` (read its comments first — queue, cores/node, module names and rank binding must be confirmed on the machine). Every number in the report comes from `results/raw/*.csv`; `plot_results.py` reads nothing else and refuses to plot if two configurations built different trees. Machine / MPI / compiler details for each run are recorded in `results/raw/env_*.txt`.

## Repository layout

```text
src/dt.cpp              # the algorithm (serial / MPI / MPI+OpenMP, one source)
src/tree_cuda.cu        # GPU arm (CUDA histogram + split scan on NVIDIA A40)
src/dt_gpu.cu           # standalone CUDA benchmark binary
src/common.h            # shared helpers used by the CUDA build
Makefile.dt             # build system for dt.cpp and tree_cuda.cu
scripts/                # run_matrix.sh, check_equivalence.sh, compare_sklearn.py, job_scaling.pbs, job_gpu.pbs
analysis/plot_results.py
results/                # raw CSVs, logs, generated report assets (Vanda 256-core + A40 GPU)
report/                 # report source (report.tex, report.pdf)
preprocess.py           # generates the synthetic instacart_clean.csv sample
```

## Status

* Serial, pure-MPI, and MPI+OpenMP: complete and validated up to 256 cores across 4 dual-socket Intel Xeon Platinum 8452Y nodes on NUS Vanda HPC.
* CUDA (`src/tree_cuda.cu` / `src/dt_gpu.cu`): complete, verified, and benchmarked on a Vanda NVIDIA A40 GPU (`results/raw/gpu_*.csv`), matching the CPU root split identically (`64.44x` compute speedup at `N=2M`, `6.46x` at `N=32M`).
* All 256-core strong/weak/hybrid/design benchmarks and NVIDIA A40 GPU benchmarks are stored in `results/raw/` and plotted into `results/report_assets/`.
* Single-node run on the 200k-row `instacart_clean.csv` sample: serial trains to 68.50% accuracy (majority-class baseline is 66.17%) and matches `scikit-learn` to 99.985% prediction agreement.

## Group members

* Jen-Chien Huang (A0353871X)
* Bing Wen Lim (A0136216M)
* Haoran Zhang (A0355802E)
