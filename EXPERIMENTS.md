# Experiment setup

The single reference for what we run, on what, and why. The report's methodology
and results sections follow this. Four **configurations** of one algorithm
(histogram-based CART, `src/dt.cpp`; CUDA in `src/tree_cuda.cu`), each measured
with the study that fits it.

## The four configurations

| Config | Binary | Parallelism | Where it runs |
|---|---|---|---|
| serial | `dt_serial` | none (1 core) | baseline |
| pure MPI | `dt_hybrid`, `OMP_NUM_THREADS=1` | MPI only | CPU nodes |
| MPI+OpenMP | `dt_hybrid`, `OMP_NUM_THREADS>1` | MPI + threads | CPU nodes |
| CUDA | `dt_cuda` | one GPU | GPU node |

serial, pure MPI and MPI+OpenMP are the **same source** (`dt.cpp`) compiled/run
differently. CUDA is separate code.

## What each configuration measures

Only the two MPI configs have a "number of cores" axis, so only they are
*scaled*. serial and CUDA are each a **single run** used for comparison.

| Config | Study | Varies | Fixed |
|---|---|---|---|
| serial | problem-size scan | N (rows), then F (features) | 1 core |
| serial | baseline point | — | N, F fixed; 1 core |
| pure MPI | strong scaling | cores 1 -> 256 | N, F |
| pure MPI | weak scaling | cores 1 -> 256 | rows/core, F |
| MPI+OpenMP | strong scaling | cores 1 -> 256 (ranks x threads) | N, F |
| MPI+OpenMP | weak scaling | cores 1 -> 256 | rows/core, F |
| MPI+OpenMP | ranks x threads | split at fixed cores | N, F |
| MPI+OpenMP | growth x omp strategy | node/level, sample/feature | N, F |
| CUDA | GPU benchmark | — (one run) | N, F; 1 GPU |

### Strong vs weak scaling (applies to the two MPI configs)
- **Strong**: fix the problem (N=32M), add cores. Ideal: time falls as 1/cores,
  speedup -> cores. Reports speedup `S=T_1/T_p` and efficiency `E=S/p`.
- **Weak**: fix work per core (1M rows/core), grow N with cores. Ideal: flat
  time, efficiency -> 1.

## Varying F (feature count)

F is the number of feature columns of the synthetic dataset (the 7 Instacart
feature domains, cycled when F>7). Required by the rubric for serial; we also
sweep it in strong+weak scaling to show the scaling is F-stable. hybrid and
design keep F fixed (F does not change their conclusion).

- F sweep values: **{7, 14, 28, 56}** for strong+weak; serial scan also uses
  112, 224.

## Sizing (to keep the longest run ~10 min on a cluster assumed ~3x slower)

Anchored on the measured 48-core run (1 core, N=32M, F=28 ~= 5-6 s/rep):

| Parameter | Value |
|---|---|
| N (strong), fixed problem | 32,000,000 |
| rows per core (weak) | 1,000,000 |
| F sweep (strong/weak) | 7 14 28 56 |
| reps | 3 |
| cores | 1 2 4 8 16 32 64 128 256 |

Longest single run (largest F, 1 core, all reps) ~= 100 s on the slow cluster;
each scaling experiment ~4-6 min. All values are overridable at `qsub` time via
`-v VAR=...`; run a `smoke` job first to confirm real timing.

## CUDA benchmark: what it measures and how we compare

One GPU, one problem (N=32M, F=28). The kernel builds the **same**
`[count, positive-count]` histogram the CPU builds, so the chosen split matches
the CPU result (correctness check). We time and report:

- **kernel time** (GPU compute, histogram build);
- **transfer time** (host->device data, device->host histogram);
- best split found (must match the CPU's).

Compared against, at the same N and F:
1. one CPU core (serial) -> raw speedup;
2. a full MPI+OpenMP node -> the honest "worth it vs the CPUs we have?" number.

Expected story (confirm with the real run): the kernel is much faster than one
core; against a full node the margin is smaller and may be dominated by
host<->device transfer, so a GPU pays off when data stays resident and is reused
across many node splits rather than for a single histogram.

## Real data

One serial run on `instacart_clean.csv` (200k rows, real 7 features) for the
"conclusions about the data" section: accuracy vs the 66.2% majority-class
baseline (serial gives ~68.5%). Too small for 256-core scaling, hence synthetic
data for all scaling studies.

## How it runs

- CPU configs (serial, pure MPI, MPI+OpenMP): `scripts/run_matrix.sh`, driven on
  the cluster by `scripts/job_scaling.pbs` (`qsub`).
- CUDA: build with `make -f Makefile.dt cuda` (needs nvcc); run on a GPU node via
  `scripts/job_gpu.pbs` (`qsub`). It also runs the serial baseline on the same
  N,F so the GPU-vs-one-core comparison is in the same CSV (`results/raw/gpu_*.csv`).
- Plots/tables: `python3 analysis/plot_results.py --raw results/raw`.
