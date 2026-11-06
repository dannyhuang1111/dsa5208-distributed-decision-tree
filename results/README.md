# results/

* `raw/*.csv`      one row per repetition, written by `scripts/run_matrix.sh` (never edited by hand)
* `raw/env_*.txt`  machine, MPI, compiler and parameters of that run
* `logs/*.log`     complete stdout / stderr of every launch
* `report_assets/` figures and tables produced by `python3 analysis/plot_results.py`

Columns of the raw CSV (`build/dt_serial --header`):

| column | meaning |
|---|---|
| tag | experiment label (`strong_mpi`, `weak_hyb4`, `design`, ...) |
| impl, ranks, threads, cores | build (`serial`/`mpi`), MPI ranks, OpenMP threads per rank, ranks x threads |
| growth, omp | `node`/`level` tree growth, `sample`/`feature` OpenMP strategy |
| profile | 1 = barrier before every Allreduce (wait and transfer timed separately) |
| N, F, depth | rows, features, tree depth |
| rep | repetition inside one process (after a warm-up tree that is not recorded) |
| t_total | wall time of training (max over ranks, barrier to barrier) |
| t_hist, t_scan, t_part | mean over ranks: local histograms, split search, partition / slot update |
| t_wait, t_allreduce | mean over ranks, only meaningful when profile=1 |
| hist_imbalance | max over ranks / mean over ranks of t_hist (1.0 = perfectly balanced) |
| accuracy | training accuracy of the tree |
| tree_hash | FNV-1a hash of the tree structure; must be identical for all runs on the same data |
| n_nodes, n_allreduce, mbytes_allreduce | tree size, number and total volume of Allreduce calls |
| approx | 1 if a feature had > 256 distinct values (bins are then approximate) |
| t_setup | data generation / loading + binning time, not part of t_total |
