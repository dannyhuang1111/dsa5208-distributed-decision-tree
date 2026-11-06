# Distributed Decision Tree on Instacart E-Commerce Dataset (MPI + OpenMP + CUDA)

This project implements a **3-Level Hybrid Parallel Decision Tree** in C++ using **MPI, OpenMP, and CUDA**, specifically tailored for massive tabular e-commerce customer behavior analysis (Instacart Market Basket Reorder Prediction, ~1.38M records).

Tested and evaluated on the **Atlas / Vanda HPC Cluster** scaling up to **256 CPU cores** and **NVIDIA GPUs**.

---

## 1. Project Architecture

The parallelization is structured into three coordinated levels:
1. **Level 1 (Inter-Node / MPI Data-Parallelism)**: Data sharding across up to 64 MPI nodes with distributed split finding via `MPI_Allreduce` / `MPI_Bcast`.
2. **Level 2 (Intra-Node / OpenMP Feature-Parallelism)**: Multi-threaded dynamic scheduling across features and candidate split thresholds (`#pragma omp parallel for`).
3. **Level 3 (Hardware Acceleration / CUDA)**: Fine-grained GPU kernel parallelizing sample evaluation and Gini calculation for each candidate threshold.

---

## 2. Directory Structure

```text
├── Makefile                # Build system (Serial, MPI+OpenMP, CUDA)
├── preprocess.py           # Preprocessing script (or synthetic data generator)
├── verify.py               # Correctness validation against Scikit-learn
├── run_scaling_256.slurm   # SLURM script for 1 to 256 CPU cores scaling
├── run_gpu.slurm           # SLURM script for single GPU evaluation
├── src/
│   ├── common.h            # Data structures, fast Gini, and CSV parser
│   ├── tree_serial.cpp     # Serial baseline decision tree
│   ├── tree_mpi_omp.cpp    # MPI + OpenMP hybrid distributed decision tree
│   └── tree_cuda.cu        # CUDA kernel for single GPU execution
└── README.md               # User guide
```

---

## 3. Quick Start (Local Development)

### Step 1: Generate or Prepare Dataset
If raw Kaggle files (`orders.csv`, `order_products__train.csv`, `products.csv`) are present:
```bash
python3 preprocess.py
```
If raw files are not downloaded yet, running `python3 preprocess.py` will automatically generate 200,000 synthetic rows matching the exact Instacart schema for immediate local compilation and debugging.

### Step 2: Build All Targets
```bash
make all
```

### Step 3: Run Baseline and Parallel Implementations
* **Serial Version**:
  ```bash
  ./tree_serial instacart_clean.csv 6
  ```
* **MPI + OpenMP Version (e.g. 4 MPI ranks x 2 OpenMP threads = 8 cores)**:
  ```bash
  export OMP_NUM_THREADS=2
  mpirun -np 4 ./tree_mpi_omp instacart_clean.csv 6
  ```
* **CUDA Version (if NVIDIA GPU available)**:
  ```bash
  ./tree_cuda instacart_clean.csv
  ```

### Step 4: Validate Correctness
Verify the tree's accuracy and root split against Python's `scikit-learn`:
```bash
python3 verify.py instacart_clean.csv
```

---

## 4. Running on Atlas / Vanda Cluster

### 256-Core Strong & Weak Scaling
Submit the 256-core batch script:
```bash
sbatch run_scaling_256.slurm
```
This will automatically evaluate cores from 1, 2, 4, 8, 16, 32, 64, 128 to **256 cores**, measuring:
* Total Runtime ($T_{total}$)
* Pure Computation Time ($T_{comp}$)
* MPI Communication Overhead ($T_{comm}$)
* Speedup ($S = T_{serial} / T_{parallel}$)
* Parallel Efficiency ($E = S / p$)

Output will be recorded in `scaling_results.csv`.

### GPU Test
Submit the GPU job:
```bash
sbatch run_gpu.slurm
```

---

## 5. Group Members
* [Member 1 Name - Student ID]
* [Member 2 Name - Student ID]
* [Member 3 Name - Student ID]
