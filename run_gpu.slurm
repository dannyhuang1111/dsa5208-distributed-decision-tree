#!/bin/bash
#SBATCH --job-name=dt_gpu_test
#SBATCH --output=gpu_test_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

# ==========================================================
# Atlas / Vanda 單張 GPU 測試腳本 (滿足評分標準第 4 點)
# ==========================================================

module purge
module load cuda/11.7
module load gcc/9.3.0

echo "=== 編譯 CUDA 版本 ==="
nvcc -O3 -std=c++17 -Isrc src/tree_cuda.cu -o tree_cuda

DATA_FILE="instacart_clean.csv"

echo "=== 正在執行 CUDA GPU 測試 ==="
./tree_cuda $DATA_FILE
