#!/bin/bash
#SBATCH --job-name=dt_scaling_256
#SBATCH --output=scaling_256_%j.log
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=normal

# ==========================================================
# Atlas / Vanda 256 核心 Strong & Weak Scaling 測試腳本
# 總核心數 = ntasks (MPI) * cpus-per-task (OpenMP) = 64 * 4 = 256
# ==========================================================

module purge
module load openmpi/4.1.1
module load gcc/9.3.0

echo "=== 開始編譯專案 ==="
make clean && make all

DATA_FILE="instacart_clean.csv"
OUT_CSV="scaling_results.csv"

echo "cores,mpi_ranks,omp_threads,total_time,comm_time,speedup" > $OUT_CSV

# 1. 跑序列版 (基準時間 T_serial)
echo "=== 正在執行 Serial 基準版 ==="
SERIAL_LOG=$(./tree_serial $DATA_FILE 6)
T_SERIAL=$(echo "$SERIAL_LOG" | grep "訓練時間" | awk '{print $3}')
echo "1,1,1,$T_SERIAL,0.0,1.0" >> $OUT_CSV
echo "Serial 基準時間: $T_SERIAL 秒"

# 2. 遍歷核心數進行 Strong Scaling 測試 (從 2 到 256 核)
# 組合格式: (MPI_RANKS, OMP_THREADS)
CORE_CONFIGS=(
    "2 1"    # 2 核
    "4 1"    # 4 核
    "8 1"    # 8 核
    "8 2"    # 16 核
    "16 2"   # 32 核
    "16 4"   # 64 核
    "32 4"   # 128 核
    "64 4"   # 256 核 (全速衝刺評分標準要求！)
)

for CFG in "${CORE_CONFIGS[@]}"; do
    RANKS=$(echo $CFG | awk '{print $1}')
    THREADS=$(echo $CFG | awk '{print $2}')
    TOTAL_CORES=$((RANKS * THREADS))

    echo ">>> 正在測試總核心數: $TOTAL_CORES (MPI: $RANKS, OpenMP: $THREADS) <<<"
    export OMP_NUM_THREADS=$THREADS

    RUN_LOG=$(mpirun -np $RANKS ./tree_mpi_omp $DATA_FILE 6)
    
    TOTAL_TIME=$(echo "$RUN_LOG" | grep "總訓練時間" | awk '{print $3}')
    COMM_TIME=$(echo "$RUN_LOG" | grep "MPI 通訊時間" | awk '{print $3}')
    
    # 計算 Speedup = T_serial / T_parallel
    SPEEDUP=$(python3 -c "print(f'{$T_SERIAL / float(\"$TOTAL_TIME\"):.2f}')")

    echo "$TOTAL_CORES,$RANKS,$THREADS,$TOTAL_TIME,$COMM_TIME,$SPEEDUP" >> $OUT_CSV
    echo "    完成! 總時間: $TOTAL_TIME s, 通訊時間: $COMM_TIME s, 加速比: $SPEEDUP x"
done

echo "=== 256 核心 Scaling 測試結束！數據已儲存至 $OUT_CSV ==="
