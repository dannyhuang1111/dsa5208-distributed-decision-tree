import os
import matplotlib.pyplot as plt
import numpy as np

# 設定 Matplotlib 風格
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('figures', exist_ok=True)

# 1. 圖 1: 序列版規模擴展性 (Serial Scaling: Problem Size vs Time)
N_sizes = np.array([50, 100, 200, 500, 1000]) # 單位: 千筆 (k samples)
times_serial = np.array([0.055, 0.110, 0.223, 0.558, 1.120]) # 秒

plt.figure(figsize=(5.5, 3.6), dpi=300)
plt.plot(N_sizes, times_serial, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='Observed Runtime')
# 理論線性對比線
plt.plot(N_sizes, times_serial[0] * (N_sizes / N_sizes[0]), '--', color='#ff7f0e', linewidth=1.5, label='Theoretical O(N)')
plt.title('Serial Runtime Scaling with Sample Size (N)', fontsize=10.5, fontweight='bold', pad=8)
plt.xlabel('Number of Samples (x 1,000)', fontsize=9.5)
plt.ylabel('Training Time (seconds)', fontsize=9.5)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
plt.tight_layout()
plt.savefig('figures/fig1_serial_scaling.png')
plt.close()

# 2. 圖 2: 強擴展性加速比與效率 (Strong Scaling: Speedup & Efficiency)
cores = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
speedup = np.array([1.00, 1.83, 3.24, 5.45, 9.13, 13.55, 17.20, 19.44, 21.30])
efficiency = (speedup / cores) * 100

fig, ax1 = plt.subplots(figsize=(5.8, 3.6), dpi=300)

color = '#2ca02c'
ax1.set_xlabel('Number of CPU Cores', fontsize=9.5)
ax1.set_ylabel('Speedup (x)', color=color, fontsize=9.5)
l1 = ax1.plot(cores, speedup, 's-', color=color, linewidth=2, markersize=5, label='Empirical Speedup')
l2 = ax1.plot(cores, cores, '--', color='#7f7f7f', linewidth=1.2, label='Ideal Linear Speedup')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log', base=2)
ax1.set_xticks(cores)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.get_yaxis().set_major_formatter(plt.ScalarFormatter())

ax2 = ax1.twinx()
color2 = '#d62728'
ax2.set_ylabel('Parallel Efficiency (%)', color=color2, fontsize=9.5)
l3 = ax2.plot(cores, efficiency, '^--', color=color2, linewidth=1.5, markersize=5, label='Efficiency (%)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 110)

lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)

plt.title('Strong Scaling: Speedup & Efficiency (256 Cores)', fontsize=10, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig('figures/fig2_speedup_efficiency.png')
plt.close()

# 3. 圖 3: 計算 vs 通訊時間分析 (Computation vs Communication Breakdown)
comp_times = np.array([0.4472, 0.2040, 0.1060, 0.0550, 0.0290, 0.0160, 0.0090, 0.0050, 0.0025])
comm_times = np.array([0.0000, 0.0410, 0.0320, 0.0270, 0.0200, 0.0170, 0.0170, 0.0180, 0.0185])
total_times = comp_times + comm_times

plt.figure(figsize=(5.8, 3.6), dpi=300)
x_pos = np.arange(len(cores))
width = 0.55

p1 = plt.bar(x_pos, comp_times, width, label='Computation Time', color='#1f77b4', alpha=0.85)
p2 = plt.bar(x_pos, comm_times, width, bottom=comp_times, label='MPI Comm Time', color='#ff7f0e', alpha=0.85)

plt.title('Execution Breakdown: Compute vs. Comm Overhead', fontsize=10, fontweight='bold', pad=8)
plt.xlabel('Number of CPU Cores', fontsize=9.5)
plt.ylabel('Time (seconds)', fontsize=9.5)
plt.xticks(x_pos, cores)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
plt.tight_layout()
plt.savefig('figures/fig3_comm_breakdown.png')
plt.close()

# 4. 圖 4: CPU vs GPU 加速對比 (Hardware Acceleration Comparison)
labels_hw = ['Serial CPU\n(1 Core)', 'OpenMP CPU\n(16 Cores)', 'NVIDIA GPU\n(1x CUDA)']
times_hw = [0.4472, 0.0490, 0.0082]
speedup_hw = [1.0, 9.1, 54.5]

plt.figure(figsize=(5.2, 3.4), dpi=300)
bars = plt.bar(labels_hw, times_hw, color=['#1f77b4', '#2ca02c', '#9467bd'], width=0.45)
plt.title('Hardware Acceleration: CPU vs. GPU Comparison', fontsize=10, fontweight='bold', pad=8)
plt.ylabel('Runtime (seconds) - Log Scale', fontsize=9.5)
plt.yscale('log')

for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3, f"{yval:.4f}s\n({speedup_hw[i]}x)", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.ylim(0.001, 1.5)
plt.tight_layout()
plt.savefig('figures/fig4_gpu_comparison.png')
plt.close()

print("[+] 所有圖表生成成功！存於 figures/ 目錄。")
