#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include "common.h"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// CUDA Kernel: 每個執行緒塊 (Thread Block) 負責一個特徵的切分搜尋，執行緒平行評估切分門檻
__global__ void evaluate_split_kernel(const float* d_X, const int* d_y, 
                                      int n_samples, int n_features, int feature_idx,
                                      float candidate_thresh, 
                                      int* d_left_pos, int* d_left_total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples) return;

    float val = d_X[tid * n_features + feature_idx];
    int label = d_y[tid];

    if (val <= candidate_thresh) {
        atomicAdd(d_left_total, 1);
        if (label == 1) {
            atomicAdd(d_left_pos, 1);
        }
    }
}

int main(int argc, char** argv) {
    std::string data_file = "instacart_clean.csv";
    if (argc > 1) data_file = argv[1];

    Dataset data;
    if (!load_csv(data_file, data)) {
        return 1;
    }

    // 檢查 GPU 設備
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "[-] 找不到支援 CUDA 的 GPU 設備！" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "========================================" << std::endl;
    std::cout << " [CUDA] 單張 GPU 加速特徵切分搜尋" << std::endl;
    std::cout << " GPU 設備: " << prop.name << std::endl;
    std::cout << " 總全域記憶體: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << " 樣本數: " << data.n_samples << ", 特徵數: " << data.n_features << std::endl;
    std::cout << "========================================" << std::endl;

    // 配置 GPU 記憶體
    float* d_X = nullptr;
    int* d_y = nullptr;
    int* d_left_pos = nullptr;
    int* d_left_total = nullptr;

    size_t x_bytes = data.n_samples * data.n_features * sizeof(float);
    size_t y_bytes = data.n_samples * sizeof(int);

    CHECK_CUDA(cudaMalloc(&d_X, x_bytes));
    CHECK_CUDA(cudaMalloc(&d_y, y_bytes));
    CHECK_CUDA(cudaMalloc(&d_left_pos, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_left_total, sizeof(int)));

    // 將資料拷貝至 GPU
    auto t_start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(d_X, data.X.data(), x_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, data.y.data(), y_bytes, cudaMemcpyHostToDevice));

    int global_pos = 0;
    for (int y : data.y) if (y == 1) global_pos++;
    int global_total = data.n_samples;
    double current_gini = compute_gini(global_pos, global_total);

    int threads_per_block = 256;
    int blocks = (data.n_samples + threads_per_block - 1) / threads_per_block;

    double best_gini_gain = -1.0;
    int best_feature = -1;
    float best_threshold = 0.0f;

    // 測試遍歷所有特徵尋找最佳分裂 (模擬決策樹關鍵熱點)
    for (int f = 0; f < data.n_features; f++) {
        // 取 20 個百分位門檻進行評估
        for (int step = 1; step <= 20; step++) {
            float thresh = (float)step * 2.0f; // 候選門檻

            CHECK_CUDA(cudaMemset(d_left_pos, 0, sizeof(int)));
            CHECK_CUDA(cudaMemset(d_left_total, 0, sizeof(int)));

            evaluate_split_kernel<<<blocks, threads_per_block>>>(
                d_X, d_y, data.n_samples, data.n_features, f, thresh, d_left_pos, d_left_total
            );
            CHECK_CUDA(cudaDeviceSynchronize());

            int h_left_pos = 0, h_left_total = 0;
            CHECK_CUDA(cudaMemcpy(&h_left_pos, d_left_pos, sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&h_left_total, d_left_total, sizeof(int), cudaMemcpyDeviceToHost));

            if (h_left_total > 0 && h_left_total < global_total) {
                int r_pos = global_pos - h_left_pos;
                int r_tot = global_total - h_left_total;
                double g_left = compute_gini(h_left_pos, h_left_total);
                double g_right = compute_gini(r_pos, r_tot);
                double gain = current_gini - (((double)h_left_total / global_total) * g_left + 
                                             ((double)r_tot / global_total) * g_right);

                if (gain > best_gini_gain) {
                    best_gini_gain = gain;
                    best_feature = f;
                    best_threshold = thresh;
                }
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_gpu_time = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n[+] GPU 運算完成！" << std::endl;
    std::cout << "    GPU 執行時間: " << std::fixed << std::setprecision(4) << total_gpu_time << " 秒" << std::endl;
    std::cout << "    最佳特徵: " << best_feature << " (門檻: " << best_threshold 
              << ", Gini 增益: " << best_gini_gain << ")" << std::endl;

    // 清理資源
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_left_pos);
    cudaFree(d_left_total);

    return 0;
}
