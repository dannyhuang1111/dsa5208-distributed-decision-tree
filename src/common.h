#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <algorithm>

// 高精度計時器 (回傳秒數)
inline double get_time_sec() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

// 樣本資料集結構
struct Dataset {
    int n_samples;
    int n_features;
    std::vector<float> X;   // 平鋪一維陣列 (Row-major: n_samples * n_features)，快取極其友善
    std::vector<int> y;     // 標籤 (0 或 1)

    // 存取特徵的輔助函數
    inline float get(int sample_idx, int feature_idx) const {
        return X[sample_idx * n_features + feature_idx];
    }
};

// 最佳切分點資訊
struct SplitCandidate {
    int feature_idx;
    float threshold;
    double gini_gain;
    int left_pos;   // 左子節點正例數
    int left_total; // 左子節點總數
    int right_pos;  // 右子節點正例數
    int right_total;// 右子節點總數

    SplitCandidate() : feature_idx(-1), threshold(0.0f), gini_gain(-1.0),
                       left_pos(0), left_total(0), right_pos(0), right_total(0) {}
};

// 決策樹節點結構
struct TreeNode {
    bool is_leaf;
    int prediction;     // 葉節點預測類別 (0 或 1)
    int feature_idx;    // 分裂特徵
    float threshold;    // 分裂門檻
    TreeNode* left;
    TreeNode* right;

    TreeNode() : is_leaf(false), prediction(-1), feature_idx(-1), threshold(0.0f),
                 left(nullptr), right(nullptr) {}
};

// 快速讀取 CSV (第一欄為 Label, 後續為 Features)
inline bool load_csv(const std::string& filename, Dataset& data, int max_rows = -1) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[-] 無法開啟檔案: " << filename << std::endl;
        return false;
    }

    std::string line;
    // 讀取標題列
    if (!std::getline(file, line)) return false;

    // 計算欄位數
    std::stringstream ss_first(line);
    std::string token;
    int total_cols = 0;
    while (std::getline(ss_first, token, ',')) total_cols++;
    data.n_features = total_cols - 1;

    data.X.clear();
    data.y.clear();

    int row_count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;

        // 第 0 欄: label
        if (std::getline(ss, val, ',')) {
            data.y.push_back(std::stoi(val));
        }

        // 後續欄位: features
        for (int f = 0; f < data.n_features; f++) {
            if (std::getline(ss, val, ',')) {
                data.X.push_back(std::stof(val));
            }
        }

        row_count++;
        if (max_rows > 0 && row_count >= max_rows) break;
    }

    data.n_samples = row_count;
    std::cout << "[+] 成功載入資料: " << data.n_samples << " 筆樣本, " 
              << data.n_features << " 個特徵。" << std::endl;
    return true;
}

// Gini Impurity 計算: 1 - (p0^2 + p1^2)
inline double compute_gini(int pos, int total) {
    if (total == 0) return 0.0;
    double p1 = (double)pos / total;
    double p0 = 1.0 - p1;
    return 1.0 - (p0 * p0 + p1 * p1);
}

#endif // COMMON_H
