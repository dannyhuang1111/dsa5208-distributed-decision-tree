#include "common.h"
#include <mpi.h>
#include <omp.h>
#include <iomanip>

struct GlobalSplitPayload {
    double gini_gain;
    int feature_idx;
    float threshold;
    int left_pos;
    int left_total;
    int right_pos;
    int right_total;
};

class DistributedDecisionTree {
public:
    int max_depth;
    int min_samples_split;
    TreeNode* root;
    int rank, size;
    double total_comm_time; // 統計 MPI 通訊時間 (評分第3點必備)

    DistributedDecisionTree(int rank, int size, int depth = 6, int min_split = 20)
        : max_depth(depth), min_samples_split(min_split), root(nullptr),
          rank(rank), size(size), total_comm_time(0.0) {}

    ~DistributedDecisionTree() {
        free_tree(root);
    }

    void fit(const Dataset& local_data) {
        std::vector<int> local_indices(local_data.n_samples);
        for (int i = 0; i < local_data.n_samples; i++) local_indices[i] = i;
        root = build_distributed_tree(local_data, local_indices, 0);
    }

    int predict_one(const Dataset& data, int sample_idx, TreeNode* node) const {
        if (node->is_leaf) return node->prediction;
        float val = data.get(sample_idx, node->feature_idx);
        if (val <= node->threshold) {
            return predict_one(data, sample_idx, node->left);
        } else {
            return predict_one(data, sample_idx, node->right);
        }
    }

    double evaluate(const Dataset& local_data) const {
        int local_correct = 0;
        for (int i = 0; i < local_data.n_samples; i++) {
            if (predict_one(local_data, i, root) == local_data.y[i]) {
                local_correct++;
            }
        }
        int global_correct = 0;
        int global_total = 0;
        MPI_Reduce(&local_correct, &global_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_data.n_samples, &global_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) return (double)global_correct / global_total;
        return 0.0;
    }

private:
    void free_tree(TreeNode* node) {
        if (!node) return;
        free_tree(node->left);
        free_tree(node->right);
        delete node;
    }

    TreeNode* build_distributed_tree(const Dataset& data, const std::vector<int>& local_indices, int depth) {
        TreeNode* node = new TreeNode();

        // 1. 本地統計正例與總數
        int local_pos = 0;
        int local_total = local_indices.size();
        for (int idx : local_indices) {
            if (data.y[idx] == 1) local_pos++;
        }

        // 2. MPI 通訊聚合：獲取全域總樣本數與正例數
        int global_counts[2] = {local_pos, local_total};
        int aggregated_counts[2] = {0, 0};

        double t_comm_start = get_time_sec();
        MPI_Allreduce(global_counts, aggregated_counts, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        total_comm_time += (get_time_sec() - t_comm_start);

        int global_pos = aggregated_counts[0];
        int global_total = aggregated_counts[1];

        // 終止條件判定 (全域視角)
        if (depth >= max_depth || global_total <= min_samples_split || 
            global_pos == 0 || global_pos == global_total) {
            node->is_leaf = true;
            node->prediction = (global_pos >= global_total - global_pos) ? 1 : 0;
            return node;
        }

        // 3. 分散式搜尋最佳切分點 (OpenMP 內部特徵平行 + MPI 全域歸約)
        SplitCandidate best_split = find_best_split_distributed(data, local_indices, global_pos, global_total);

        if (best_split.gini_gain <= 0.0) {
            node->is_leaf = true;
            node->prediction = (global_pos >= global_total - global_pos) ? 1 : 0;
            return node;
        }

        node->feature_idx = best_split.feature_idx;
        node->threshold = best_split.threshold;

        // 4. 各 Rank 依據全域決定的 (feature, threshold) 同步切分本地樣本集
        std::vector<int> left_indices, right_indices;
        for (int idx : local_indices) {
            if (data.get(idx, node->feature_idx) <= node->threshold) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }

        // 5. 遞迴構建左右子樹 (保持所有 Rank 樹結構完全一致)
        node->left = build_distributed_tree(data, left_indices, depth + 1);
        node->right = build_distributed_tree(data, right_indices, depth + 1);
        return node;
    }

    SplitCandidate find_best_split_distributed(const Dataset& data, const std::vector<int>& local_indices, 
                                               int global_pos, int global_total) {
        SplitCandidate local_best;
        double current_gini = compute_gini(global_pos, global_total);

        // Level 3 平行：使用 OpenMP 平行遍歷不同特徵尋找切分點
        #pragma omp parallel
        {
            SplitCandidate thread_best;

            #pragma omp for schedule(dynamic)
            for (int f = 0; f < data.n_features; f++) {
                std::vector<std::pair<float, int>> feat_vals;
                feat_vals.reserve(local_indices.size());
                for (int idx : local_indices) {
                    feat_vals.push_back({data.get(idx, f), data.y[idx]});
                }

                std::sort(feat_vals.begin(), feat_vals.end(), 
                          [](const auto& a, const auto& b) { return a.first < b.first; });

                int l_pos = 0;
                int l_tot = 0;
                for (size_t i = 0; i + 1 < feat_vals.size(); i++) {
                    if (feat_vals[i].second == 1) l_pos++;
                    l_tot++;

                    if (feat_vals[i].first != feat_vals[i + 1].first) {
                        float cand_thresh = (feat_vals[i].first + feat_vals[i + 1].first) / 2.0f;
                        
                        // 初步估算本地增益 (快速篩選)
                        int r_pos = global_pos - l_pos;
                        int r_tot = global_total - l_tot;
                        double g_left = compute_gini(l_pos, l_tot);
                        double g_right = compute_gini(r_pos, r_tot);
                        double gain = current_gini - (((double)l_tot / global_total) * g_left + 
                                                     ((double)r_tot / global_total) * g_right);

                        if (gain > thread_best.gini_gain) {
                            thread_best.gini_gain = gain;
                            thread_best.feature_idx = f;
                            thread_best.threshold = cand_thresh;
                        }
                    }
                }
            }

            #pragma omp critical
            {
                if (thread_best.gini_gain > local_best.gini_gain) {
                    local_best = thread_best;
                }
            }
        }

        // MPI 通訊：將各 Rank 找到的 local_best 收集並比對出 global_best
        struct {
            double gain;
            int rank;
        } local_max = {local_best.gini_gain, rank}, global_max;

        double t_comm_start = get_time_sec();
        // 找出哪一個 rank 找到了全域最高的 Gini Gain
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        GlobalSplitPayload payload;
        if (rank == global_max.rank) {
            payload.gini_gain = local_best.gini_gain;
            payload.feature_idx = local_best.feature_idx;
            payload.threshold = local_best.threshold;
        }

        // 由該贏家 Rank 廣播其特徵索引與門檻給全體 256 個節點
        MPI_Bcast(&payload, sizeof(GlobalSplitPayload), MPI_BYTE, global_max.rank, MPI_COMM_WORLD);
        total_comm_time += (get_time_sec() - t_comm_start);

        local_best.gini_gain = payload.gini_gain;
        local_best.feature_idx = payload.feature_idx;
        local_best.threshold = payload.threshold;

        return local_best;
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string data_file = "instacart_clean.csv";
    if (argc > 1) data_file = argv[1];

    int max_depth = 6;
    if (argc > 2) max_depth = std::stoi(argv[2]);

    // 1. 各節點平均讀取資料 (Data Sharding)
    Dataset full_data;
    Dataset local_data;

    double t_start = get_time_sec();

    if (rank == 0) {
        if (!load_csv(data_file, full_data)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // 廣播特徵數量與總樣本數
    int meta[2] = {0, 0};
    if (rank == 0) {
        meta[0] = full_data.n_samples;
        meta[1] = full_data.n_features;
    }
    MPI_Bcast(meta, 2, MPI_INT, 0, MPI_COMM_WORLD);

    int total_samples = meta[0];
    int n_features = meta[1];

    // 計算每個 rank 負責的樣本數
    int samples_per_rank = total_samples / size;
    int local_n = (rank == size - 1) ? (total_samples - (size - 1) * samples_per_rank) : samples_per_rank;

    local_data.n_samples = local_n;
    local_data.n_features = n_features;
    local_data.X.resize(local_n * n_features);
    local_data.y.resize(local_n);

    // Rank 0 將數據分發給各節點
    std::vector<int> sendcounts(size), displs(size);
    std::vector<int> sendcounts_x(size), displs_x(size);

    for (int i = 0; i < size; i++) {
        sendcounts[i] = (i == size - 1) ? (total_samples - (size - 1) * samples_per_rank) : samples_per_rank;
        displs[i] = i * samples_per_rank;
        sendcounts_x[i] = sendcounts[i] * n_features;
        displs_x[i] = displs[i] * n_features;
    }

    MPI_Scatterv(rank == 0 ? full_data.y.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
                 local_data.y.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? full_data.X.data() : nullptr, sendcounts_x.data(), displs_x.data(), MPI_FLOAT,
                 local_data.X.data(), local_n * n_features, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << " [MPI + OpenMP] 分散式決策樹訓練" << std::endl;
        std::cout << " MPI Ranks: " << size << ", OpenMP Threads/Rank: " << omp_get_max_threads() << std::endl;
        std::cout << " 總核心數 (Total Cores): " << size * omp_get_max_threads() << std::endl;
        std::cout << " 總樣本數: " << total_samples << ", 特徵數: " << n_features << std::endl;
        std::cout << "========================================" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_train_start = get_time_sec();

    DistributedDecisionTree tree(rank, size, max_depth, 20);
    tree.fit(local_data);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_train_end = get_time_sec();

    double local_train_time = t_train_end - t_train_start;
    double max_train_time = 0.0;
    double max_comm_time = 0.0;

    MPI_Reduce(&local_train_time, &max_train_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tree.total_comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double accuracy = tree.evaluate(local_data);

    if (rank == 0) {
        double compute_time = max_train_time - max_comm_time;
        std::cout << "\n[+] 訓練完成！" << std::endl;
        std::cout << "    總訓練時間 (Total Train Time): " << std::fixed << std::setprecision(4) << max_train_time << " 秒" << std::endl;
        std::cout << "    純計算時間 (Compute Time):     " << std::fixed << std::setprecision(4) << compute_time << " 秒" << std::endl;
        std::cout << "    MPI 通訊時間 (Comm Time):       " << std::fixed << std::setprecision(4) << max_comm_time 
                  << " 秒 (" << (max_comm_time / max_train_time * 100.0) << " %)" << std::endl;
        std::cout << "    全域準確率 (Global Accuracy):   " << std::fixed << std::setprecision(2) << (accuracy * 100.0) << " %" << std::endl;
        std::cout << "    根節點分裂特徵: " << tree.root->feature_idx 
                  << " (門檻: " << tree.root->threshold << ")" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
