#include "common.h"
#include <iomanip>

class DecisionTreeSerial {
public:
    int max_depth;
    int min_samples_split;
    TreeNode* root;

    DecisionTreeSerial(int depth = 6, int min_split = 10)
        : max_depth(depth), min_samples_split(min_split), root(nullptr) {}

    ~DecisionTreeSerial() {
        free_tree(root);
    }

    void fit(const Dataset& data) {
        std::vector<int> sample_indices(data.n_samples);
        for (int i = 0; i < data.n_samples; i++) sample_indices[i] = i;
        root = build_tree(data, sample_indices, 0);
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

    double evaluate(const Dataset& data) const {
        int correct = 0;
        for (int i = 0; i < data.n_samples; i++) {
            if (predict_one(data, i, root) == data.y[i]) {
                correct++;
            }
        }
        return (double)correct / data.n_samples;
    }

private:
    void free_tree(TreeNode* node) {
        if (!node) return;
        free_tree(node->left);
        free_tree(node->right);
        delete node;
    }

    TreeNode* build_tree(const Dataset& data, const std::vector<int>& indices, int depth) {
        TreeNode* node = new TreeNode();
        int total = indices.size();

        // 統計當前節點的正例數量
        int pos = 0;
        for (int idx : indices) {
            if (data.y[idx] == 1) pos++;
        }

        // 判定終止條件 (達最大深度、樣本過少、或是純度已達 100%)
        if (depth >= max_depth || total <= min_samples_split || pos == 0 || pos == total) {
            node->is_leaf = true;
            node->prediction = (pos >= total - pos) ? 1 : 0;
            return node;
        }

        // 尋找當前節點的最佳分裂特徵與門檻 (Best Split)
        SplitCandidate best_split = find_best_split(data, indices, pos, total);

        if (best_split.gini_gain <= 0.0) {
            node->is_leaf = true;
            node->prediction = (pos >= total - pos) ? 1 : 0;
            return node;
        }

        node->feature_idx = best_split.feature_idx;
        node->threshold = best_split.threshold;

        // 分裂樣本子集
        std::vector<int> left_indices, right_indices;
        for (int idx : indices) {
            if (data.get(idx, node->feature_idx) <= node->threshold) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }

        // 遞迴建樹
        node->left = build_tree(data, left_indices, depth + 1);
        node->right = build_tree(data, right_indices, depth + 1);
        return node;
    }

    SplitCandidate find_best_split(const Dataset& data, const std::vector<int>& indices, int pos, int total) {
        SplitCandidate best_split;
        double current_gini = compute_gini(pos, total);

        // 遍歷所有特徵 (Level 3 平行化的切入點)
        for (int f = 0; f < data.n_features; f++) {
            // 收集該特徵的所有取值
            std::vector<std::pair<float, int>> feat_vals;
            feat_vals.reserve(total);
            for (int idx : indices) {
                feat_vals.push_back({data.get(idx, f), data.y[idx]});
            }

            std::sort(feat_vals.begin(), feat_vals.end(), 
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            int left_pos = 0;
            int left_total = 0;

            // 掃描門檻 candidate
            for (int i = 0; i < total - 1; i++) {
                if (feat_vals[i].second == 1) left_pos++;
                left_total++;

                // 只有特徵值變動時才評估切分點
                if (feat_vals[i].first != feat_vals[i + 1].first) {
                    int right_pos = pos - left_pos;
                    int right_total = total - left_total;

                    double g_left = compute_gini(left_pos, left_total);
                    double g_right = compute_gini(right_pos, right_total);
                    double weighted_gini = ((double)left_total / total) * g_left + 
                                           ((double)right_total / total) * g_right;
                    double gain = current_gini - weighted_gini;

                    if (gain > best_split.gini_gain) {
                        best_split.gini_gain = gain;
                        best_split.feature_idx = f;
                        best_split.threshold = (feat_vals[i].first + feat_vals[i + 1].first) / 2.0f;
                        best_split.left_pos = left_pos;
                        best_split.left_total = left_total;
                        best_split.right_pos = right_pos;
                        best_split.right_total = right_total;
                    }
                }
            }
        }
        return best_split;
    }
};

int main(int argc, char** argv) {
    std::string data_file = "instacart_clean.csv";
    if (argc > 1) data_file = argv[1];

    int max_depth = 6;
    if (argc > 2) max_depth = std::stoi(argv[2]);

    Dataset data;
    if (!load_csv(data_file, data)) {
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << " [Serial] Instacart Decision Tree 訓練" << std::endl;
    std::cout << " Max Depth: " << max_depth << std::endl;
    std::cout << "========================================" << std::endl;

    double t_start = get_time_sec();
    DecisionTreeSerial tree(max_depth, 20);
    tree.fit(data);
    double t_end = get_time_sec();

    double train_time = t_end - t_start;
    double accuracy = tree.evaluate(data);

    std::cout << "\n[+] 訓練完成！" << std::endl;
    std::cout << "    訓練時間 (Training Time): " << std::fixed << std::setprecision(4) << train_time << " 秒" << std::endl;
    std::cout << "    訓練準確率 (Accuracy): " << std::fixed << std::setprecision(2) << (accuracy * 100.0) << " %" << std::endl;
    std::cout << "    根節點特徵: " << tree.root->feature_idx 
              << " (Threshold: " << tree.root->threshold << ")" << std::endl;

    return 0;
}
