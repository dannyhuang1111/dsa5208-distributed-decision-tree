import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys

def verify(filename="instacart_clean.csv"):
    print(f"[*] 正在讀取 {filename} 進行 Scikit-Learn 基線驗證...")
    df = pd.read_csv(filename)
    
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    print(f"[+] 資料筆數: {len(X)}, 特徵數: {X.shape[1]}")
    
    clf = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=20, random_state=42)
    clf.fit(X, y)
    
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    
    print("\n========================================")
    print(" [Python Scikit-Learn] 基準驗證結果")
    print(f" 訓練準確率 (Accuracy): {acc * 100:.2f} %")
    print(f" 根節點特徵索引 (Feature Index): {clf.tree_.feature[0]}")
    print(f" 根節點門檻值 (Threshold): {clf.tree_.threshold[0]:.4f}")
    print("========================================")
    print("[✓] 請將上述結果與 C++ tree_serial / tree_mpi_omp 的輸出進行對比。")
    print("[✓] 兩者數值若相近，即可證明 C++ 分散式演算法的數學正確性！")

if __name__ == '__main__':
    filename = sys.argv[1] if len(sys.argv) > 1 else "instacart_clean.csv"
    verify(filename)
