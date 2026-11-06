#!/usr/bin/env python3
"""Compare the whole tree written by `dt --dump-tree` with scikit-learn's tree.

    build/dt_serial --csv data.csv --dump-tree tree.txt --reps 1 --warmup 0
    python3 scripts/compare_sklearn.py data.csv tree.txt [--depth 6] [--min-split 20]

CSV layout: first column = label, remaining columns = features (header row).
`--min-split S` in dt means "a node with <= S samples is a leaf", which is
scikit-learn's min_samples_split = S + 1.
A leaf whose two classes are exactly tied is accepted either way: dt predicts 1,
scikit-learn predicts 0.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def parse_dump(path):
    """Preorder dump -> nested dicts."""
    lines = [l.split() for l in open(path) if l.strip()]
    pos = 0

    def rec():
        nonlocal pos
        t = lines[pos]
        pos += 1
        if t[0] == "L":
            return {"leaf": True, "pred": int(t[1]), "tot": int(t[2]), "pos": int(t[3])}
        node = {"leaf": False, "feat": int(t[1]), "thr": float(t[3]), "tot": int(t[4]), "pos": int(t[5])}
        node["left"] = rec()
        node["right"] = rec()
        return node

    return rec()


def count(node):
    return 1 if node["leaf"] else 1 + count(node["left"]) + count(node["right"])


def predict(node, X, idx, out):
    if node["leaf"]:
        out[idx] = node["pred"]
        return
    go_left = X[idx, node["feat"]] <= node["thr"]
    predict(node["left"], X, idx[go_left], out)
    predict(node["right"], X, idx[~go_left], out)


def compare(ours, sk, nid, path, issues, stats):
    stats["nodes"] += 1
    sk_leaf = sk.children_left[nid] == -1
    if ours["leaf"] != sk_leaf:
        issues.append(f"{path}: ours={'leaf' if ours['leaf'] else 'split'}  sklearn={'leaf' if sk_leaf else 'split'}")
        return
    if ours["leaf"]:
        v = sk.value[nid][0]
        sk_pred = int(np.argmax(v))
        tie = ours["pos"] * 2 == ours["tot"]
        if ours["pred"] != sk_pred:
            if tie:
                stats["ties"] += 1
            else:
                issues.append(f"{path}: leaf prediction ours={ours['pred']} sklearn={sk_pred}")
        return
    if ours["feat"] != sk.feature[nid] or abs(ours["thr"] - sk.threshold[nid]) > 1e-6:
        issues.append(f"{path}: ours=(f{ours['feat']}<={ours['thr']:.6g})  "
                      f"sklearn=(f{sk.feature[nid]}<={sk.threshold[nid]:.6g})")
        return  # subtrees are no longer comparable
    compare(ours["left"], sk, sk.children_left[nid], path + "L", issues, stats)
    compare(ours["right"], sk, sk.children_right[nid], path + "R", issues, stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("dump")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--min-split", type=int, default=20)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    y = df.iloc[:, 0].to_numpy().astype(int)
    X = df.iloc[:, 1:].to_numpy().astype(np.float64)

    clf = DecisionTreeClassifier(criterion="gini", max_depth=a.depth,
                                 min_samples_split=a.min_split + 1, random_state=0).fit(X, y)
    ours = parse_dump(a.dump)

    pred_ours = np.zeros(len(y), dtype=int)
    predict(ours, X, np.arange(len(y)), pred_ours)
    pred_sk = clf.predict(X)

    issues, stats = [], {"nodes": 0, "ties": 0}
    compare(ours, clf.tree_, 0, "root", issues, stats)

    print(f"nodes:  ours={count(ours)}   sklearn={clf.tree_.node_count}")
    print(f"training accuracy:  ours={np.mean(pred_ours == y) * 100:.4f}%   sklearn={np.mean(pred_sk == y) * 100:.4f}%")
    print(f"identical predictions on {np.mean(pred_ours == pred_sk) * 100:.4f}% of rows")
    print(f"nodes compared: {stats['nodes']}   exact-tie leaves accepted: {stats['ties']}")
    if issues:
        print(f"STRUCTURE DIFFERS ({len(issues)} issue(s)); first {min(5, len(issues))}:")
        for s in issues[:5]:
            print("  ", s)
        sys.exit(1)
    print("STRUCTURE MATCHES: same splits and same leaf predictions as scikit-learn")


if __name__ == "__main__":
    main()
