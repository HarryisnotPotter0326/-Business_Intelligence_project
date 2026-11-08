"""RQ2：學習曲線（TF-IDF 與凍結 BERT）——訓練資料只有 1k／5k／10k／30k 筆時，各方法表現如何？

只用抽出的訓練子集訓練（不合併驗證集），C 由驗證集挑選，回報測試 AUC。
凍結 BERT 直接讀 cache/ 裡 rq1_frozen.py 存的向量。微調的學習曲線由 rq1_finetune.py --train_size 負責。
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CACHE, load_olist, merge_save, subsample_idx
from monitor import LOG_PATH, Bar, log, log_text

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 30464])
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
ap.add_argument("--out", default="rq2_learning_curve.csv")
args = ap.parse_args()
SIZES, SEEDS = args.sizes, args.seeds
# 各模型在 RQ1 表現最好的聚合方式
FROZEN = {
    "bertimbau-base": ["cls", "mean", "std"],
    "bertimbau-large": ["cls", "mean", "max"],
    "mbert": ["cls", "mean", "std"],
}

train, val, test = load_olist()


def best_c(make, Xtr, ytr, Xva, Xte, grid):
    """用驗證集挑 C，回傳 (C, 驗證 AUC, 測試 AUC)。"""
    best = None
    for C in grid:
        clf = make(C).fit(Xtr, ytr)
        va = roc_auc_score(val.label, clf.predict_proba(Xva)[:, 1])
        if best is None or va > best[1]:
            best = (C, va, roc_auc_score(test.label, clf.predict_proba(Xte)[:, 1]))
    return best


frozen = {}
for name, parts in FROZEN.items():
    try:
        frozen[name] = {s: np.hstack([np.load(CACHE / f"frozen_{name}_{s}.npz")[p] for p in parts])
                        for s in ["train", "val", "test"]}
    except FileNotFoundError:
        log(f"找不到 {name} 的向量快取，略過（請先跑 rq1_frozen.py {name}）")

rows = []
jobs = [(n, s) for n in SIZES for s in SEEDS]
for n, seed in Bar(jobs, desc="學習曲線", unit="組"):
    idx = subsample_idx(train, n, seed)
    tr = train.loc[idx]
    if n >= len(train) and seed != SEEDS[0]:
        continue  # 全部資料時抽樣結果都一樣，只跑一次

    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=min(6, max(1, n // 1000)), max_features=10000)
    Xtr = vec.fit_transform(tr.text_tokens)
    C, va, te = best_c(lambda C: LogisticRegression(C=C, max_iter=3000), Xtr, tr.label,
                       vec.transform(val.text_tokens), vec.transform(test.text_tokens), [0.3, 1, 3, 10])
    rows.append(dict(method="TF-IDF + LR", train_size=len(tr), seed=seed, C=C, val_auc=round(va, 4), test_auc=round(te, 4)))
    log(f"n={len(tr):6d} seed={seed}  {'TF-IDF + LR':24s} C={C:<6} 測試 AUC {te:.4f}")

    for name, X in frozen.items():
        C, va, te = best_c(lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)),
                           X["train"][idx], tr.label, X["val"], X["test"], [1e-4, 1e-3, 3e-3, 1e-2, 0.1])
        method = f"{name} 凍結 + LR"
        rows.append(dict(method=method, train_size=len(tr), seed=seed, C=C, val_auc=round(va, 4), test_auc=round(te, 4)))
        log(f"n={len(tr):6d} seed={seed}  {method:24s} C={C:<6} 測試 AUC {te:.4f}")
    out, path = merge_save(rows, args.out, ["method", "train_size", "seed"])

log(f"結果已存到 {path}")
log_text(out.groupby(["train_size", "method"]).test_auc.mean().unstack().round(4).to_string())
log(f"記錄檔 {LOG_PATH}")
