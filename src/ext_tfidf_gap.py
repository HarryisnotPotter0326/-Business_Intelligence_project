"""加深 1：為什麼我們的 TF-IDF（97.4%）比論文（96.5%）高 0.9 個百分點？

逐一改變可能的差異，看哪一個能把 AUC 拉回 96.5%：
  - 測試折的隨機性：10 折交叉驗證，看不同測試折的 AUC 差多少
  - 訓練資料：只用訓練集 vs 訓練＋驗證
  - 文字欄位：作者斷詞後（tokenized）／作者前處理（processed）／原始文字
  - 正則化強度 C、sublinear TF、class_weight、solver
輸出：results/ext_tfidf_gap.csv、ext_tfidf_folds.csv
用法：python src/ext_tfidf_gap.py
"""
import ast
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import DATA, RESULTS, load_olist
from monitor import LOG_PATH, Bar, log, log_text

PAPER = 0.965
BASE = dict(col="text_tokens", fit_on="train+val", ngram=(1, 3), min_df=6, vocab=10000, C=1.0,
            sublinear=False, class_weight=None, solver="lbfgs")


def run(train, val, test, cfg):
    tr = pd.concat([train, val]) if cfg["fit_on"] == "train+val" else train
    vec = TfidfVectorizer(ngram_range=cfg["ngram"], min_df=cfg["min_df"], max_features=cfg["vocab"],
                          sublinear_tf=cfg["sublinear"], lowercase=True)
    X = vec.fit_transform(tr[cfg["col"]])
    clf = LogisticRegression(C=cfg["C"], max_iter=3000, class_weight=cfg["class_weight"], solver=cfg["solver"])
    clf.fit(X, tr.label)
    return roc_auc_score(test.label, clf.predict_proba(vec.transform(test[cfg["col"]]))[:, 1])


def main():
    train, val, test = load_olist()
    variants = [("我們的設定（論文描述的設定）", {})]
    variants += [("只用訓練集（不合併驗證集）", dict(fit_on="train"))]
    variants += [(f"文字欄位：{c}", dict(col=c)) for c in ["review_text_processed", "review_text"]]
    variants += [(f"C = {c}", dict(C=c)) for c in [0.03, 0.1, 0.3, 3.0, 10.0]]
    variants += [("sublinear TF", dict(sublinear=True)), ("class_weight=balanced", dict(class_weight="balanced")),
                 ("solver=liblinear（舊版 sklearn 預設）", dict(solver="liblinear"))]
    variants += [("只用 1-gram", dict(ngram=(1, 1))), ("詞彙 1,000", dict(vocab=1000)), ("min_df = 2", dict(min_df=2))]
    variants += [("強正則化＋只用訓練集＋1-gram", dict(C=0.1, fit_on="train", ngram=(1, 1)))]
    rows = []
    for name, ch in Bar(variants, desc="TF-IDF 變化", unit="組"):
        cfg = {**BASE, **ch}
        t0 = time.time()
        auc = run(train, val, test, cfg)
        rows.append(dict(變化=name, test_auc=round(auc, 4), 與論文差距_百分點=round(100 * (auc - PAPER), 2),
                         秒=round(time.time() - t0, 1)))
        log(f"{name:<32} AUC {auc:.4f}（比論文 {100 * (auc - PAPER):+.2f}）")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ext_tfidf_gap.csv", index=False)

    # 10 折交叉驗證：每一折輪流當測試集（其他 9 折訓練），看測試折之間的差異
    d = pd.read_csv(DATA / "olist.csv")
    d = d[d.kfold_polarity > 0].copy()
    d["text_tokens"] = d.review_text_tokenized.fillna("[]").map(lambda s: " ".join(ast.literal_eval(s)))
    d["label"] = d.polarity.astype(int)
    folds = []
    for k in Bar(range(1, 11), desc="10 折交叉驗證", unit="折"):
        te, tr = d[d.kfold_polarity == k], d[d.kfold_polarity != k]
        auc = run(tr, tr.iloc[:0], te, {**BASE, "fit_on": "train"})
        folds.append(dict(測試折=k, test_auc=round(auc, 4)))
    f = pd.DataFrame(folds)
    f.to_csv(RESULTS / "ext_tfidf_folds.csv", index=False)
    log_text("\n各變化：\n" + out.to_string(index=False))
    log_text("\n10 折交叉驗證：\n" + f.to_string(index=False))
    log(f"10 折平均 {f.test_auc.mean():.4f} ± {f.test_auc.std():.4f}（最低 {f.test_auc.min():.4f}、最高 {f.test_auc.max():.4f}）；"
        f"論文 {PAPER}；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
