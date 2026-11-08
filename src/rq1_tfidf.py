"""RQ1：TF-IDF + 邏輯斯迴歸基準，並找出與論文 96.5% 的差距來源。"""
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import RESULTS, load_olist
from monitor import LOG_PATH, Bar, log, log_text

train, val, test = load_olist()
trval = pd.concat([train, val])
print(f"train={len(train)} val={len(val)} test={len(test)}  positive rate={trval.label.mean():.3f}")

rows = []
grid = []
for col in ["text_tokens", "review_text_processed", "review_text"]:
    for ngram in [(1, 1), (1, 2), (1, 3)]:
        for vocab in [1000, 5000, 10000, None]:
            grid.append(dict(col=col, ngram=ngram, vocab=vocab, min_df=6, C=1.0, fit_on="train+val"))
# LR 正則化強度、只用訓練集（不合併驗證集）
for C in [0.1, 0.3, 3.0, 10.0]:
    grid.append(dict(col="text_tokens", ngram=(1, 3), vocab=10000, min_df=6, C=C, fit_on="train+val"))
grid.append(dict(col="text_tokens", ngram=(1, 3), vocab=10000, min_df=6, C=1.0, fit_on="train"))
grid.append(dict(col="text_tokens", ngram=(1, 3), vocab=10000, min_df=1, C=1.0, fit_on="train+val"))

for g in Bar(grid, desc="TF-IDF 設定", unit="組"):
    fit_df = trval if g["fit_on"] == "train+val" else train
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=g["ngram"], min_df=g["min_df"], max_features=g["vocab"],
                          lowercase=True)
    Xtr = vec.fit_transform(fit_df[g["col"]])
    clf = LogisticRegression(C=g["C"], max_iter=3000).fit(Xtr, fit_df.label)
    auc_val = roc_auc_score(val.label, clf.predict_proba(vec.transform(val[g["col"]]))[:, 1])
    auc = roc_auc_score(test.label, clf.predict_proba(vec.transform(test[g["col"]]))[:, 1])
    rows.append({**g, "ngram": f"{g['ngram'][0]}-{g['ngram'][1]}", "n_features": Xtr.shape[1],
                 "val_auc": round(auc_val, 4), "test_auc": round(auc, 4), "sec": round(time.time() - t0, 1)})
    r = rows[-1]
    log(f"{r['col']:22s} {r['ngram']}-gram 詞彙 {str(r['vocab']):5s} C={r['C']:<4} {r['fit_on']:9s} "
        f"驗證 AUC {r['val_auc']:.4f}  測試 AUC {r['test_auc']:.4f}")

out = pd.DataFrame(rows)
out.to_csv(RESULTS / "rq1_tfidf.csv", index=False)
log_text(out.sort_values("test_auc", ascending=False).head(10).to_string())
log(f"結果已存到 {RESULTS / 'rq1_tfidf.csv'}；記錄檔 {LOG_PATH}")
