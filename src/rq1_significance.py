"""RQ1 加深：方法之間的差距是真的，還是隨機誤差？

- 每個方法在同一個測試集上的預測，做 2,000 次「成對 bootstrap」（每次重抽測試資料，所有方法用同一組樣本）
- 回報每個方法 AUC 的 95% 信賴區間，以及兩兩差距的信賴區間與 p 值
- 微調方法有多個種子：每次 bootstrap 取各種子 AUC 的平均，所以同時考慮了「測試資料」與「隨機種子」兩種變異
- 凍結 BERT 的聚合方式用「驗證 AUC 最高」來選（不偷看測試集）
"""
import glob

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CACHE, RESULTS, load_olist
from monitor import LOG_PATH, Bar, log, log_text
from rq1_frozen import STRATEGIES

B = 2000
PRED = RESULTS / "preds"
PAPER_FT = {"bertimbau-base": (2.5e-5, 0.1, 1), "bertimbau-large": (2.5e-5, 0.1, 1), "mbert": (2.5e-5, 0.0, 2)}

train, val, test = load_olist()
trval = pd.concat([train, val])
y = test.label.values
methods = {}  # 名稱 -> 預測陣列的 list（每個種子一個）

# TF-IDF（論文設定）
vec = TfidfVectorizer(ngram_range=(1, 3), min_df=6, max_features=10000)
clf = LogisticRegression(C=1.0, max_iter=3000).fit(vec.fit_transform(trval.text_tokens), trval.label)
methods["TF-IDF + LR"] = [clf.predict_proba(vec.transform(test.text_tokens))[:, 1]]
np.save(PRED / "tfidf_paper.npy", methods["TF-IDF + LR"][0])
log("TF-IDF 預測完成")

# 凍結 BERT：用驗證 AUC 選聚合方式
fz = pd.read_csv(RESULTS / "rq1_frozen.csv")
for m, g in fz.groupby("model"):
    r = g.loc[g.val_auc.idxmax()]
    emb = {s: np.load(CACHE / f"frozen_{m}_{s}.npz") for s in ["train", "val", "test"]}
    X = {s: np.hstack([emb[s][p] for p in STRATEGIES[r.strategy]]) for s in emb}
    c = make_pipeline(StandardScaler(), LogisticRegression(C=r.C, max_iter=3000))
    c.fit(np.vstack([X["train"], X["val"]]), np.concatenate([train.label, val.label]))
    p = c.predict_proba(X["test"])[:, 1]
    methods[f"{m} 凍結（{r.strategy}）"] = [p]
    np.save(PRED / f"frozen_{m}_valbest.npy", p)
    log(f"{m} 凍結：驗證最佳聚合 {r.strategy}，C={r.C}，測試 AUC {roc_auc_score(y, p):.4f}")

# 微調：收集論文設定、全部訓練資料的各種子預測（新舊兩種檔名都找）
for m, (lr, do, E) in PAPER_FT.items():
    files = glob.glob(str(PRED / f"finetune_{m}_n{len(train)}_lr{lr:g}_do{do:g}_E{E}_s*_ep{E}.npy"))
    seeds_new = {f.split("_s")[-1].split("_")[0] for f in files}
    for f in glob.glob(str(PRED / f"finetune_{m}_n{len(train)}_s*_ep{E}.npy")):  # 舊檔名
        if f.split("_s")[-1].split("_")[0] not in seeds_new:
            files.append(f)
    if files:
        methods[f"{m} 微調"] = [np.load(f) for f in sorted(files)]
        log(f"{m} 微調：找到 {len(files)} 個種子的預測")

names = list(methods)
rng = np.random.default_rng(42)
boot = {n: np.zeros(B) for n in names}
for b in Bar(range(B), desc="bootstrap", unit="次"):
    idx = rng.integers(0, len(y), len(y))
    for n in names:
        boot[n][b] = np.mean([roc_auc_score(y[idx], p[idx]) for p in methods[n]])

rows = []
for n in names:
    aucs = [roc_auc_score(y, p) for p in methods[n]]
    rows.append(dict(method=n, seeds=len(aucs), test_auc=round(np.mean(aucs), 4),
                     seed_std=round(np.std(aucs, ddof=1), 4) if len(aucs) > 1 else None,
                     ci_low=round(np.percentile(boot[n], 2.5), 4), ci_high=round(np.percentile(boot[n], 97.5), 4)))
single = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
single.to_csv(RESULTS / "rq1_significance.csv", index=False)

pairs = []
for i, a in enumerate(names):
    for b_ in names[i + 1:]:
        d = boot[a] - boot[b_]
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        pairs.append(dict(A=a, B=b_, diff_pct=round(100 * d.mean(), 3), ci_low_pct=round(100 * np.percentile(d, 2.5), 3),
                          ci_high_pct=round(100 * np.percentile(d, 97.5), 3), p_value=round(min(p, 1.0), 4),
                          significant=bool(p < 0.05)))
pairs = pd.DataFrame(pairs)
pairs.to_csv(RESULTS / "rq1_significance_pairs.csv", index=False)

log_text("\n各方法測試 AUC 與 95% 信賴區間：\n" + single.to_string(index=False))
log_text("\n兩兩比較（A − B，百分點）：\n" + pairs.to_string(index=False))
log(f"結果已存到 results/rq1_significance*.csv；記錄檔 {LOG_PATH}")
