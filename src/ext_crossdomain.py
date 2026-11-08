"""加深 9：跨資料集——在 Olist 學到的模型，換到別的網站還準嗎？

5 個資料集（同一作者、同樣的正負面定義與 10 折切分）：
  olist（電商）、b2w（電商）、buscape（比價網站）、utlc_apps（App 評論）、utlc_movies（電影評論）
每個資料集：訓練用折 1–9 分層抽 30,000 筆，測試用折 10 分層抽 5,000 筆。
方法：
  - TF-IDF ＋ LR：5 × 5 全部組合（在 A 訓練、在 B 測試）
  - BERTimbau Base 微調（max_len 128，論文設定 1 epoch）：5 × 5
  - Qwen-1.5B 零樣本：不需要訓練，當作「與領域無關」的參考（每個測試集取 2,000 筆）
輸出：results/ext_crossdomain.csv；figures/ext_crossdomain.png
用法：python src/ext_crossdomain.py [--skip_bert] [--skip_llm]
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import bert_utils as BU  # noqa: E402
from common import DATA, FIGURES, RESULTS, merge_save  # noqa: E402
from monitor import LOG_PATH, Bar, log, log_text  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
DOMAINS = ["olist", "b2w", "buscape", "utlc_apps", "utlc_movies"]
N_TRAIN, N_TEST, N_LLM = 30000, 5000, 2000


def strat(df, n, seed=42):
    if len(df) <= n:
        return df
    return df.groupby("label").sample(frac=n / len(df), random_state=seed)


def load(name):
    d = pd.read_csv(DATA / f"{name}.csv", usecols=["review_text", "review_text_processed", "polarity", "kfold_polarity"])
    d = d[d.kfold_polarity > 0].dropna(subset=["polarity"])
    d["label"] = d.polarity.astype(int)
    d["review_text"] = d.review_text.fillna("").astype(str)
    d["review_text_processed"] = d.review_text_processed.fillna("").astype(str)
    tr, te = strat(d[d.kfold_polarity <= 9], N_TRAIN), strat(d[d.kfold_polarity == 10], N_TEST)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_bert", action="store_true")
    ap.add_argument("--skip_llm", action="store_true")
    a = ap.parse_args()
    data = {}
    for n in Bar(DOMAINS, desc="讀取資料集", unit="個"):
        data[n] = load(n)
        tr, te = data[n]
        log(f"{n}：訓練 {len(tr):,}（正面 {tr.label.mean():.1%}）、測試 {len(te):,}；平均 {te.review_text.str.split().str.len().mean():.1f} 個字")
    key = ["method", "train", "test"]

    rows = []
    for src in Bar(DOMAINS, desc="TF-IDF", unit="個"):
        vec = TfidfVectorizer(ngram_range=(1, 3), min_df=6, max_features=10000)
        X = vec.fit_transform(data[src][0].review_text_processed)
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(X, data[src][0].label)
        for tgt in DOMAINS:
            te = data[tgt][1]
            rows.append(dict(method="TF-IDF", train=src, test=tgt,
                             auc=round(roc_auc_score(te.label, clf.predict_proba(vec.transform(te.review_text_processed))[:, 1]), 4)))
    merge_save(rows, "ext_crossdomain.csv", key)
    log("TF-IDF 完成")

    if not a.skip_bert:
        for src in DOMAINS:
            tr = data[src][0]
            model, tok, minutes = BU.train("bertimbau-base", tr.review_text.tolist(), tr.label.values, max_len=128, desc=f"BERT 訓練於 {src}")
            rows = []
            for tgt in DOMAINS:
                te = data[tgt][1]
                p = BU.predict(model, BU.encode(tok, te.review_text, 128))
                rows.append(dict(method="BERTimbau Base 微調", train=src, test=tgt, auc=round(roc_auc_score(te.label, p), 4),
                                 train_minutes=round(minutes, 2)))
            BU.free(model)
            merge_save(rows, "ext_crossdomain.csv", key)
            log(f"BERT 訓練於 {src}：" + "、".join(f"{r['test']} {r['auc']:.4f}" for r in rows))

    if not a.skip_llm:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import rq2_llm as L
        tok = AutoTokenizer.from_pretrained(L.MODELS["qwen-1.5b"])
        tok.padding_side = "left"
        ids = [tok.encode(w, add_special_tokens=False)[0] for w in L.LABELS]
        m = AutoModelForCausalLM.from_pretrained(L.MODELS["qwen-1.5b"], dtype=torch.float16).cuda()
        rows = []
        for tgt in DOMAINS:
            te = strat(data[tgt][1], N_LLM)
            s = L.score(m, tok, ids, te.review_text.tolist(), desc=f"Qwen-1.5B 零樣本：{tgt}")
            rows.append(dict(method="Qwen-1.5B 零樣本", train="（不訓練）", test=tgt, auc=round(roc_auc_score(te.label, s), 4)))
            log(f"Qwen-1.5B 零樣本 {tgt}：AUC {rows[-1]['auc']:.4f}")
        merge_save(rows, "ext_crossdomain.csv", key)
        del m
        torch.cuda.empty_cache()

    out = pd.read_csv(RESULTS / "ext_crossdomain.csv")
    for meth, g in out.groupby("method"):
        log_text(f"\n{meth}（列＝訓練資料集、欄＝測試資料集）：\n" + g.pivot(index="train", columns="test", values="auc").to_string())
    mats = [(m, g.pivot(index="train", columns="test", values="auc").reindex(index=DOMAINS, columns=DOMAINS))
            for m, g in out.groupby("method") if g.train.nunique() > 1]
    if mats:
        fig, axes = plt.subplots(1, len(mats), figsize=(6.2 * len(mats), 5))
        axes = np.atleast_1d(axes)
        for ax, (m, M) in zip(axes, mats):
            im = ax.imshow(M.values * 100, cmap="viridis", vmin=75, vmax=100)
            ax.set_xticks(range(5), DOMAINS, rotation=30)
            ax.set_yticks(range(5), DOMAINS)
            for i in range(5):
                for j in range(5):
                    if not np.isnan(M.values[i, j]):
                        ax.text(j, i, f"{100 * M.values[i, j]:.1f}", ha="center", va="center",
                                color="white" if M.values[i, j] < 0.9 else "black", fontsize=9)
            ax.set(xlabel="測試資料集", ylabel="訓練資料集", title=f"{m}：跨資料集 AUC（%）")
        fig.colorbar(im, ax=axes, fraction=0.025)
        fig.savefig(FIGURES / "ext_crossdomain.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    log(f"結果已存到 results/ext_crossdomain.csv、figures/ext_crossdomain.png；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
