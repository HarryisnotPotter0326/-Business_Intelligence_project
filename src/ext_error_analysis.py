"""加深 2：最好的模型還錯在哪裡？（錯誤分析）

- 主要模型：BERTimbau Large 微調（8 個種子的預測取平均，門檻 0.5）
- 比較：TF-IDF、Qwen-1.5B LoRA、Qwen-1.5B 零樣本 是否也判錯同一則（「所有模型都錯」＝很可能是標籤本身有問題）
- 依評分、文字長度、文字特徵（轉折詞、否定詞、問號…）分組計算錯誤率
- 匯出最有把握卻判錯的 120 則（假負評 60、假正評 60），供人工歸類原因

輸出：results/ext_error_*.csv
用法：python src/ext_error_analysis.py
"""
import glob
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import RESULTS, load_olist
from monitor import LOG_PATH, log, log_text

PRED = RESULTS / "preds"
TAGS = {
    "轉折詞（mas、porém…）": r"\b(mas|porem|porém|entretanto|contudo|todavia|so que|só que)\b",
    "否定詞（não、nunca、nem）": r"\b(nao|não|nunca|nem|jamais)\b",
    "問號": r"\?",
    "驚嘆號": r"!",
    "提到物流（entrega、prazo、chegou）": r"\b(entrega|entregue|prazo|chegou|chegaram|atraso|demor)",
    "提到退換貨（troca、devolução）": r"\b(troca|trocar|devolu|devolver|reembolso)",
    "正面詞（bom、ótimo、recomendo）": r"\b(bom|boa|otimo|ótimo|excelente|recomendo|gostei|perfeito|adorei)\b",
}


def load_preds():
    p = {}
    f = sorted(glob.glob(str(PRED / "finetune_bertimbau-large_n30464_lr2.5e-05_do0.1_E1_s*_ep1.npy")))
    p["BERTimbau Large 微調"] = np.mean([np.load(x) for x in f], axis=0)
    p["TF-IDF"] = np.load(PRED / "tfidf_paper.npy")
    for name, fn, thr in [("Qwen-1.5B LoRA", "llm_qwen-1.5b_lora_n30464_s42.npy", 0.0),
                          ("Qwen-1.5B 零樣本", "llm_qwen-1.5b_zeroshot_n0_s42.npy", 0.0)]:
        if (PRED / fn).exists():
            p[name] = np.load(PRED / fn)
    log(f"BERTimbau Large 微調：{len(f)} 個種子平均")
    return p


def main():
    _, _, test = load_olist()
    y = test.label.values
    p = load_preds()
    thr = {"BERTimbau Large 微調": 0.5, "TF-IDF": 0.5, "Qwen-1.5B LoRA": 0.0, "Qwen-1.5B 零樣本": 0.0}
    pred = {k: (v > thr[k]).astype(int) for k, v in p.items()}
    d = test[["review_text", "rating", "label"]].copy()
    d["n_words"] = d.review_text.str.split().str.len().fillna(0)
    d["p_bert"] = p["BERTimbau Large 微調"]
    for k, v in pred.items():
        d[f"錯_{k}"] = (v != y).astype(int)
    main_err = d["錯_BERTimbau Large 微調"] == 1
    log(f"BERTimbau Large 微調：AUC {roc_auc_score(y, d.p_bert):.4f}，錯 {main_err.sum()} / {len(d)} 則（{main_err.mean():.1%}）；"
        f"假正評（負評判成正評）{((d.label == 0) & main_err).sum()}、假負評 {((d.label == 1) & main_err).sum()}")

    # 1. 模型之間是否錯在同一則
    others = [k for k in pred if k != "BERTimbau Large 微調"]
    d["其他模型也錯的數量"] = d[[f"錯_{k}" for k in others]].sum(1)
    agree = d[main_err].其他模型也錯的數量.value_counts().sort_index().rename("則數").to_frame()
    agree["占 BERT 錯誤"] = (agree.則數 / main_err.sum()).round(3)
    log_text(f"\nBERT 判錯的評論，其他 {len(others)} 個模型也判錯的數量：\n" + agree.to_string())

    # 2. 依評分、長度分組的錯誤率
    by_rating = d.groupby("rating")["錯_BERTimbau Large 微調"].agg(錯誤率="mean", 則數="size").round(4)
    bins = [0, 2, 5, 10, 20, 1000]
    d["長度"] = pd.cut(d.n_words, bins, labels=["1–2 字", "3–5 字", "6–10 字", "11–20 字", ">20 字"], include_lowest=True)
    by_len = d.groupby("長度", observed=True)["錯_BERTimbau Large 微調"].agg(錯誤率="mean", 則數="size").round(4)
    log_text("\n錯誤率 × 評分：\n" + by_rating.to_string())
    log_text("\n錯誤率 × 文字長度：\n" + by_len.to_string())

    # 3. 文字特徵：在錯誤中出現的比例 vs 在正確中出現的比例
    low = d.review_text.fillna("").str.lower()
    rows = []
    for tag, pat in TAGS.items():
        has = low.str.contains(pat, regex=True)
        rows.append(dict(特徵=tag, 錯誤中比例=round(has[main_err].mean(), 3), 正確中比例=round(has[~main_err].mean(), 3),
                         有此特徵的錯誤率=round(main_err[has].mean(), 4), 則數=int(has.sum())))
        d[tag] = has.astype(int)
    tags = pd.DataFrame(rows).sort_values("有此特徵的錯誤率", ascending=False)
    log_text("\n文字特徵與錯誤：\n" + tags.to_string(index=False))

    # 4. 匯出最有把握卻判錯的評論，供人工歸類
    fp = d[(d.label == 0) & main_err].sort_values("p_bert", ascending=False).head(60).assign(錯誤類型="假正評：負評判成正評")
    fn = d[(d.label == 1) & main_err].sort_values("p_bert").head(60).assign(錯誤類型="假負評：正評判成負評")
    sample = pd.concat([fp, fn])[["錯誤類型", "rating", "p_bert", "其他模型也錯的數量", "n_words", "review_text"]]
    sample["p_bert"] = sample.p_bert.round(3)
    sample["人工歸類"] = ""  # 由人工（或 AI 助理協助）填寫：反諷、評分與文字矛盾、正負混雜、太短、其他

    by_rating.reset_index().to_csv(RESULTS / "ext_error_by_rating.csv", index=False)
    by_len.reset_index().to_csv(RESULTS / "ext_error_by_length.csv", index=False)
    tags.to_csv(RESULTS / "ext_error_tags.csv", index=False)
    agree.reset_index().to_csv(RESULTS / "ext_error_agreement.csv", index=False)
    sample.to_csv(RESULTS / "ext_error_sample.csv", index=False, encoding="utf-8-sig")
    log(f"結果已存到 results/ext_error_*.csv（人工歸類用：ext_error_sample.csv）；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
