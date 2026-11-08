"""加深 8：把 RQ1 的 BERT 用在真實業務上（串起 RQ1 與 RQ3／RQ4）

對 Olist 原始評論表中「有留言」的每一則評論，用 BERTimbau Large 微調模型估計「文字是正面的機率」。
為了不讓模型看過自己要評分的評論，採 2 折交叉評分：依訂單隨機分成兩半，用 A 半（1–2、4–5 分）訓練、
對 B 半評分，再反過來；3 分評論從不參與訓練，取兩個模型的平均。

分析：
  1. 3 分評論（論文排除的中性評論）的文字偏正還偏負？偏負的 3 分評論和物流有沒有關係？
  2. 「隱性不滿」：給 4–5 星、文字卻是負面的評論有多少？和延遲有沒有關係？
  3. 「評分與文字矛盾」：給 1–2 星、文字卻很正面（可能是按錯星等）
  4. 連續的情緒分數 vs 延遲天數：就算是 4–5 星，延遲也會讓文字變負面嗎？
輸出：results/ext_apply_*.csv；figures/ext_apply_bert.png
用法：python src/ext_apply_bert.py
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import bert_utils as BU  # noqa: E402
from common import FIGURES, RESULTS  # noqa: E402
from monitor import LOG_PATH, log, log_text  # noqa: E402
import rq3_data  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
MODEL = "bertimbau-large"


def main():
    d = rq3_data.load()
    d = d[d.review_comment_message.fillna("").str.strip().str.len() > 0].copy().reset_index(drop=True)
    d["text"] = d.review_comment_message.str.strip()
    d["half"] = (np.random.default_rng(42).random(len(d)) < 0.5).astype(int)
    d["lab"] = np.where(d.review_score <= 2, 0, np.where(d.review_score >= 4, 1, -1))
    log(f"有留言的評論 {len(d):,} 則；1–2 分 {(d.lab == 0).sum():,}、3 分 {(d.lab == -1).sum():,}、4–5 分 {(d.lab == 1).sum():,}")

    d["p_pos"] = np.nan
    three = d.lab == -1
    p3 = []
    for h in [0, 1]:
        tr = d[(d.half == h) & (d.lab >= 0)]
        model, tok, minutes = BU.train(MODEL, tr.text.tolist(), tr.lab.values, desc=f"交叉評分 {h + 1}/2")
        other = d.half != h
        p = BU.predict(model, BU.encode(tok, d.loc[other & ~three, "text"], 60), desc="評分另一半")
        d.loc[other & ~three, "p_pos"] = p
        p3.append(BU.predict(model, BU.encode(tok, d.loc[three, "text"], 60), desc="評分 3 分評論"))
        BU.free(model)
        log(f"交叉評分 {h + 1}/2 完成，訓練 {minutes:.1f} 分")
    d.loc[three, "p_pos"] = np.mean(p3, axis=0)
    from sklearn.metrics import roc_auc_score
    lab = d.lab >= 0
    log(f"交叉評分 AUC（1–2 vs 4–5 分）：{roc_auc_score(d.lab[lab], d.p_pos[lab]):.4f}")
    d[["review_id", "order_id", "review_score", "p_pos"]].to_csv(RESULTS / "ext_apply_scores.csv", index=False)

    # 1. 各星等的文字情緒分布
    by_star = d.groupby("review_score").p_pos.agg(平均正面機率="mean", 文字偏負比例=lambda x: (x < 0.5).mean(), 則數="size").round(3)
    log_text("\n各星等的文字情緒：\n" + by_star.to_string())

    # 3 分評論：偏負 vs 偏正，和物流的關係
    t3 = d[three].assign(傾向=lambda x: np.where(x.p_pos < 0.5, "文字偏負", "文字偏正"))
    t3s = t3.groupby("傾向").agg(則數=("p_pos", "size"), 延遲率=("is_late", "mean"), 未送達率=("delivered", lambda x: 1 - x.mean()),
                                平均延遲天數=("delay_days", "mean"), 多賣家比例=("n_sellers", lambda x: (x > 1).mean())).round(3)
    log_text("\n3 分評論的文字傾向與物流：\n" + t3s.to_string())

    # 2. 隱性不滿：4–5 星但文字偏負
    pos = d[d.review_score >= 4].copy()
    pos["隱性不滿"] = pos.p_pos < 0.5
    hid = pos.groupby("隱性不滿").agg(則數=("p_pos", "size"), 延遲率=("is_late", "mean"), 平均延遲天數=("delay_days", "mean"),
                                    多賣家比例=("n_sellers", lambda x: (x > 1).mean()), 平均評分=("review_score", "mean")).round(3)
    log(f"隱性不滿：4–5 星評論中文字偏負的比例 {pos.隱性不滿.mean():.1%}（{pos.隱性不滿.sum():,} 則）")
    log_text(hid.to_string())

    # 3. 評分與文字矛盾：1–2 星但文字非常正面
    neg = d[d.review_score <= 2]
    contra = neg[neg.p_pos > 0.9]
    log(f"評分與文字矛盾：1–2 星但文字正面機率 > 0.9 的有 {len(contra):,} 則（{len(contra) / len(neg):.1%}）")

    ex = pd.concat([
        pos[pos.隱性不滿].nsmallest(20, "p_pos").assign(類型="隱性不滿（4–5 星、文字負面）"),
        contra.nlargest(20, "p_pos").assign(類型="矛盾（1–2 星、文字正面）"),
        t3.nsmallest(10, "p_pos").assign(類型="3 分、文字最負面"),
        t3.nlargest(10, "p_pos").assign(類型="3 分、文字最正面"),
    ])[["類型", "review_score", "p_pos", "is_late", "delay_days", "category", "text"]]
    ex["p_pos"] = ex.p_pos.round(3)
    ex.to_csv(RESULTS / "ext_apply_examples.csv", index=False, encoding="utf-8-sig")
    log_text("\n範例：\n" + ex.to_string(index=False, max_colwidth=100))

    # 4. 延遲天數 vs 文字情緒（全部、只看 4–5 星）
    bins = [-np.inf, -10, -5, -2, 0, 2, 5, 10, np.inf]
    lab_b = ["提早>10", "提早5–10", "提早2–5", "提早0–2", "延遲0–2", "延遲2–5", "延遲5–10", "延遲>10"]
    d["延遲分組"] = pd.cut(d.delay_days, bins, labels=lab_b)
    curve = pd.DataFrame({
        "全部評論_平均正面機率": d.groupby("延遲分組", observed=True).p_pos.mean(),
        "4–5星_平均正面機率": d[d.review_score >= 4].groupby("延遲分組", observed=True).p_pos.mean(),
        "4–5星_隱性不滿比例": d[d.review_score >= 4].groupby("延遲分組", observed=True).p_pos.apply(lambda x: (x < 0.5).mean()),
        "則數": d.groupby("延遲分組", observed=True).size()}).round(3)
    log_text("\n延遲 vs 文字情緒：\n" + curve.to_string())

    by_star.reset_index().to_csv(RESULTS / "ext_apply_by_star.csv", index=False)
    t3s.reset_index().to_csv(RESULTS / "ext_apply_3star.csv", index=False)
    hid.reset_index().to_csv(RESULTS / "ext_apply_hidden.csv", index=False)
    curve.reset_index().to_csv(RESULTS / "ext_apply_delay.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for s, col in zip(range(1, 6), ["#b03a2e", "#d9822b", "#999999", "#7fa7d1", "#3b6ea5"]):
        axes[0].hist(d.p_pos[d.review_score == s], bins=40, histtype="step", lw=2, density=True, color=col, label=f"{s} 星")
    axes[0].set(yscale="log", xlabel="BERT 判斷文字為正面的機率", ylabel="密度（對數）", title="各星等評論的文字情緒分布")
    axes[0].legend()
    x = np.arange(len(curve))
    axes[1].plot(x, curve["4–5星_隱性不滿比例"] * 100, "o-", c="#b03a2e", label="4–5 星中「文字負面」的比例")
    axes[1].set_xticks(x, curve.index.astype(str), rotation=30)
    axes[1].set(ylabel="%", title="就算給了 4–5 星，延遲越久，文字越負面")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "ext_apply_bert.png", dpi=200)
    plt.close(fig)
    log(f"結果已存到 results/ext_apply_*.csv、figures/ext_apply_bert.png；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
