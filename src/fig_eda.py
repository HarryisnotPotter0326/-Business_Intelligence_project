"""報告第 3、4 章用圖：研究流程圖與探索式資料分析（EDA）

  figures/research_flow.png   研究架構與流程
  figures/eda_reviews.png     評論資料：星等分布、文字長度
  figures/eda_monthly.png     訂單資料：每月訂單數與負評率（標出依時間切分的界線）
  results/eda_stats.csv       EDA 描述統計（報告第 4 章引用）
用法：python src/fig_eda.py
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from common import DATA, FIGURES, RESULTS  # noqa: E402
from monitor import LOG_PATH, log, log_text  # noqa: E402
import rq3_data  # noqa: E402

C_POS, C_NEG, C_MID = "#2a78d6", "#eb6834", "#898781"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": ["Microsoft JhengHei", "sans-serif"], "axes.unicode_minus": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"已存 figures/{name}")


def fig_flow():
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
        ax.text(x + 0.15, y + h - 0.2, title, ha="left", va="top", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x + 0.15, y + h - 0.58, body, ha="left", va="top", fontsize=8.6, color=INK2, linespacing=1.45)

    def arrow(p, q, style="-|>", ls="-", c=INK2, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=1.3, color=c, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}"))

    blue_fc, blue_ec = "#eaf2fc", "#2a78d6"
    org_fc, org_ec = "#fdeee7", "#eb6834"
    gry_fc, gry_ec = "#f3f2ef", "#898781"
    # 上排：評論資料 → RQ1 → RQ2（人工智慧）；下排：訂單資料 → RQ3 → RQ4（資料探勘與決策）
    ax.text(0.2, 5.25, "人工智慧：評論文字的情感分類", fontsize=10, color=blue_ec, fontweight="bold", va="top")
    ax.text(0.2, 2.55, "資料探勘與智慧決策：訂單層級的負評分析", fontsize=10, color=org_ec, fontweight="bold", va="top")
    h, yt, yb = 2.0, 2.95, 0.25
    box(0.2, yt, 2.9, h, "評論資料（作者切分）",
        "Olist 評論 41,744 則；3 星排除\n1–2 星＝負、4–5 星＝正\n訓練 30,464／驗證 3,808\n測試 3,807", gry_fc, gry_ec)
    box(3.75, yt, 3.85, h, "RQ1　重現",
        "TF-IDF＋LR（基準）\nBERT 凍結向量＋LR（9 種聚合）\nBERT 微調；8 個種子、bootstrap 檢定\n超參數搜尋、錯誤分析", blue_fc, blue_ec)
    box(8.15, yt, 3.65, h, "RQ2　延伸",
        "學習曲線（1k～30k 筆）\nQwen2.5 零樣本／少樣本／LoRA\n門檻校正、集成、跨資料集\n推論速度與記憶體", blue_fc, blue_ec)
    box(0.2, yb, 2.9, h, "Olist 原始資料（9 張表）",
        "訂單、品項、評論、商品、賣家\n顧客、付款、地理位置、品類\n→ 98,673 筆有評論的訂單\n（2016-09～2018-10）", gry_fc, gry_ec)
    box(3.75, yb, 3.85, h, "RQ3　資料探勘",
        "負評預測（LR、DT、RF、XGB、LGBM）\nSHAP、規則、各因素負評率\n因果推論（PSM、AIPW、斷點迴歸）\n主題模型、分群、滾動回測", org_fc, org_ec)
    box(8.15, yb, 3.65, h, "RQ4　智慧決策",
        "三個預警時點、Precision@K\n成本敏感的名單長度\n蒙地卡羅成本模擬\n可避免負評 → 管理建議", org_fc, org_ec)
    for y in [yt + h / 2, yb + h / 2]:
        arrow((3.1, y), (3.75, y))
        arrow((7.6, y), (8.15, y))
    arrow((5.0, yt), (5.0, yb + h), ls="--", c=blue_ec)
    ax.text(5.1, (yt + yb + h) / 2, "BERT 交叉評分有留言的評論\n（3 星評論、隱性不滿）", fontsize=8, color=blue_ec, va="center")
    save(fig, "research_flow.png")


def fig_reviews(stats):
    d = pd.read_csv(DATA / "olist.csv", usecols=["review_text", "rating", "polarity", "kfold_polarity"])
    d["n_words"] = d.review_text.fillna("").astype(str).str.split().str.len()
    cnt = d.rating.value_counts().sort_index()
    used = d[d.kfold_polarity > 0]
    stats += [("評論資料總則數", len(d)), ("3 星（排除）", int((d.rating == 3).sum())), ("建模用則數", len(used)),
              ("正面比例", round(used.polarity.mean(), 4)),
              ("平均字數（全部）", round(used.n_words.mean(), 2)), ("中位數字數（全部）", used.n_words.median()),
              ("平均字數（負面）", round(used[used.polarity == 0].n_words.mean(), 2)),
              ("平均字數（正面）", round(used[used.polarity == 1].n_words.mean(), 2)),
              ("字數 ≤ 3 的比例", round((used.n_words <= 3).mean(), 4))]
    for s in range(1, 6):
        stats.append((f"{s} 星則數", int(cnt.get(s, 0))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    colors = [C_NEG, C_NEG, C_MID, C_POS, C_POS]
    ax.bar(cnt.index, cnt.values, color=colors, width=0.62)
    for s, v in cnt.items():
        ax.text(s, v + 300, f"{v:,}\n（{v / cnt.sum():.1%}）", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(1, 6), [f"{s} 星" for s in range(1, 6)])
    ax.set_ylim(0, cnt.max() * 1.22)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("評論則數")
    ax.set_title("（a）星等分布（橘＝負面、藍＝正面、灰＝排除）")
    ax = axes[1]
    bins = np.arange(0, 42, 1)
    for lab, c, name in [(0, C_NEG, "負面（1–2 星）"), (1, C_POS, "正面（4–5 星）")]:
        x = used[used.polarity == lab].n_words.clip(upper=40)
        ax.hist(x, bins=bins, density=True, histtype="step", lw=2, color=c,
                label=f"{name}：平均 {x.mean():.1f} 字")
    ax.set(xlabel="評論字數（40 字以上併入最後一格）", ylabel="比例")
    ax.legend(fontsize=9)
    ax.set_title("（b）文字長度分布：負評較長")
    fig.tight_layout()
    save(fig, "eda_reviews.png")


def fig_monthly(stats):
    d = rq3_data.load()
    d["month"] = d.purchase_ts.dt.to_period("M")
    m = d.groupby("month").agg(orders=("order_id", "size"), neg=("neg", "mean"), late=("is_late", "mean")).reset_index()
    m = m[(m.month >= pd.Period("2017-01", "M")) & (m.month <= pd.Period("2018-08", "M"))]
    stats += [("訂單數", len(d)), ("負評率（1–2 分）", round(d.neg.mean(), 4)), ("負評率（1–3 分）", round(d.neg3.mean(), 4)),
              ("已送達比例", round(d.delivered.mean(), 4)), ("延遲比例（已送達中）", round(d.loc[d.delivered == 1, "is_late"].mean(), 4)),
              ("準時送達負評率", round(d.loc[(d.delivered == 1) & (d.is_late == 0), "neg"].mean(), 4)),
              ("延遲送達負評率", round(d.loc[(d.delivered == 1) & (d.is_late == 1), "neg"].mean(), 4)),
              ("未送達負評率", round(d.loc[d.delivered == 0, "neg"].mean(), 4)),
              ("單一賣家訂單比例", round((d.n_sellers == 1).mean(), 4)), ("單一品項訂單比例", round((d.n_items == 1).mean(), 4)),
              ("訂單金額中位數", round(d.price_total.median(), 2)), ("運費占比中位數", round(d.freight_ratio.median(), 4)),
              ("顧客州 SP 比例", round((d.customer_state == "SP").mean(), 4))]
    log_text("\n每月訂單與負評率：\n" + m.to_string(index=False))
    x = np.arange(len(m))
    labels = [str(p) for p in m.month]
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.2]))
    ax = axes[0]
    ax.bar(x, m.orders, color=C_MID, width=0.7)
    i = labels.index("2017-11")
    ax.annotate(f"2017-11（黑色星期五）\n{m.orders.iat[i]:,} 筆", (x[i], m.orders.iat[i]), xytext=(-60, -6),
                textcoords="offset points", ha="right", va="top", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_ylabel("有評論的訂單數")
    ax.grid(axis="x", visible=False)
    ax.set_title("（a）每月訂單數")
    ax = axes[1]
    ax.plot(x, 100 * m.neg, "o-", c=C_NEG, lw=2, ms=5, label="負評率（1–2 分）")
    ax.plot(x, 100 * m.late, "s--", c=C_POS, lw=1.6, ms=4, label="延遲送達比例")
    ax.set_ylabel("%")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("（b）每月負評率與延遲比例：兩者同步起伏")
    for a in axes:
        for cut, txt in [("2018-04", "驗證期起"), ("2018-06", "測試期起")]:
            j = labels.index(cut) - 0.5
            a.axvline(j, c=INK, ls=":", lw=1)
            if a is axes[1]:
                a.text(j + 0.1, a.get_ylim()[1] * 0.97, txt, fontsize=8.5, color=INK, va="top")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    fig.tight_layout()
    save(fig, "eda_monthly.png")


def main():
    stats = []
    fig_flow()
    fig_reviews(stats)
    fig_monthly(stats)
    s = pd.DataFrame(stats, columns=["項目", "數值"])
    s.to_csv(RESULTS / "eda_stats.csv", index=False)
    log_text("\nEDA 描述統計：\n" + s.to_string(index=False))
    log(f"完成；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
