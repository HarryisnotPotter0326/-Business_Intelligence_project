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
from matplotlib.patches import Ellipse, FancyArrowPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402

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
    """研究架構圖：泳道式（兩條主線 × 資料來源→研究問題→研究產出），採標準流程圖符號
    （圓柱＝資料、矩形＝分析、文件形＝產出）。以實際列印尺寸（約 16 cm 寬）繪製，座標單位為英吋。"""
    W, H = 6.3, 3.42
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ACC, LINE, TXT, TXT2 = "#1f3a5f", "#3a3a3a", "#1a1a1a", "#3f3f3f"
    BAND, HEAD = "#f4f5f7", "#e3e6ea"
    LW = 0.7

    def rect(x, y, w, h, fc, ec="none", lw=LW, z=1):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))

    def cylinder(x, y, w, h, title, body, e=0.13):
        ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, fc="white", ec=LINE, lw=LW, zorder=3))
        rect(x, y + e / 2, w, h - e, "white", z=3)
        ax.plot([x, x], [y + e / 2, y + h - e / 2], color=LINE, lw=LW, zorder=4)
        ax.plot([x + w, x + w], [y + e / 2, y + h - e / 2], color=LINE, lw=LW, zorder=4)
        ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, fc="white", ec=LINE, lw=LW, zorder=4))
        cy = y + (h - e) / 2 + 0.02
        ax.text(x + w / 2, cy + 0.2, title, ha="center", va="center", fontsize=7.4, fontweight="bold", color=TXT, zorder=5)
        ax.text(x + w / 2, cy - 0.1, body, ha="center", va="center", fontsize=6.5, color=TXT2, linespacing=1.45,
                zorder=5)

    def process(x, y, w, h, tag, title, body, hh=0.25):
        rect(x, y, w, h, "white", LINE, z=3)
        rect(x, y + h - hh, w, hh, ACC, LINE, z=4)
        ax.text(x + 0.08, y + h - hh / 2, f"{tag}　{title}", ha="left", va="center", fontsize=7.4, fontweight="bold",
                color="white", zorder=5)
        ax.text(x + 0.09, y + (h - hh) / 2, body, ha="left", va="center", fontsize=6.6, color=TXT2, linespacing=1.55,
                zorder=5)

    def document(x, y, w, h, text, a=0.055):
        verts = [(x, y + a), (x, y + h), (x + w, y + h), (x + w, y + a),
                 (x + 0.72 * w, y - 0.9 * a), (x + 0.32 * w, y + 2.9 * a), (x, y + a)]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        ax.add_patch(PathPatch(Path(verts, codes), fc="white", ec=LINE, lw=LW, zorder=3))
        ax.text(x + w / 2, y + h / 2 + a * 0.6, text, ha="center", va="center", fontsize=7.0, fontweight="bold",
                color=TXT, linespacing=1.5, zorder=5)

    def arrow(p, q, ls="-", c=LINE):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=c, ls=ls,
                                     shrinkA=0, shrinkB=0, zorder=6))

    # 欄：泳道標籤｜資料來源｜研究問題與分析方法（兩欄）｜研究產出
    xl, xd, xa, xb, xo = (0.0, 0.34), (0.47, 1.47), (1.70, 3.22), (3.44, 4.96), (5.19, 6.27)
    head_y, head_h = 3.12, 0.28
    lanes = [(1.79, 1.31, "人工智慧"), (0.40, 1.31, "資料探勘與智慧決策")]
    # 表頭
    rect(xl[0], head_y, xl[1] - xl[0], head_h, HEAD)
    rect(xd[0] - 0.08, head_y, xd[1] - xd[0] + 0.16, head_h, HEAD)
    rect(xa[0] - 0.08, head_y, xb[1] - xa[0] + 0.16, head_h, HEAD)
    rect(xo[0] - 0.08, head_y, W - xo[0] + 0.08, head_h, HEAD)
    for (x0, x1), t in [(xl, "主線"), ((xd[0] - 0.08, xd[1] + 0.08), "資料來源"),
                        ((xa[0] - 0.08, xb[1] + 0.08), "研究問題與分析方法"), ((xo[0] - 0.08, W), "研究產出")]:
        ax.text((x0 + x1) / 2, head_y + head_h / 2, t, ha="center", va="center", fontsize=7.6, fontweight="bold",
                color=TXT)
    # 泳道
    for y0, h, name in lanes:
        rect(xl[0], y0, xl[1] - xl[0], h, HEAD)
        rect(xl[1] + 0.05, y0, W - xl[1] - 0.05, h, BAND)
        ax.text((xl[0] + xl[1]) / 2, y0 + h / 2, "\n".join(name), ha="center", va="center", fontsize=7.4,
                fontweight="bold", color=TXT, linespacing=1.08)
    pad = 0.13
    (ty, th, _), (by, bh, _) = lanes
    boxes = {}
    for (y0, h, _), dat, p1, p2, out in [
        (lanes[0], ("評論資料", "作者公開的切分\n38,079 則評論\n（排除 3 星）"),
         ("RQ1", "重現原論文", "TF-IDF 基準\nBERT 凍結向量 vs. 微調\n8 個種子、bootstrap 檢定\n錯誤原因分析"),
         ("RQ2", "資料量與 LLM", "學習曲線（1k–30k 筆）\nQwen2.5 提示與 LoRA 微調\n跨資料集泛化\n準確度與運算成本"),
         "論文結論的修正\n模型選型建議"),
        (lanes[1], ("Olist 原始資料", "9 張資料表\n98,673 筆訂單\n2016-09 至 2018-10"),
         ("RQ3", "負評成因", "五種分類模型＋SHAP\n因果推論（PSM、AIPW、RDD）\n負評主題與顧客分群\n文字中的隱性不滿"),
         ("RQ4", "預警與決策", "三個預警時點比較\nPrecision@K 與成本模型\n蒙地卡羅成本模擬\n可避免負評與優先順序"),
         "負評成因與\n七項管理建議"),
    ]:
        bx, bh_ = y0 + pad, h - 2 * pad
        mid = y0 + h / 2
        cylinder(xd[0], bx, xd[1] - xd[0], bh_, *dat)
        process(xa[0], bx, xa[1] - xa[0], bh_, *p1)
        process(xb[0], bx, xb[1] - xb[0], bh_, *p2)
        document(xo[0], mid - 0.33, xo[1] - xo[0], 0.66, out)
        arrow((xd[1], mid), (xa[0], mid))
        arrow((xa[1], mid), (xb[0], mid))
        arrow((xb[1], mid), (xo[0], mid))
        boxes[p1[0]] = (bx, bh_)
    # 跨主線：RQ1 的 BERT 套用到訂單評論
    x_link = xa[0] + 0.42
    arrow((x_link, boxes["RQ1"][0]), (x_link, boxes["RQ3"][0] + boxes["RQ3"][1]), ls=(0, (3, 2)), c=ACC)
    ax.text(x_link + 0.07, (ty + by + bh) / 2, "BERT 交叉評分 40,748 則留言", ha="left", va="center", fontsize=6.4,
            color=ACC, zorder=6)
    # 圖例
    ly = 0.17
    items = []
    lx = 1.35
    ax.add_patch(Ellipse((lx, ly - 0.045), 0.16, 0.05, fc="white", ec=LINE, lw=0.6, zorder=3))
    rect(lx - 0.08, ly - 0.045, 0.16, 0.09, "white", z=3)
    ax.plot([lx - 0.08, lx - 0.08], [ly - 0.045, ly + 0.045], color=LINE, lw=0.6, zorder=4)
    ax.plot([lx + 0.08, lx + 0.08], [ly - 0.045, ly + 0.045], color=LINE, lw=0.6, zorder=4)
    ax.add_patch(Ellipse((lx, ly + 0.045), 0.16, 0.05, fc="white", ec=LINE, lw=0.6, zorder=4))
    items.append((lx + 0.13, "資料"))
    lx = 2.05
    rect(lx - 0.08, ly - 0.06, 0.16, 0.12, "white", LINE, lw=0.6, z=3)
    rect(lx - 0.08, ly + 0.02, 0.16, 0.04, ACC, LINE, lw=0.6, z=4)
    items.append((lx + 0.13, "分析方法"))
    lx = 2.92
    a = 0.02
    verts = [(lx - 0.08, ly - 0.06 + a), (lx - 0.08, ly + 0.06), (lx + 0.08, ly + 0.06), (lx + 0.08, ly - 0.06 + a),
             (lx + 0.035, ly - 0.06 - a), (lx - 0.03, ly - 0.06 + 3 * a), (lx - 0.08, ly - 0.06 + a)]
    ax.add_patch(PathPatch(Path(verts, [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CURVE4] * 3), fc="white", ec=LINE,
                           lw=0.6, zorder=3))
    items.append((lx + 0.13, "研究產出"))
    lx = 3.75
    arrow((lx - 0.1, ly), (lx + 0.12, ly))
    items.append((lx + 0.18, "流程"))
    lx = 4.4
    arrow((lx - 0.1, ly), (lx + 0.12, ly), ls=(0, (3, 2)), c=ACC)
    items.append((lx + 0.18, "跨主線的連結"))
    for x, t in items:
        ax.text(x, ly, t, ha="left", va="center", fontsize=6.4, color=TXT2)
    fig.savefig(FIGURES / "research_flow.png", dpi=300)
    plt.close(fig)
    log("已存 figures/research_flow.png")


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
