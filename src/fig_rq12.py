"""RQ1／RQ2 報告用圖（全部從 results/ 已存的結果畫，不重新訓練）

  figures/rq1_forest.png      各方法測試 AUC 與 95% 信賴區間，對照論文數值
  figures/rq1_sweep.png       微調超參數搜尋（學習率 × dropout × epoch 數）
  figures/rq1_tfidf_gap.png   TF-IDF 與論文的差距（正則化強度、10 折交叉驗證）
  figures/rq1_error.png       錯誤分析（星等、文字長度、文字特徵）
  figures/rq2_learning_curve.png  學習曲線：訓練資料量 vs 測試 AUC
  figures/rq2_llm.png         LLM：零樣本／少樣本／LoRA，以及門檻校正
  figures/rq2_pareto.png      準確度 vs 推論速度

顏色固定代表方法類別（每張圖都一樣）：灰＝TF-IDF、綠＝BERT 凍結、藍＝BERT 微調、橘＝LLM。
用法：python src/fig_rq12.py [--only forest,sweep,...]
"""
import argparse
import glob

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from common import FIGURES, RESULTS, load_olist  # noqa: E402
from monitor import LOG_PATH, Bar, log  # noqa: E402

PRED = RESULTS / "preds"
# 類別色（dataviz 參考色盤前 3 格，已驗證色盲可分；TF-IDF 用中性灰）
C_TFIDF, C_FROZEN, C_FT, C_LLM = "#898781", "#1baf7a", "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = {1e-5: "#86b6ef", 2.5e-5: "#2a78d6", 5e-5: "#104281"}
# 論文 Figure 1 的 Olist 數值（凍結取 first+mean+std）；TF-IDF 取前一篇 Table VI；表 3 的 Large 微調另外標示
PAPER = {"TF-IDF ＋ LR": 96.5, "m-BERT 凍結": 97.0, "BERTimbau Base 凍結": 97.8, "BERTimbau Large 凍結": 98.1,
         "m-BERT 微調": 97.6, "BERTimbau Base 微調": 98.5, "BERTimbau Large 微調": 98.6}
PAPER_T3 = {"BERTimbau Large 微調": 97.9}

plt.rcParams.update({
    "font.family": ["Microsoft JhengHei", "sans-serif"], "axes.unicode_minus": False,
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name):
    # 報告採學術格式：圖名寫在圖下方的圖說，圖內不放總標題（多面板圖保留（a）（b）小標）
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    for ax in fig.axes:
        if len(fig.axes) == 1 or not ax.get_title(loc="left").startswith("（"):
            ax.set_title("", loc="left")
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"已存 figures/{name}")


# ---------------------------------------------------------------- 1. 森林圖
FOREST = [  # （名稱, 類別色, 預測檔樣式）
    ("TF-IDF ＋ LR", C_TFIDF, "tfidf_paper.npy"),
    ("m-BERT 凍結", C_FROZEN, "frozen_mbert_valbest.npy"),
    ("BERTimbau Base 凍結", C_FROZEN, "frozen_bertimbau-base_valbest.npy"),
    ("BERTimbau Large 凍結", C_FROZEN, "frozen_bertimbau-large_valbest.npy"),
    ("m-BERT 微調", C_FT, "finetune_mbert_n30464_lr2.5e-05_do0_E2_s*_ep2.npy"),
    ("BERTimbau Base 微調", C_FT, "finetune_bertimbau-base_n30464_lr2.5e-05_do0.1_E1_s*_ep1.npy"),
    ("BERTimbau Large 微調", C_FT, "finetune_bertimbau-large_n30464_lr2.5e-05_do0.1_E1_s*_ep1.npy"),
    ("Qwen-1.5B 零樣本", C_LLM, "llm_qwen-1.5b_zeroshot_n0_s42.npy"),
    ("Qwen-3B 零樣本", C_LLM, "llm_qwen-3b_zeroshot_n0_s42.npy"),
    ("Qwen-0.5B LoRA", C_LLM, "llm_qwen-0.5b_lora_n30464_s??.npy"),
    ("Qwen-1.5B LoRA", C_LLM, "llm_qwen-1.5b_lora_n30464_s??.npy"),
    # 3B：第二輪（11/20）學習率 1e-4 的 3 個種子；第一次（學習率 2e-4）的預測檔是 llm_qwen-3b_lora_n30464_s42.npy
    ("Qwen-3B LoRA", C_LLM, "llm_qwen-3b_lora_n30464_s??_lr0.0001_E1_ep1.npy"),
]


def forest_table(B=2000):
    """點估計＝各種子 AUC 平均；CI＝同一組 bootstrap 樣本上「各種子 AUC 平均」的 2.5／97.5 百分位。"""
    cache = RESULTS / "fig_forest_ci.csv"
    if cache.exists():
        return pd.read_csv(cache)
    y = load_olist()[2].label.values
    rng = np.random.default_rng(42)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    rows = []
    for name, _, pat in Bar(FOREST, desc="bootstrap 信賴區間", unit="個"):
        files = sorted(f for f in glob.glob(str(PRED / pat)) if not f.endswith("_val.npy"))
        ps = [np.load(f) for f in files]
        point = np.mean([roc_auc_score(y, p) for p in ps])
        boot = np.mean([[roc_auc_score(y[i], p[i]) for i in idx] for p in ps], axis=0)
        rows.append(dict(method=name, seeds=len(ps), auc=100 * point,
                         lo=100 * np.percentile(boot, 2.5), hi=100 * np.percentile(boot, 97.5)))
    t = pd.DataFrame(rows)
    t.to_csv(cache, index=False)
    return t


def fig_forest():
    t = forest_table()
    color = {n: c for n, c, _ in FOREST}
    t = t.sort_values("auc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for i, r in t.iterrows():
        c = color[r.method]
        ax.plot([r.lo, r.hi], [i, i], c=c, lw=2, solid_capstyle="round")
        ax.plot(r.auc, i, "o", ms=8, c=c, mec="white", mew=1.5, zorder=3)
        right = max(r.hi, PAPER.get(r.method, 0) + 0.05)
        ax.text(right + 0.06, i, f"{r.auc:.2f}", va="center", fontsize=9, color=INK)
        if r.method in PAPER:
            ax.plot(PAPER[r.method], i, "D", ms=7, mfc="white", mec=INK, mew=1.4, zorder=4)
        if r.method in PAPER_T3:
            ax.plot(PAPER_T3[r.method], i, "s", ms=7, mfc="white", mec=INK2, mew=1.4, zorder=4)
    labels = [f"{m}（{s} 個種子）" if s > 1 else m for m, s in zip(t.method, t.seeds)]
    ax.set_yticks(range(len(t)), labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("測試集 ROC-AUC（%）")
    ax.set_xlim(95.4, 99.3)
    ax.set_title("RQ1：各方法的測試 AUC 與 95% 信賴區間")
    handles = [Line2D([], [], marker="o", ls="-", c=c, mec="white", ms=8, label=l) for l, c in
               [("TF-IDF", C_TFIDF), ("BERT 凍結＋LR", C_FROZEN), ("BERT 微調", C_FT), ("LLM", C_LLM)]]
    handles.append(Line2D([], [], marker="D", ls="", mfc="white", mec=INK, ms=7, label="論文圖 1"))
    handles.append(Line2D([], [], marker="s", ls="", mfc="white", mec=INK2, ms=7, label="論文表 3"))
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    save(fig, "rq1_forest.png")


# ---------------------------------------------------------------- 2. 超參數搜尋
def fig_sweep():
    s = pd.read_csv(RESULTS / "rq1_sweep.csv")
    s = s[s.epoch == s.epochs]
    g = s.groupby(["model", "lr", "dropout", "epochs"]).agg(val=("val_auc", "mean"), auc=("test_auc", "mean"),
                                                          sd=("test_auc", "std")).reset_index()
    g[["val", "auc", "sd"]] *= 100
    panels = [("bertimbau-base", 0.0, "Base，dropout 0"), ("bertimbau-base", 0.1, "Base，dropout 0.1（論文）"),
              ("bertimbau-large", 0.1, "Large，dropout 0.1（論文）")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, gridspec_kw=dict(width_ratios=[3, 3, 2]))
    xs = {1: 0, 2: 1, 4: 2}
    for ax, (m, do, title) in zip(axes, panels):
        sub = g[(g.model == m) & (g.dropout == do)]
        best = g[g.model == m].sort_values("val", ascending=False).iloc[0]
        for lr, c in BLUE_RAMP.items():
            d = sub[np.isclose(sub.lr, lr)].sort_values("epochs")
            if d.empty:
                continue
            x = [xs[e] for e in d.epochs]
            ax.errorbar(x, d.auc, yerr=d.sd, fmt="o-", c=c, ms=6, capsize=3, lw=2, label=f"學習率 {lr:g}")
            for xi, (_, r) in zip(x, d.iterrows()):
                if np.isclose(r.lr, best.lr) and r.dropout == best.dropout and r.epochs == best.epochs:
                    ax.plot(xi, r.auc, "*", ms=16, mfc="none", mec=INK, mew=1.2, zorder=5)
                if np.isclose(r.lr, 2.5e-5) and r.dropout == 0.1 and r.epochs == 1:
                    ax.annotate("論文設定", (xi, r.auc), xytext=(10, 14), textcoords="offset points", fontsize=8.5,
                                color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.set_xticks(range(3 if m == "bertimbau-base" else 2), ["1", "2", "4"][:3 if m == "bertimbau-base" else 2])
        ax.set_xlim(-0.4, (2 if m == "bertimbau-base" else 1) + 0.4)
        ax.set_xlabel("訓練 epoch 數")
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("測試 AUC（%，3 個種子平均 ± 標準差）")
    axes[0].legend(loc="lower left", fontsize=9)
    handles = [Line2D([], [], marker="*", ls="", mfc="none", mec=INK, ms=14, label="驗證 AUC 最佳的組合")]
    axes[2].legend(handles=handles, loc="lower right", fontsize=9)
    fig.suptitle("RQ1：微調超參數搜尋——訓練 1–2 個 epoch 最好，4 個 epoch 會過度配適；dropout 影響很小",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "rq1_sweep.png")


# ---------------------------------------------------------------- 3. TF-IDF 差距
def fig_tfidf_gap():
    t = pd.read_csv(RESULTS / "ext_tfidf_gap.csv")
    cs = t[t.變化.str.startswith("C = ")].assign(C=lambda d: d.變化.str[4:].astype(float))
    ours = t[t.變化.str.startswith("我們的設定")].test_auc.iat[0]
    cs = pd.concat([cs[["C", "test_auc"]], pd.DataFrame(dict(C=[1.0], test_auc=[ours]))]).sort_values("C")
    folds = pd.read_csv(RESULTS / "ext_tfidf_folds.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[3, 2]))
    ax = axes[0]
    ax.plot(cs.C, 100 * cs.test_auc, "o-", c=C_TFIDF, ms=6)
    ax.plot(1.0, 100 * ours, "o", ms=10, mfc="none", mec=INK, mew=1.4)
    ax.annotate(f"論文描述的設定（C = 1）{100 * ours:.2f}", (1.0, 100 * ours), xytext=(12, -32),
                textcoords="offset points", fontsize=9, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.axhline(96.5, c=INK, ls="--", lw=1.2)
    ax.text(0.03, 96.56, "論文 96.5", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([0.03, 0.1, 0.3, 1, 3, 10], ["0.03", "0.1", "0.3", "1", "3", "10"])
    ax.minorticks_off()
    ax.set(xlabel="正則化參數 C（越小越強）", ylabel="測試 AUC（%）")
    ax.set_title("（a）只有正則化很強（C ≤ 0.1）時才會掉到論文的數值")
    ax = axes[1]
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(folds))
    ax.scatter(jitter, 100 * folds.test_auc, s=40, c=C_TFIDF, edgecolors="white", linewidths=1, zorder=3)
    m, sd = 100 * folds.test_auc.mean(), 100 * folds.test_auc.std()
    ax.errorbar(0.35, m, yerr=sd, fmt="o", c=INK, ms=7, capsize=4)
    ax.text(0.45, m, f"平均 {m:.2f} ± {sd:.2f}", va="center", fontsize=9, color=INK)
    ax.axhline(96.5, c=INK, ls="--", lw=1.2)
    ax.text(-0.3, 96.56, "論文 96.5", fontsize=9, color=INK)
    ax.set_xlim(-0.35, 1.0)
    ax.set_xticks([])
    ax.set_ylim(96.2, 98.0)
    ax.set_ylabel("測試 AUC（%）")
    ax.set_title("（b）10 折交叉驗證：每一折都高於論文")
    fig.tight_layout()
    save(fig, "rq1_tfidf_gap.png")


# ---------------------------------------------------------------- 4. 錯誤分析
def fig_error():
    r = pd.read_csv(RESULTS / "ext_error_by_rating.csv")
    ln = pd.read_csv(RESULTS / "ext_error_by_length.csv")
    tg = pd.read_csv(RESULTS / "ext_error_tags.csv").sort_values("有此特徵的錯誤率")
    overall = 100 * (r.錯誤率 * r.則數).sum() / r.則數.sum()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained", gridspec_kw=dict(width_ratios=[2, 2.4, 3]))
    box = dict(fc="white", ec="none", pad=0.6)

    def bars(ax, labels, vals, ns, horizontal=False, note_right=False):
        pos = np.arange(len(vals))
        if horizontal:
            ax.axvline(overall, c=INK, ls="--", lw=1, zorder=1)
            ax.barh(pos, vals, color=C_FT, height=0.62, zorder=2)
            ax.set_yticks(pos, labels, fontsize=9)
            ax.grid(axis="y", visible=False)
            for p, v, n in zip(pos, vals, ns):
                ax.text(v + 0.3, p, f"{v:.1f}%（{n:,} 則）", va="center", fontsize=8.5, color=INK2, bbox=box, zorder=3)
            ax.set_xlim(0, max(vals) * 1.45)
        else:
            ax.axhline(overall, c=INK, ls="--", lw=1, zorder=1)
            ax.bar(pos, vals, color=C_FT, width=0.62, zorder=2)
            ax.set_xticks(pos, labels)
            ax.grid(axis="x", visible=False)
            for p, v, n in zip(pos, vals, ns):
                ax.text(p, v + 0.25, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=INK, bbox=box, zorder=3)
                ax.text(p, -0.5, f"{n:,} 則", ha="center", va="top", fontsize=7.5, color=MUTED)
            ax.text(len(vals) - 0.55 if note_right else -0.45, overall + 0.15, f"整體 {overall:.1f}%",
                    ha="right" if note_right else "left", va="bottom", fontsize=8.5, color=INK, bbox=box, zorder=3)
            ax.set_ylim(-1.6 * max(vals) / 10, max(vals) * 1.2)

    bars(axes[0], [f"{s} 星" for s in r.rating], 100 * r.錯誤率, r.則數, note_right=True)
    axes[0].set(ylabel="錯誤率（%）", title="（a）依星等：2 星與 4 星最難")
    bars(axes[1], [s.replace(" 字", "") for s in ln.長度], 100 * ln.錯誤率, ln.則數)
    axes[1].set(xlabel="文字長度（字數）", title="（b）依文字長度：越長越容易錯")
    tags = [s.replace("（", "\n（") for s in tg.特徵]
    bars(axes[2], tags, 100 * tg.有此特徵的錯誤率, tg.則數, horizontal=True)
    axes[2].set(xlabel="有此特徵的評論中，判錯的比例（%）", title="（c）依文字特徵：轉折詞的錯誤率是平均的 3 倍")
    fig.suptitle("RQ1 錯誤分析：BERTimbau Large 微調（8 個種子平均）在測試集上錯了哪些評論（虛線＝整體錯誤率）",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    save(fig, "rq1_error.png")


# ---------------------------------------------------------------- LLM 結果（含第二輪補做）
V2_LR_3B = 1e-4  # 第二輪依「種子 42 的驗證 AUC」選定的 3B 學習率


def lora_v2():
    """rq2_llm_v2.csv：每個設定依各種子平均的驗證 AUC 挑 epoch，回傳挑到的那個 epoch 的各種子結果。"""
    v = pd.read_csv(RESULTS / "rq2_llm_v2.csv")
    key = ["model", "train_size", "lr", "epochs"]
    m = v.groupby(key + ["epoch"]).val_auc.mean().reset_index()
    best = m.loc[m.groupby(key).val_auc.idxmax(), key + ["epoch"]]
    return v.merge(best, on=key + ["epoch"])


def llm_results():
    """rq2_llm.csv，但 Qwen-3B LoRA 換成第二輪（學習率 1e-4）的結果；第一次用 2e-4，學習率沒調好。"""
    llm = pd.read_csv(RESULTS / "rq2_llm.csv")
    v = lora_v2()
    v3 = v[(v.model == "qwen-3b") & np.isclose(v.lr, V2_LR_3B)]
    llm = llm[~((llm.model == "qwen-3b") & (llm["mode"] == "lora"))]
    return pd.concat([llm, v3[[c for c in llm.columns if c in v3.columns]]], ignore_index=True)


# ---------------------------------------------------------------- 5. 學習曲線
def ft_curve():
    """Base 微調：每個資料量依驗證 AUC 選 epoch（3 個種子平均）。"""
    f = pd.read_csv(RESULTS / "rq2_finetune.csv")
    f = f[(f.model == "bertimbau-base") & (f.epochs == 3)]  # epochs＝8 是第二輪檢查「是否訓練不足」的補做（附錄表 A-10）
    g = f.groupby(["train_size", "epoch"]).agg(val=("val_auc", "mean"), auc=("test_auc", "mean"), sd=("test_auc", "std"))
    best = g.loc[g.groupby(level=0).val.idxmax()].reset_index()
    best[["auc", "sd"]] *= 100
    return best


def fig_learning_curve():
    lc = pd.read_csv(RESULTS / "rq2_learning_curve.csv")
    lc = lc.groupby(["method", "train_size"]).test_auc.agg(["mean", "std"]).reset_index()
    lc[["mean", "std"]] *= 100
    ft = ft_curve()
    llm = llm_results()
    v = lora_v2()  # Qwen-1.5B 的 10,000 筆是第二輪補做的（2 epoch，依驗證集選 epoch）
    llm = pd.concat([llm, v[(v.model == "qwen-1.5b") & (v.train_size == 10000)]], ignore_index=True)
    lora = llm[llm["mode"] == "lora"].groupby(["model", "train_size"]).test_auc.agg(["mean", "std", "count"]).reset_index()
    lora[["mean", "std"]] *= 100
    fin = pd.read_csv(RESULTS / "rq1_finetune.csv")
    large = 100 * fin[(fin.model == "bertimbau-large") & (fin.epoch == fin.epochs)].test_auc.agg(["mean", "std"])
    zs = 100 * llm[(llm.model == "qwen-1.5b") & (llm["mode"] == "zeroshot")].test_auc.iat[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ticks = [1000, 5000, 10000, 30464]

    def line(ax, x, y, sd, c, marker, label, ls="-", dy=0.0, direct=True):
        ax.errorbar(x, y, yerr=sd, fmt=marker + ls, c=c, ms=6.5, capsize=3, mec="white", mew=1, lw=2, label=label)
        if direct:
            ax.text(x.iloc[-1] * 1.12, y.iloc[-1] + dy, label, va="center", fontsize=9, color=INK)

    ax = axes[0]
    spec = [("TF-IDF + LR", C_TFIDF, "o", "TF-IDF", 0), ("mbert 凍結 + LR", C_FROZEN, "^", "m-BERT 凍結", 0),
            ("bertimbau-base 凍結 + LR", C_FROZEN, "s", "Base 凍結", 0), ("bertimbau-large 凍結 + LR", C_FROZEN, "D", "Large 凍結", -0.13)]
    for m, c, mk, lab, dy in spec:
        d = lc[lc.method == m]
        line(ax, d.train_size, d["mean"], d["std"].fillna(0), c, mk, lab, dy=dy)
    line(ax, ft.train_size, ft.auc, ft.sd, C_FT, "o", "Base 微調", dy=0.13)
    ax.set_title("（a）TF-IDF 與 BERT")
    ax.set_ylabel("測試 AUC（%）")

    ax = axes[1]
    line(ax, ft.train_size, ft.auc, ft.sd, C_FT, "o", "BERTimbau Base 微調", direct=False)
    ax.errorbar(30464, large["mean"], yerr=large["std"], fmt="D", c=C_FT, ms=7, capsize=3, mec="white",
                label="BERTimbau Large 微調（只跑全部資料）")
    for m, mk, ls, lab in [("qwen-1.5b", "o", "-", "Qwen-1.5B LoRA"), ("qwen-3b", "s", "--", "Qwen-3B LoRA（1k／5k 為 1 個種子）")]:
        d = lora[lora.model == m].sort_values("train_size")
        line(ax, d.train_size, d["mean"], d["std"].fillna(0), C_LLM, mk, lab, ls=ls, direct=False)
    d = lora[lora.model == "qwen-0.5b"]
    ax.errorbar(d.train_size, d["mean"], yerr=d["std"], fmt="^", c=C_LLM, ms=7, capsize=3, mec="white",
                label="Qwen-0.5B LoRA（只跑全部資料）")
    ax.axhline(zs, c=C_LLM, ls=":", lw=1.5)
    ax.text(1000, zs + 0.03, f"Qwen-1.5B 零樣本（不用訓練資料）{zs:.2f}", fontsize=9, color=INK2)
    ax.set_ylim(96.85, 98.95)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("（b）BERT 微調與 LLM（LoRA）（放大 y 軸）")

    for ax, xmax in zip(axes, [30464 * 3.2, 30464 * 1.6]):
        ax.set_xscale("log")
        ax.set_xticks(ticks, ["1,000", "5,000", "10,000", "30,464"])
        ax.minorticks_off()
        ax.set_xlim(800, xmax)
        ax.set_xlabel("訓練資料筆數（對數尺度）")
    fig.suptitle("RQ2 學習曲線：資料越少，預訓練模型的優勢越大；LLM 用 1,000 筆做 LoRA 就達到 BERT 用 5,000 筆的水準",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, -0.03, "誤差線＝3 個種子的標準差（全部資料的 TF-IDF／凍結模型只跑 1 次）。Base 微調每個資料量依驗證 AUC 選 epoch。"
             "Qwen-1.5B 的 10,000 筆與 Qwen-3B（學習率 1e-4）為第二輪補做，依驗證 AUC 選 epoch。",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "rq2_learning_curve.png")


# ---------------------------------------------------------------- 6. LLM 比較與門檻校正
def fig_llm():
    llm = llm_results()
    cal = pd.read_csv(RESULTS / "ext_llm_calibration.csv")
    fin = pd.read_csv(RESULTS / "rq1_finetune.csv")
    fin = fin[(fin.model == "bertimbau-large") & (fin.epoch == fin.epochs)]
    tf = 100 * pd.read_csv(RESULTS / "ext_tfidf_gap.csv").test_auc.iat[0]
    models = ["qwen-0.5b", "qwen-1.5b", "qwen-3b"]
    names = {"qwen-0.5b": "Qwen-0.5B", "qwen-1.5b": "Qwen-1.5B", "qwen-3b": "Qwen-3B"}
    modes = [("zeroshot", "零樣本", "o", "none", MUTED), ("fewshot4", "4 則範例", "^", MUTED, MUTED),
             ("fewshot8", "8 則範例", "s", INK2, INK2), ("lora", "LoRA 微調（全部資料）", "o", C_LLM, C_LLM)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15]))

    ax = axes[0]
    for i, m in enumerate(models):
        for j, (mode, lab, mk, fc, ec) in enumerate(modes):
            d = llm[(llm.model == m) & (llm["mode"] == mode)]
            if mode == "lora":
                d = d[d.train_size == d.train_size.max()]
            if d.empty:
                continue
            v = 100 * d.test_auc.mean()
            if len(d) > 1:
                ax.errorbar(v, i, xerr=100 * d.test_auc.std(), fmt=mk, ms=8, mfc=fc, mec=ec, mew=1.5, c=ec, capsize=3,
                            label=lab if i == 0 else None, zorder=3)
            else:
                ax.plot(v, i, mk, ms=8, mfc=fc, mec=ec, mew=1.5, label=lab if i == 0 else None, zorder=3)
        ax.plot([100 * llm[(llm.model == m) & (llm["mode"] != "lora")].test_auc.min(),
                 100 * llm[(llm.model == m) & (llm["mode"] == "lora")].test_auc.max()], [i, i], c=GRID, lw=1, zorder=1)
    ref = 100 * fin.test_auc.mean()
    for x, lab, c, ls in [(ref, f"BERTimbau Large 微調 {ref:.2f}", C_FT, "--"), (tf, f"TF-IDF {tf:.2f}", C_TFIDF, ":")]:
        ax.axvline(x, c=c, ls=ls, lw=1.3)
        ax.text(x, 2.6, lab, va="bottom", ha="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(3), [names[m] for m in models])
    ax.set_ylim(-0.5, 2.55)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("測試 AUC（%）")
    ax.set_title("（a）提示工程幫助很小，LoRA 微調才追上 BERT", pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8.5, ncol=4)

    ax = axes[1]
    a = cal[cal.校正方式.str.startswith("不校正")].set_index(["model", "mode"]).test_acc
    b = cal[cal.校正方式.str.startswith("驗證集")].set_index(["model", "mode"]).test_acc
    order = [(m, mode) for m in models for mode in ["zeroshot", "fewshot4", "fewshot8"]]
    labels = [f"{names[m]} {dict(zeroshot='零樣本', fewshot4='4 則範例', fewshot8='8 則範例')[mode]}" for m, mode in order]
    for i, k in enumerate(order):
        x0, x1 = 100 * a[k], 100 * b[k]
        ax.annotate("", (x1, i), (x0, i), arrowprops=dict(arrowstyle="-|>", color=C_LLM, lw=1.6, shrinkA=4, shrinkB=5))
        ax.plot(x0, i, "o", ms=7, mfc="white", mec=MUTED, mew=1.5)
        ax.plot(x1, i, "o", ms=7, c=C_LLM, mec="white")
        ax.text(x1 + 0.25, i, f"+{x1 - x0:.1f}", va="center", fontsize=8.5, color=INK2)
    acc = 100 * fin.test_acc.mean()
    ax.axvline(acc, c=C_FT, ls="--", lw=1.3)
    ax.set_yticks(range(len(order)), labels)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.text(acc, -0.6, f"BERTimbau Large 微調 {acc:.1f}", va="bottom", ha="center", fontsize=8.5, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("測試準確率（%）")
    ax.set_title("（b）在驗證集上調判斷門檻：準確率提高 1–4 個百分點", pad=22)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, mew=1.5, ms=7, label="不校正（門檻 0）"),
               Line2D([], [], marker="o", ls="", c=C_LLM, mec="white", ms=7, label="驗證集最佳門檻")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8.5, ncol=2)
    fig.tight_layout()
    save(fig, "rq2_llm.png")


# ---------------------------------------------------------------- 7. 準確度 vs 速度
def fig_pareto():
    sp = pd.read_csv(RESULTS / "ext_speed.csv").set_index("method").sec_per_1k
    t = forest_table().set_index("method").auc
    llm = llm_results()
    pts = [  # （標籤, 秒／千則, AUC, 顏色, 標記, 實心, 標籤位移）
        ("TF-IDF", sp["TF-IDF ＋ LR"], t["TF-IDF ＋ LR"], C_TFIDF, "o", True, (0, -14)),
        ("m-BERT 凍結", sp["mbert"], t["m-BERT 凍結"], C_FROZEN, "^", True, (8, -4)),
        ("Base 凍結", sp["bertimbau-base"], t["BERTimbau Base 凍結"], C_FROZEN, "s", True, (-8, 0)),
        ("Large 凍結", sp["bertimbau-large"], t["BERTimbau Large 凍結"], C_FROZEN, "D", True, (8, -6)),
        ("m-BERT 微調", sp["mbert"], t["m-BERT 微調"], C_FT, "^", True, (-8, 0)),
        ("Base 微調", sp["bertimbau-base"], t["BERTimbau Base 微調"], C_FT, "s", True, (-10, 6)),
        ("Large 微調", sp["bertimbau-large"], t["BERTimbau Large 微調"], C_FT, "D", True, (-12, 8)),
    ]
    off = {("qwen-0.5b", "zeroshot"): (8, -4), ("qwen-1.5b", "zeroshot"): (-9, 0), ("qwen-3b", "zeroshot"): (8, -6),
           ("qwen-0.5b", "fewshot8"): (8, -4), ("qwen-1.5b", "fewshot8"): (-9, -6), ("qwen-3b", "fewshot8"): (0, 13),
           ("qwen-0.5b", "lora"): (8, -4), ("qwen-1.5b", "lora"): (6, -12), ("qwen-3b", "lora"): (8, 10)}
    nm = {"qwen-0.5b": "0.5B", "qwen-1.5b": "1.5B", "qwen-3b": "3B"}
    md = {"zeroshot": ("零樣本", "o", False), "fewshot8": ("8 則範例", "s", False), "lora": ("LoRA", "o", True)}
    for (m, mode), o in off.items():
        d = llm[(llm.model == m) & (llm["mode"] == mode)]
        if mode == "lora":
            d = d[d.train_size == d.train_size.max()]
        lab, mk, fill = md[mode]
        pts.append((f"Qwen-{nm[m]} {lab}", d.infer_sec_per_1k.mean(), 100 * d.test_auc.mean(), C_LLM, mk, fill, o))

    fig, ax = plt.subplots(figsize=(10, 5.6))
    p = pd.DataFrame(pts, columns=["lab", "x", "y", "c", "mk", "fill", "off"])
    front = []
    for _, r in p.sort_values(["x", "y"], ascending=[True, False]).iterrows():
        if not front or r.y > front[-1][1]:
            front.append((r.x, r.y))
    fx, fy = zip(*front)
    ax.step(list(fx) + [2000], list(fy) + [fy[-1]], where="post", c=MUTED, lw=1.2, ls="--", zorder=1)
    ax.text(1000, fy[-1] + 0.06, "效率前緣", ha="right", fontsize=8.5, color=MUTED)
    for _, r in p.iterrows():
        ax.plot(r.x, r.y, r.mk, ms=8.5, mfc=r.c if r.fill else "white", mec=r.c if not r.fill else "white",
                mew=1.6 if not r.fill else 1, zorder=3)
        ax.annotate(r.lab, (r.x, r.y), xytext=r.off, textcoords="offset points", fontsize=8.5, color=INK,
                    ha="left" if r.off[0] > 0 else "right" if r.off[0] < 0 else "center", va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.008, 2000)
    ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000], ["0.01", "0.1", "1", "10", "100", "1,000"])
    ax.set(xlabel="推論時間：每 1,000 則評論的秒數（對數尺度；TF-IDF 用 CPU，其餘用 RTX 2080 Ti）", ylabel="測試 AUC（%）")
    ax.set_title("RQ2：準確度與速度的取捨——BERT Base 微調只要 Qwen-1.5B LoRA 的 1/26 時間，AUC 只低 0.3 個百分點")
    handles = [Line2D([], [], marker="o", ls="", c=c, mec="white", ms=8, label=l) for l, c in
               [("TF-IDF", C_TFIDF), ("BERT 凍結＋LR", C_FROZEN), ("BERT 微調", C_FT), ("LLM（實心＝LoRA、空心＝提示）", C_LLM)]]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5)
    save(fig, "rq2_pareto.png")


FIGS = dict(forest=fig_forest, sweep=fig_sweep, tfidf_gap=fig_tfidf_gap, error=fig_error,
            learning_curve=fig_learning_curve, llm=fig_llm, pareto=fig_pareto)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="逗號分隔：" + ",".join(FIGS))
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(FIGS)
    for n in names:
        FIGS[n]()
    log(f"完成；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
