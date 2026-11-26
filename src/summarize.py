"""把 results/ 裡的結果整理成 自動結果摘要.md（run_all.py 每跑完一段會自動呼叫）。

也可以手動執行：python src/summarize.py
"""
import json
import time

import pandas as pd

from common import RESULTS, ROOT

OUT = ROOT / "自動結果摘要.md"
STATES = {"第一晚 run_all.py": ROOT / "logs" / "run_all_state.json", "第二晚 run_night2.py": ROOT / "logs" / "night2_state.json",
          "第三晚 run_night3.py（修正 dropout 後重跑）": ROOT / "logs" / "night3_state.json",
          "RQ3 run_rq3.py": ROOT / "logs" / "rq3_state.json",
          "RQ4 run_rq4.py": ROOT / "logs" / "rq4_state.json",
          "加深・白天 run_extra_cpu.py": ROOT / "logs" / "extra_cpu_state.json",
          "加深・晚上 run_extra_gpu.py": ROOT / "logs" / "extra_gpu_state.json",
          "第四晚 run_night4.py（11/20 補做）": ROOT / "logs" / "night4_state.json",
          "第五晚 run_night5.py（11/26 補強）": ROOT / "logs" / "night5_state.json"}
PAPER = {"TF-IDF + LR": "96.5%", "bertimbau-large 凍結": "98.1%", "bertimbau-large 微調": "97.9%"}


def pct(x):
    return "—" if pd.isna(x) else f"{100 * x:.2f}%"


def md(df):
    return df.to_markdown(index=False) if len(df) else "（無資料）"


def read(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def last_epoch(ft):
    return ft.sort_values("epoch").groupby(["model", "train_size", "lr", "dropout", "epochs", "seed"]).tail(1)


def pick_epoch(df, group):
    """每個設定依「各種子平均的驗證 AUC」挑 epoch；回報每個 epoch 的平均，以及挑到的那一列。"""
    g = (df.groupby(group + ["epoch"]).agg(種子數=("seed", "nunique"), train_loss=("train_loss", "mean"),
                                          val_auc=("val_auc", "mean"), test_auc=("test_auc", "mean"),
                                          test_sd=("test_auc", "std")).reset_index())
    return g, g.loc[g.groupby(group).val_auc.idxmax()]


def section_night5():
    lines = ["## 第五晚（11/26 補強）：統計檢定、樣本選擇與兩條主線的連結"]
    for name, title in [("ext_robust_rq1_holm.csv", "RQ1：21 組成對比較的 Holm 校正"),
                        ("ext_robust_rq3.csv", "RQ3：模型 AUC 差距的成對 bootstrap（族內 Holm 校正）"),
                        ("ext_robust_rq4.csv", "RQ4：三個預警時點的 95% 信賴區間"),
                        ("ext_robust_rq4_diff.csv", "RQ4：時點之間的差距"),
                        ("ext_robust_selection.csv", "樣本選擇：有留言 vs 沒有留言"),
                        ("ext_robust_comment_rate.csv", "留言率（各星等；4–5 星依延遲天數）"),
                        ("ext_bridge_aipw.csv", "延遲的因果效果：星等 vs 加入文字（AIPW）"),
                        ("ext_bridge_avoidable.csv", "可避免的不滿：只看星等 vs 加入文字")]:
        t = read(name)
        if t is not None:
            lines += [f"\n### {title}", md(t)]
    if len(lines) == 1:
        lines.append("（尚未執行）")
    return "\n".join(lines) + "\n"


def section_night4():
    lines = ["## 第四晚（11/20 補做）：重跑第一次沒做好的實驗"]
    m = read("ext_preship_models.csv")
    if m is not None:
        lines += ["\n### 出貨前特徵比較：測試 AUC（依時間切分）",
                  md(m.pivot(index="model", columns="特徵組", values="test_auc").reset_index())]
        lines += ["\n### 出貨前特徵比較：LightGBM 明細",
                  md(m[m.model == "LightGBM"][["特徵組", "特徵數", "val_auc", "test_auc", "test_ap", "P@1%", "P@5%", "P@10%"]])]
    w = read("ext_preship_warning.csv")
    if w is not None:
        lines += ["\n### 預警時點：原本 vs 加入延遲歷史",
                  md(w[["版本", "時點", "test_auc", "test_pr_auc", "P@1%", "P@5%", "Lift@5%", "驗證集選的名單比例", "測試集淨效益_每千筆"]])]
    b = read("ext_preship_backtest.csv")
    if b is not None:
        lines += ["\n### 每月滾動回測（每月重訓的測試 AUC）",
                  md(b.groupby("特徵組").auc_每月重訓.agg(["mean", "std", "min", "max"]).round(4).reset_index())]
    mc = read("ext_montecarlo_v2.csv")
    if mc is not None:
        lines += ["\n### 蒙地卡羅（加入延遲歷史）", md(mc)]
    ft = read("rq2_finetune.csv")
    if ft is not None and (ft.epochs == 8).any():
        g, best = pick_epoch(ft[ft.epochs == 8], ["train_size"])
        _, old = pick_epoch(ft[ft.epochs == 3], ["train_size"])
        lines += ["\n### BERT Base 微調 8 epoch：各 epoch 平均", md(g.round(4))]
        cmp = best[["train_size", "epoch", "val_auc", "test_auc", "test_sd"]].merge(
            old[["train_size", "epoch", "val_auc", "test_auc"]], on="train_size", suffixes=("_8ep", "_3ep"))
        lines += ["\n### 依驗證集挑 epoch：8 epoch 排程 vs 原本 3 epoch 排程", md(cmp.round(4))]
    v = read("rq2_llm_v2.csv")
    if v is not None:
        g, best = pick_epoch(v, ["model", "train_size", "lr", "epochs"])
        def keep_lr(x):  # 學習率不要四捨五入（5e-5 會變成 0）
            return x.round(4).assign(lr=x.lr.map(lambda r: f"{r:g}"))
        lines += ["\n### LoRA 逐 epoch 評估：各設定 × epoch 平均", md(keep_lr(g))]
        lines += ["\n### LoRA：依驗證集挑 epoch 後", md(keep_lr(best))]
    if len(lines) == 1:
        lines.append("（尚未執行）")
    return "\n".join(lines) + "\n"


def section_status():
    out = ["## 執行狀態"]
    for title, path in STATES.items():
        if not path.exists():
            continue
        st = json.loads(path.read_text(encoding="utf-8"))
        rows = [dict(階段=k, 說明=v.get("title", ""), 狀態=v["status"], 開始=v.get("start", ""),
                     耗時分鐘=v.get("minutes", ""), 記錄檔=v.get("log", "")) for k, v in st.items()]
        out += [f"### {title}", md(pd.DataFrame(rows))]
    return "\n".join(out) + "\n"


def section_dropout():
    """修正前（第 1 個 epoch 完全沒有 dropout）vs 修正後，論文設定、全部訓練資料。"""
    lines = ["## Dropout 對照：修正 bug 前（第 1 epoch 無 dropout）vs 修正後（論文設定，全部資料）"]
    old_p = RESULTS / "nodropout_bug" / "rq1_finetune.csv"
    new = read("rq1_finetune.csv")
    if not old_p.exists() or new is None:
        return "\n".join(lines) + "\n（修正後的結果尚未產生）\n"
    rows = []
    for tag, d in [("修正前：無 dropout", pd.read_csv(old_p)), ("修正後：有 dropout", new)]:
        d = last_epoch(d[d.train_size == d.train_size.max()])
        for m, g in d.groupby("model"):
            rows.append(dict(model=m, 版本=tag, seeds=len(g), test_auc=pct(g.test_auc.mean()),
                             std=f"±{100 * g.test_auc.std():.2f}" if len(g) > 1 else ""))
    lines.append(md(pd.DataFrame(rows).sort_values(["model", "版本"])))
    return "\n".join(lines) + "\n"


def section_significance():
    lines = ["## RQ1 加深：bootstrap 95% 信賴區間與顯著性（2,000 次成對重抽）"]
    s, p = read("rq1_significance.csv"), read("rq1_significance_pairs.csv")
    if s is None:
        return "\n".join(lines) + "\n（尚未執行）\n"
    s = s.copy()
    s["95% 信賴區間"] = s.apply(lambda r: f"{pct(r.ci_low)} – {pct(r.ci_high)}", axis=1)
    s["test_auc"] = s.test_auc.map(pct)
    lines.append(md(s[["method", "seeds", "test_auc", "seed_std", "95% 信賴區間"]]))
    if p is not None:
        p = p.copy()
        p["差距（百分點）"] = p.apply(lambda r: f"{r.diff_pct:+.2f}（{r.ci_low_pct:+.2f} ~ {r.ci_high_pct:+.2f}）", axis=1)
        p["顯著"] = p.significant.map({True: "是", False: "否"})
        lines += ["\n### 兩兩比較（A − B）", md(p[["A", "B", "差距（百分點）", "p_value", "顯著"]])]
    return "\n".join(lines) + "\n"


def section_sweep():
    lines = ["## RQ1 加深：微調超參數搜尋（每組 3 個種子平均；★ = 驗證 AUC 最佳）"]
    d = read("rq1_sweep.csv")
    if d is None:
        return "\n".join(lines) + "\n（尚未執行）\n"
    d = last_epoch(d)
    g = (d.groupby(["model", "lr", "dropout", "epochs"])
         .agg(seeds=("seed", "count"), val_auc=("val_auc", "mean"), test_auc=("test_auc", "mean"),
              test_std=("test_auc", "std"), minutes=("minutes", "mean")).reset_index())
    g["驗證最佳"] = g.groupby("model").val_auc.transform("max").eq(g.val_auc).map({True: "★", False: ""})
    for c in ["val_auc", "test_auc"]:
        g[c] = g[c].map(pct)
    g["test_std"] = g.test_std.map(lambda x: "" if pd.isna(x) else f"±{100 * x:.2f}")
    g["minutes"] = g.minutes.round(1)
    lines.append(md(g.sort_values(["model", "lr", "dropout", "epochs"])))
    return "\n".join(lines) + "\n"


def section_rq1():
    lines = ["## RQ1 重現：測試集 ROC-AUC 對照論文"]
    rows = []
    tf = read("rq1_tfidf.csv")
    if tf is not None:
        r = tf[(tf.col == "text_tokens") & (tf.ngram == "1-3") & (tf.vocab == 10000) & (tf.min_df == 6)
               & (tf.C == 1.0) & (tf.fit_on == "train+val")]
        if len(r):
            rows.append(dict(方法="TF-IDF + LR", 我們=pct(r.test_auc.iloc[0]), 標準差="", 論文=PAPER["TF-IDF + LR"], 備註="論文設定"))
    fz = read("rq1_frozen.csv")
    if fz is not None:
        for m, g in fz.groupby("model"):
            b = g.loc[g.test_auc.idxmax()]
            rows.append(dict(方法=f"{m} 凍結", 我們=pct(b.test_auc), 標準差="", 論文=PAPER.get(f"{m} 凍結", "—"),
                             備註=f"最佳聚合：{b.strategy}"))
    ft = read("rq1_finetune.csv")
    if ft is not None:
        ft = last_epoch(ft[ft.train_size == ft.train_size.max()])
        for (m, ep), g in ft.groupby(["model", "epochs"]):
            rows.append(dict(方法=f"{m} 微調", 我們=pct(g.test_auc.mean()),
                             標準差=f"±{100 * g.test_auc.std():.2f}" if len(g) > 1 else "",
                             論文=PAPER.get(f"{m} 微調", "—"),
                             備註=f"{len(g)} 個種子，{ep} epoch，每種子約 {g.minutes.mean():.1f} 分"))
    lines.append(md(pd.DataFrame(rows)))
    if ft is not None:
        d = ft[["model", "seed", "epochs", "train_loss", "val_auc", "test_auc", "test_acc", "test_f1_neg", "minutes"]]
        lines += ["\n### 微調各種子明細", md(d.sort_values(["model", "seed"]))]
    return "\n".join(lines) + "\n"


def section_rq2_curve():
    lines = ["## RQ2 學習曲線：訓練資料量 vs 測試 AUC（多個種子平均）"]
    parts = []
    lc = read("rq2_learning_curve.csv")
    if lc is not None:
        parts.append(lc.groupby(["train_size", "method"]).test_auc.mean().unstack())
    ft = read("rq2_finetune.csv")
    if ft is not None:
        ft = ft[ft.epochs == 3]  # 第四晚另跑的 8 epoch 排程列在「第四晚」段落
        for ep, g in ft.groupby("epoch"):
            parts.append(g.groupby(["train_size", "model"]).test_auc.mean().unstack()
                         .add_suffix(f" 微調（第 {ep} epoch）"))
    if not parts:
        return "\n".join(lines) + "\n（尚未執行）\n"
    tab = pd.concat(parts, axis=1).sort_index()
    lines.append(tab.map(pct).reset_index().to_markdown(index=False))
    return "\n".join(lines) + "\n"


def section_rq2_llm():
    lines = ["## RQ2 LLM：零樣本 vs LoRA 微調"]
    d = read("rq2_llm.csv")
    if d is None:
        return "\n".join(lines) + "\n（尚未執行）\n"
    d = d.sort_values(["model", "mode", "train_size"])
    show = d[["model", "mode", "train_size", "test_auc", "test_acc", "test_f1_neg", "pred_pos_rate",
              "trainable_params", "train_minutes", "infer_sec_per_1k", "n_test"]].copy()
    for c in ["test_auc", "test_acc", "test_f1_neg"]:
        show[c] = show[c].map(pct)
    lines.append(md(show))
    return "\n".join(lines) + "\n"


def section_rq3():
    lines = ["## RQ3 資料探勘：用訂單資料預測負評（1–2 分）"]
    m = read("rq3_models.csv")
    if m is None:
        return "\n".join(lines) + "\n（尚未執行）\n"
    show = m[["features", "split", "label", "model", "test_pos_rate", "val_auc", "test_auc", "test_ap",
              "test_precision", "test_recall", "test_f1", "seconds"]].sort_values(
        ["label", "split", "features", "test_auc"], ascending=[True, False, True, False])
    lines += ["> features：full＝含配送結果、preship＝出貨前就知道；split：time＝依時間、random＝隨機；"
              "test_ap＝PR-AUC（隨機猜的基準＝test_pos_rate）", md(show)]
    for feat in ["full", "preship"]:
        s = read(f"rq3_shap_{feat}.csv")
        if s is not None:
            lines += [f"\n### SHAP 重要度前 10 名（LightGBM，{feat}，依時間切分）", md(s.head(10)[["特徵", "mean_abs_shap"]].round(4))]
    for key, title in [("delay", "延遲天數"), ("seller_hist", "賣家過去負評率"), ("n_sellers", "訂單賣家數"),
                       ("category", "品類（≥500 筆，前 10 名）")]:
        t = read(f"rq3_negrate_{key}.csv")
        if t is not None:
            lines += [f"\n### 負評率 × {title}", md(t.head(10))]
    p = RESULTS / "rq3_tree_rules.txt"
    if p.exists():
        lines += ["\n### 淺層決策樹規則（深度 3）", "```", p.read_text(encoding="utf-8"), "```"]
    c = read("rq3_cluster_profile.csv")
    if c is not None:
        lines += ["\n### 顧客分群（RFM ＋ K-means）", md(c)]
    t = read("rq3_topics.csv")
    if t is not None:
        lines += ["\n### 負評文字主題（NMF）", md(t)]
    return "\n".join(lines) + "\n"


def section_rq4():
    lines = ["## RQ4 智慧決策：負評預警與改善優先順序"]
    w = read("rq4_warning.csv")
    if w is None:
        return "\n".join(lines) + "\n（尚未執行）\n"
    lines += ["### 預警時點比較（測試集，依時間切分；P@K＝名單前 K% 中真的是負評的比例）", md(w)]
    for name, title in [("rq4_cost.csv", "成本敏感名單長度（預設：聯繫 R$10、負評 R$60、成功率 30%）"),
                        ("rq4_watchlist_sample.csv", "預警名單範例（時點 B：出貨期限日）"),
                        ("rq4_priority_category.csv", "改善優先：品類（前 10）"),
                        ("rq4_priority_state.csv", "改善優先：州（前 10）"),
                        ("rq4_category_theme.csv", "前 15 大品類的負評主題占比")]:
        t = read(name)
        if t is not None:
            lines += [f"\n### {title}", md(t.head(15) if "category_theme" in name else t.head(10))]
    s = read("rq4_cost_sensitivity.csv")
    if s is not None:
        lines += ["\n### 敏感度：最佳名單比例 × 價值成本比", md(s.pivot(index="價值成本比", columns="時點", values="最佳名單比例").reset_index())]
    return "\n".join(lines) + "\n"


EXTRA = [
    ("ext_tfidf_gap.csv", "加深 1：TF-IDF 與論文的差距", None),
    ("ext_tfidf_folds.csv", "加深 1：TF-IDF 10 折交叉驗證", None),
    ("ext_error_agreement.csv", "加深 2：BERT 判錯的評論，其他模型也判錯幾個", None),
    ("ext_error_by_length.csv", "加深 2：錯誤率 × 文字長度", None),
    ("ext_error_tags.csv", "加深 2：文字特徵與錯誤", None),
    ("ext_error_coding.csv", "加深 2：120 則判錯評論的原因歸類（AI 初分、人工複核）", None),
    ("ext_ensemble.csv", "加深 3：集成", 15),
    ("ext_backtest.csv", "加深 4：依時間滾動回測", None),
    ("ext_causal_rd.csv", "加深 5：斷點迴歸", None),
    ("ext_causal_psm.csv", "加深 5：傾向分數配對", None),
    ("ext_causal_aipw.csv", "加深 5：AIPW（雙重穩健）", None),
    ("ext_causal_avoidable.csv", "加深 5：因果估計的可避免負評", None),
    ("ext_cluster2_profile.csv", "加深 6：購物經驗分群", None),
    ("ext_montecarlo.csv", "加深 7：成本不確定下的淨效益", None),
    ("ext_apply_by_star.csv", "加深 8：各星等的文字情緒", None),
    ("ext_apply_3star.csv", "加深 8：3 分評論的文字傾向", None),
    ("ext_apply_hidden.csv", "加深 8：隱性不滿（4–5 星、文字負面）", None),
    ("ext_apply_delay.csv", "加深 8：延遲 vs 文字情緒", None),
    ("ext_crossdomain.csv", "加深 9：跨資料集", None),
    ("ext_speed.csv", "加深 10：推論速度", None),
    ("ext_llm_calibration.csv", "加深 11：LLM 門檻校正", None),
]


def section_extra():
    lines = ["## 加深實驗"]
    for name, title, n in EXTRA:
        t = read(name)
        if t is None:
            continue
        if name == "ext_crossdomain.csv":
            for m, g in t.groupby("method"):
                lines += [f"\n### {title}：{m}（列＝訓練、欄＝測試）", md(g.pivot(index="train", columns="test", values="auc").reset_index())]
            continue
        lines += [f"\n### {title}", md(t.head(n) if n else t)]
    if len(lines) == 1:
        lines.append("（尚未執行）")
    return "\n".join(lines) + "\n"


def main():
    text = [f"# 自動結果摘要\n\n> 由 `src/summarize.py` 自動產生，最後更新：{time.strftime('%Y-%m-%d %H:%M:%S')}。"
            "\n> 原始數字在 `results/`，完整輸出在 `logs/`。人工整理與觀察寫在 [實驗紀錄.md](實驗紀錄.md)。\n"]
    for f in [section_status, section_night5, section_night4, section_rq1, section_dropout, section_significance, section_sweep, section_rq2_curve, section_rq2_llm, section_rq3, section_rq4, section_extra]:
        try:
            text.append(f())
        except Exception as e:  # 某一段整理失敗不影響其他段
            text.append(f"## {f.__name__}\n整理失敗：{e!r}\n")
    OUT.write_text("\n".join(text), encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(f"已更新 {main()}")
