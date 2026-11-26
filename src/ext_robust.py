"""第五晚（11/26 補強）：統計檢定與樣本選擇的檢查（只用 CPU，約 5 分鐘）

和其他同學的報告比較後，補上四個不足：
1. RQ1 的 21 組成對比較原本沒有做多重比較校正 → 以 Holm 法校正（直接讀 rq1_significance_pairs.csv）。
2. RQ3 五種模型的 AUC 只差約 0.01，原本沒有檢定差距是否顯著 → 用已存的測試集預測，
   對「各模型 vs 最佳模型」做成對 bootstrap（2,000 次），並以 Holm 法校正；出貨前特徵的三種改良也一併檢定。
3. RQ4 三個預警時點的 AUC 與 Precision@K 原本只有點估計 → 重新訓練三個時點的模型（與表 8-1 相同設定），
   用成對 bootstrap 估計 95% 信賴區間，並檢定「越晚預警越準」的差距。
4. 樣本選擇：主題分析與 BERT 交叉評分只用到「有留言」的評論（約 41%）。有留言與沒留言的評論是否不同？
   延遲會不會讓顧客更願意留言？另外也比較有評論與沒有評論的訂單。

輸出：results/ext_robust_rq1_holm.csv、ext_robust_rq3.csv、ext_robust_rq4.csv、ext_robust_rq4_diff.csv、
      ext_robust_selection.csv、ext_robust_comment_rate.csv
用法：python src/ext_robust.py            （全部，約 5 分鐘）
      python src/ext_robust.py --quick    （bootstrap 只做 50 次，快速檢查程式用）
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import RESULTS
from monitor import LOG_PATH, Bar, log, log_text
import rq3_data
import rq4_decision as R4

PRED = RESULTS / "preds"
MODELS = ["LR", "DecisionTree", "RandomForest", "XGBoost", "LightGBM"]
ZH_MODEL = {"LR": "邏輯斯迴歸", "DecisionTree": "決策樹", "RandomForest": "隨機森林", "XGBoost": "XGBoost", "LightGBM": "LightGBM"}
ZH_FEAT = {"preship": "出貨前（原本）", "preship_hist": "加入延遲歷史", "preship_nomonth": "拿掉下單月份",
           "preship2": "加入延遲歷史＋拿掉下單月份"}


def holm(p):
    """Holm 逐步校正，回傳校正後的 p 值（與原順序相同）。"""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def boot_p(diff):
    """雙尾 bootstrap p 值：2 × min(P(Δ* ≤ 0), P(Δ* ≥ 0))，與式 (3-1) 相同。"""
    return float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))


# ---------- 1. RQ1 的多重比較校正 ----------
def rq1_holm():
    p = pd.read_csv(RESULTS / "rq1_significance_pairs.csv")
    p["p_holm"] = holm(p.p_value.values).round(4)
    p["significant_holm"] = p.p_holm < 0.05
    p.to_csv(RESULTS / "ext_robust_rq1_holm.csv", index=False)
    changed = p[p.significant != p.significant_holm]
    log(f"RQ1：21 組成對比較中，未校正顯著 {int(p.significant.sum())} 組，Holm 校正後仍顯著 {int(p.significant_holm.sum())} 組")
    log_text("校正後結論改變的比較：\n" + (changed.to_string(index=False) if len(changed) else "（沒有）"))
    main3 = p[p.A.str.contains("微調") & p.B.str.contains("凍結") & (p.A.str.split().str[0] == p.B.str.split().str[0])
              | p.B.str.contains("微調") & p.A.str.contains("凍結") & (p.A.str.split().str[0] == p.B.str.split().str[0])]
    log_text("同一模型「微調 vs 凍結」的三組比較（論文主要結論）：\n" + main3.to_string(index=False))


# ---------- 2. RQ3 模型差距的成對 bootstrap ----------
def rq3_models(d, B, rng):
    _, _, te = rq3_data.split(d, "time")
    y = te.neg.values
    fams = {"完整特徵（依時間切分）": {ZH_MODEL[m]: f"full_time_neg_{m}" for m in MODELS},
            "出貨前特徵（依時間切分）": {ZH_MODEL[m]: f"preship_time_neg_{m}" for m in MODELS},
            "出貨前特徵的改良（LightGBM）": {ZH_FEAT[f]: f"{f}_time_neg_LightGBM" for f in ZH_FEAT}}
    preds = {}
    for fam, items in fams.items():
        for name, tag in items.items():
            p = np.load(PRED / f"rq3_{tag}.npy")
            assert len(p) == len(y), f"{tag} 的預測筆數 {len(p)} 與測試集 {len(y)} 不同"
            preds[(fam, name)] = p
    keys = list(preds)
    point = {k: roc_auc_score(y, preds[k]) for k in keys}
    boot = {k: np.empty(B) for k in keys}
    for b in Bar(range(B), desc="RQ3 bootstrap", unit="次"):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        for k in keys:
            boot[k][b] = roc_auc_score(yb, preds[k][idx])
    rows = []
    for fam, items in fams.items():
        names = list(items)
        # 前兩組：各模型 vs 測試 AUC 最高的模型；改良組：各改良 vs 原本
        ref = max(names, key=lambda n: point[(fam, n)]) if "改良" not in fam else ZH_FEAT["preship"]
        fam_rows = []
        for n in names:
            if n == ref:
                continue
            diff = boot[(fam, n)] - boot[(fam, ref)]
            fam_rows.append(dict(比較族=fam, 方法=n, 對照=ref, 方法AUC=round(point[(fam, n)], 4), 對照AUC=round(point[(fam, ref)], 4),
                                 差距_百分點=round(100 * (point[(fam, n)] - point[(fam, ref)]), 2),
                                 CI低=round(100 * np.percentile(diff, 2.5), 2), CI高=round(100 * np.percentile(diff, 97.5), 2),
                                 p值=round(boot_p(diff), 4)))
        adj = holm([r["p值"] for r in fam_rows])
        for r, a in zip(fam_rows, adj):
            r["p值_Holm"] = round(float(a), 4)
            r["顯著_Holm"] = bool(a < 0.05)
        rows += fam_rows
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ext_robust_rq3.csv", index=False)
    log_text(f"\nRQ3 模型差距（測試集 {len(y):,} 筆，成對 bootstrap {B:,} 次，族內 Holm 校正）：\n" + out.to_string(index=False))


# ---------- 3. RQ4 預警時點的信賴區間 ----------
def at_k_fast(y, p, k):
    n = max(1, int(round(k * len(y))))
    top = np.argpartition(-p, n - 1)[:n]
    return y[top].mean()


def rq4_moments(d, B, rng):
    d = R4.add_features(d)
    tr, va, te = rq3_data.split(d, "time")
    y = te.neg.values
    preds = {}
    for name, num in R4.moments("preship").items():
        _, _, pte, it = R4.fit_moment(num, tr, va, te)
        preds[name] = pte
        log(f"{name}：重新訓練完成（{it} 棵樹），測試 AUC {roc_auc_score(y, pte):.4f}")
    old = pd.read_csv(RESULTS / "rq4_warning.csv").set_index("時點").test_auc
    for name, p in preds.items():
        gap = abs(roc_auc_score(y, p) - old[name])
        log(f"  與表 8-1（rq4_warning.csv）的 AUC 差距：{gap:.4f}" + ("" if gap < 0.002 else "  ← 注意：與原結果不同"))
    metrics = {"AUC": lambda yy, pp: roc_auc_score(yy, pp),
               "Precision@1%": lambda yy, pp: at_k_fast(yy, pp, 0.01),
               "Precision@5%": lambda yy, pp: at_k_fast(yy, pp, 0.05)}
    names = list(preds)
    boot = {(n, m): np.empty(B) for n in names for m in metrics}
    for b in Bar(range(B), desc="RQ4 bootstrap", unit="次"):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        for n in names:
            pb = preds[n][idx]
            for m, f in metrics.items():
                boot[(n, m)][b] = f(yb, pb)
    rows = []
    for n in names:
        for m, f in metrics.items():
            v = boot[(n, m)]
            rows.append(dict(時點=n, 指標=m, 點估計=round(f(y, preds[n]), 4),
                             CI低=round(np.percentile(v, 2.5), 4), CI高=round(np.percentile(v, 97.5), 4)))
    ci = pd.DataFrame(rows)
    ci.to_csv(RESULTS / "ext_robust_rq4.csv", index=False)
    diffs = []
    for a, b2 in [("B 出貨期限日", "A 下單時"), ("C 預計送達日", "B 出貨期限日"), ("C 預計送達日", "A 下單時")]:
        for m, f in metrics.items():
            dv = boot[(a, m)] - boot[(b2, m)]
            diffs.append(dict(比較=f"{a} − {b2}", 指標=m, 差距=round(f(y, preds[a]) - f(y, preds[b2]), 4),
                              CI低=round(np.percentile(dv, 2.5), 4), CI高=round(np.percentile(dv, 97.5), 4), p值=round(boot_p(dv), 4)))
    dfd = pd.DataFrame(diffs)
    dfd["p值_Holm"] = holm(dfd.p值.values).round(4)
    dfd.to_csv(RESULTS / "ext_robust_rq4_diff.csv", index=False)
    log_text(f"\nRQ4 三個預警時點（測試集 {len(y):,} 筆，bootstrap {B:,} 次）：\n" + ci.to_string(index=False))
    log_text("\n時點之間的差距（9 個比較一起做 Holm 校正）：\n" + dfd.to_string(index=False))


# ---------- 4. 樣本選擇 ----------
def describe(g):
    late = g.is_late.dropna()
    return dict(訂單數=len(g), 負評率=round(g.neg.mean(), 4), 平均星等=round(g.review_score.mean(), 2),
                延遲率_已送達=round(late.mean(), 4) if len(late) else np.nan, 未送達率=round(1 - g.delivered.mean(), 4),
                多賣家比例=round((g.n_sellers > 1).mean(), 4))


def selection(d):
    d = d.copy()
    d["has_text"] = d.review_comment_message.fillna("").str.strip().str.len() > 0
    rows = [dict(群組="全部有評論的訂單", **describe(d)),
            dict(群組="有留言", **describe(d[d.has_text])),
            dict(群組="沒有留言", **describe(d[~d.has_text]))]
    for lo, hi, nm in [(1, 2, "1–2 星"), (3, 3, "3 星"), (4, 5, "4–5 星")]:
        g = d[(d.review_score >= lo) & (d.review_score <= hi)]
        rows += [dict(群組=f"{nm}・有留言", **describe(g[g.has_text])), dict(群組=f"{nm}・沒有留言", **describe(g[~g.has_text]))]
    # 有評論 vs 沒有評論的訂單（原始訂單表）
    o = rq3_data._read("orders", parse_dates=["order_delivered_customer_date", "order_estimated_delivery_date"])
    no_rev = o[~o.order_id.isin(d.order_id)]
    dl = no_rev.order_delivered_customer_date.notna()
    late = (no_rev.order_delivered_customer_date > no_rev.order_estimated_delivery_date)[dl]
    rows.append(dict(群組="沒有評論的訂單（原始訂單表）", 訂單數=len(no_rev), 負評率=np.nan, 平均星等=np.nan,
                     延遲率_已送達=round(late.mean(), 4) if dl.any() else np.nan, 未送達率=round(1 - dl.mean(), 4), 多賣家比例=np.nan))
    sel = pd.DataFrame(rows)
    sel.insert(2, "占全部有評論訂單", (sel.訂單數 / len(d)).round(4))
    sel.loc[sel.群組.str.startswith("沒有評論"), "占全部有評論訂單"] = np.nan
    sel.to_csv(RESULTS / "ext_robust_selection.csv", index=False)
    log(f"原始訂單 {len(o):,} 筆，有評論 {len(d):,} 筆（{len(d) / len(o):.1%}），沒有評論 {len(no_rev):,} 筆")
    log_text("\n樣本選擇：有留言 vs 沒有留言\n" + sel.to_string(index=False))

    # 留言率：各星等、各延遲分組（4–5 星）
    bins = [-np.inf, -10, -5, -2, 0, 2, 5, 10, np.inf]
    lab = ["提早>10", "提早5–10", "提早2–5", "提早0–2", "延遲0–2", "延遲2–5", "延遲5–10", "延遲>10"]
    d["延遲分組"] = pd.cut(d.delay_days, bins, labels=lab)
    r1 = d.groupby("review_score").has_text.agg(留言率="mean", 訂單數="size").reset_index()
    r1.insert(0, "分組方式", "星等")
    r1 = r1.rename(columns={"review_score": "組別"})
    hi = d[d.review_score >= 4]
    r2 = hi.groupby("延遲分組", observed=True).has_text.agg(留言率="mean", 訂單數="size").reset_index()
    r2.insert(0, "分組方式", "4–5 星・延遲天數")
    r2 = r2.rename(columns={"延遲分組": "組別"})
    rate = pd.concat([r1, r2])
    rate["留言率"] = rate.留言率.round(4)
    rate.to_csv(RESULTS / "ext_robust_comment_rate.csv", index=False)
    log_text("\n留言率：\n" + rate.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    B = 50 if a.quick else a.boot
    rng = np.random.default_rng(42)
    d = rq3_data.load()
    log("步驟 1／4：RQ1 成對比較的 Holm 校正")
    rq1_holm()
    log(f"步驟 2／4：RQ3 模型差距的成對 bootstrap（{B:,} 次）")
    rq3_models(d, B, rng)
    log(f"步驟 3／4：RQ4 預警時點的信賴區間（重新訓練 3 個 LightGBM，bootstrap {B:,} 次）")
    rq4_moments(d, B, rng)
    log("步驟 4／4：樣本選擇（有留言 vs 沒有留言）")
    selection(d)
    log(f"結果已存到 results/ext_robust_*.csv；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
