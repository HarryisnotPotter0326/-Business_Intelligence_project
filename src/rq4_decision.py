"""RQ4 智慧決策：如何提早攔截負評？該優先改善什麼？（要先跑完 RQ3）

1. 三個預警時點（依時間切分，LightGBM；越晚資訊越多，但能補救的時間越少）
   A 下單時        ：出貨前就知道的特徵
   B 出貨期限日    ：A ＋ 到出貨期限時，賣家是否「還沒交給物流」
   C 預計送達日當天：B ＋ 到預計送達日時是否「還沒送到」
   （只用「在那個時間點就知道」的資訊：例如時點 B 只知道「還沒出貨」，不知道最後會不會出貨）
   指標：AUC、PR-AUC、Precision@K、Recall@K、Lift@K（K = 名單長度占全部訂單比例）
2. 成本敏感門檻：主動聯繫一位顧客的成本 vs 一則負評的損失 × 介入成功率
   在驗證集上選讓淨效益最大的名單長度，再套用到測試集；並做成本比例的敏感度分析
3. 預警名單範例：時點 B 風險最高的訂單，附上 SHAP 前 3 個原因
4. 改善優先順序：賣家／品類／州的「可避免負評數」（延遲訂單若準時，預期可少掉的負評）
   並結合 RQ3 負評主題，判斷每個品類該改善「物流」還是「商品」

輸出：results/rq4_*.csv；figures/rq4_*.png
用法：python src/rq4_decision.py [--contact_cost 10 --neg_cost 60 --success 0.3]
"""
import argparse

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402
from monitor import LOG_PATH, log, log_text  # noqa: E402
import rq3_data  # noqa: E402
from rq3_explain import ZH, group_of  # noqa: E402
from rq3_model import preprocessor  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

NUM_A, CAT = rq3_data.FEATURES["preship"]
MOMENTS = {
    "A 下單時": NUM_A,
    "B 出貨期限日": NUM_A + ["late_handover_at_limit"],
    "C 預計送達日": NUM_A + ["late_handover_at_limit", "late_at_est"],
}
ZH = {**ZH, "late_at_est": "預計送達日仍未送到", "late_handover_at_limit": "出貨期限仍未交物流"}
KS = [0.01, 0.02, 0.05, 0.10, 0.20]

# RQ3 主題（rq3_topics.py，K=9）的人工命名；主題編號不同時會自動略過「品類 × 主題」分析
THEMES = {0: "未收到商品", 1: "未收到商品", 2: "未收到商品", 6: "未收到商品", 3: "少寄／缺件",
          4: "瑕疵／寄錯", 5: "配送延遲", 7: "要求退款／客服", 8: "品質不佳"}
THEME_TYPE = {"未收到商品": "物流", "配送延遲": "物流", "少寄／缺件": "物流／出貨", "瑕疵／寄錯": "商品／出貨",
              "品質不佳": "商品", "要求退款／客服": "客服"}


def add_features(d):
    d = d.copy()
    # 到預計送達日當天還沒送到（含最後沒送達）
    late = (d.order_delivered_customer_date > d.order_estimated_delivery_date) | d.order_delivered_customer_date.isna()
    d["late_at_est"] = late.astype(float)
    # 到出貨期限時還沒交給物流（含最後都沒出貨）
    lh = (d.order_delivered_carrier_date > d.shipping_limit) | d.order_delivered_carrier_date.isna()
    d["late_handover_at_limit"] = lh.astype(float)
    return d


def fit_moment(num, tr, va, te):
    cols = num + CAT
    pre = preprocessor(num, CAT, impute=False).fit(tr[cols])
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=42, verbose=-1)
    m.fit(pre.transform(tr[cols]), tr.neg, eval_X=(pre.transform(va[cols]),), eval_y=(va.neg,),
          eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    it = m.best_iteration_
    pva = m.predict_proba(pre.transform(va[cols]))[:, 1]
    trva = pd.concat([tr, va])
    pipe = Pipeline([("pre", preprocessor(num, CAT, impute=False)),
                     ("model", lgb.LGBMClassifier(n_estimators=it, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                                  subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1))])
    pipe.fit(trva[cols], trva.neg)
    return pipe, pva, pipe.predict_proba(te[cols])[:, 1], it


def at_k(y, p, k):
    n = max(1, int(round(k * len(y))))
    top = np.argsort(-p)[:n]
    prec = y[top].mean()
    return prec, y[top].sum() / y.sum(), prec / y.mean()


def benefit_curve(y, p, a):
    """依風險由高到低聯繫前 n 位顧客的淨效益（每 1,000 筆訂單，R$）。"""
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    n = np.arange(1, len(y) + 1)
    gain = a.success * a.neg_cost * tp - a.contact_cost * n
    return n / len(y), gain / len(y) * 1000


def warning_and_cost(d, a):
    tr, va, te = rq3_data.split(d, "time")
    rows, curves, cost_rows, fitted = [], {}, [], {}
    for name, num in MOMENTS.items():
        pipe, pva, pte, it = fit_moment(num, tr, va, te)
        fitted[name] = (pipe, num, pte)
        y = te.neg.values
        r = dict(時點=name, 特徵數=len(num) + len(CAT), 樹數=it, test_auc=round(roc_auc_score(y, pte), 4),
                 test_pr_auc=round(average_precision_score(y, pte), 4), 測試負評率=round(y.mean(), 4))
        for k in KS:
            prec, rec, lift = at_k(y, pte, k)
            r[f"P@{k:.0%}"], r[f"R@{k:.0%}"], r[f"Lift@{k:.0%}"] = round(prec, 3), round(rec, 3), round(lift, 2)
        rows.append(r)
        # 成本：在驗證集選名單長度（比例），套用到測試集
        fv, gv = benefit_curve(va.neg.values, pva, a)
        best_frac = fv[int(np.argmax(gv))]
        ft, gt = benefit_curve(y, pte, a)
        i = min(int(round(best_frac * len(y))), len(y)) - 1
        cost_rows.append(dict(時點=name, 驗證集選的名單比例=round(best_frac, 4), 測試集淨效益_每千筆=round(gt[i], 1),
                              測試集最佳名單比例=round(ft[int(np.argmax(gt))], 4), 測試集最大淨效益_每千筆=round(gt.max(), 1),
                              全部聯繫的淨效益_每千筆=round(gt[-1], 1)))
        curves[name] = (ft, gt, y, pte)
        log(f"{name}：AUC {r['test_auc']:.4f}  PR-AUC {r['test_pr_auc']:.4f}  P@5% {r['P@5%']:.3f}  "
            f"Lift@5% {r['Lift@5%']:.2f}  淨效益 {gt[i]:.1f} R$／千筆（名單 {best_frac:.1%}）")
    w = pd.DataFrame(rows)
    w.to_csv(RESULTS / "rq4_warning.csv", index=False)
    c = pd.DataFrame(cost_rows)
    c.to_csv(RESULTS / "rq4_cost.csv", index=False)
    log_text("\n預警時點比較（測試集，依時間切分）：\n" + w.to_string(index=False))
    log_text(f"\n成本設定：聯繫成本 R${a.contact_cost}、負評損失 R${a.neg_cost}、介入成功率 {a.success:.0%}\n" + c.to_string(index=False))

    # 圖：Precision@K 與淨效益曲線
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (ft, gt, y, p) in curves.items():
        order = np.argsort(-p)
        prec = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
        m = ft <= 0.3
        axes[0].plot(100 * ft[m], 100 * prec[m], label=name)
        axes[1].plot(100 * ft, gt, label=name)
    axes[0].axhline(100 * y.mean(), ls="--", c="gray", lw=1, label="隨機挑選")
    axes[0].set(xlabel="名單長度（占全部訂單 %）", ylabel="名單中真的是負評的比例（%）", title="預警名單的精準度")
    axes[1].axhline(0, c="gray", lw=1)
    axes[1].set(xlabel="名單長度（占全部訂單 %）", ylabel="淨效益（R\\$／每 1,000 筆訂單）",
                title=f"主動聯繫的淨效益（聯繫 R\\${a.contact_cost:g}，負評 R\\${a.neg_cost:g}，成功率 {a.success:.0%}）")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "rq4_warning.png", dpi=200)
    plt.close(fig)

    # 敏感度：「每則負評的期望可挽回價值 ÷ 聯繫成本」從 0.5 到 10
    sens = []
    for name, (ft, gt, y, p) in curves.items():
        order = np.argsort(-p)
        tp = np.cumsum(y[order])
        n = np.arange(1, len(y) + 1)
        for ratio in [0.5, 1, 2, 3, 5, 7, 10]:
            g = ratio * tp - n  # 以聯繫成本為 1 單位
            j = int(np.argmax(g))
            sens.append(dict(時點=name, 價值成本比=ratio, 最佳名單比例=round((j + 1) / len(y), 4) if g[j] > 0 else 0.0,
                             淨效益_聯繫成本單位_每千筆=round(max(g[j], 0) / len(y) * 1000, 1)))
    s = pd.DataFrame(sens)
    s.to_csv(RESULTS / "rq4_cost_sensitivity.csv", index=False)
    log_text("\n敏感度分析（價值成本比 = 負評損失 × 成功率 ÷ 聯繫成本）：\n" + s.pivot(index="價值成本比", columns="時點", values="最佳名單比例").to_string())
    return te, fitted


def fmt(v):
    return f"{v:.1f}" if isinstance(v, (float, np.floating)) else str(v)


def watchlist(te, fitted, n=20):
    pipe, num, p = fitted["B 出貨期限日"]
    # 第一層：規則型預警。沒有品項紀錄的訂單（多為缺貨 unavailable／取消），下單後立刻就知道
    rule = te.n_items.isna().values
    log(f"規則型預警（缺貨／沒有品項紀錄）：測試集 {rule.sum():,} 筆，負評率 {te.neg.values[rule].mean():.1%}"
        f"（全站 {te.neg.mean():.1%}）→ 建議直接主動退款／聯繫，不需要模型")
    # 第二層：其餘訂單依模型風險排序
    p_model = np.where(rule, -1, p)
    top = np.argsort(-p_model)[:n]
    sub = te.iloc[top]
    p = p_model
    X = pipe["pre"].transform(sub[num + CAT])
    names = pipe["pre"].get_feature_names_out()
    sv = shap.TreeExplainer(pipe["model"]).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    groups = np.array([group_of(x, CAT) for x in names])
    reasons = []
    for i in range(len(sub)):
        s = pd.Series(sv[i], index=groups).groupby(level=0).sum().sort_values(ascending=False)
        reasons.append("、".join(f"{ZH.get(k, k)}（{fmt(sub.iloc[i][k]) if k in sub.columns else ''}）" for k in s.index[:3]))
    out = pd.DataFrame({"order_id": sub.order_id.values, "風險分數": np.round(p[top], 3),
                        "實際評分": sub.review_score.values, "品類": sub.category.values,
                        "顧客州": sub.customer_state.values, "主要原因（SHAP 前 3）": reasons})
    out.to_csv(RESULTS / "rq4_watchlist_sample.csv", index=False)
    log_text("\n預警名單範例（時點 B，排除缺貨訂單後風險最高 20 筆）：\n" + out.drop(columns="order_id").to_string(index=False))
    log(f"名單前 {n} 筆中實際是負評的比例：{(out.實際評分 <= 2).mean():.0%}")


def shrink(neg, n, prior, a=20):
    return (neg + a * prior) / (n + a)


def priorities(d):
    on = d[d.late_at_est == 0]
    base_on = on.neg.mean()
    out = {}
    for key, col, min_n in [("seller", "seller_id", 30), ("category", "category", 200), ("state", "customer_state", 200)]:
        g = d.groupby(col).agg(訂單數=("neg", "size"), 負評數=("neg", "sum"), 延遲或未送達數=("late_at_est", "sum"))
        g["負評率"] = g.負評數 / g.訂單數
        ong = on.groupby(col).neg.agg(["sum", "size"])
        g["準時訂單負評率_收縮"] = shrink(ong["sum"], ong["size"], base_on).reindex(g.index).fillna(base_on)
        late_neg = d[d.late_at_est == 1].groupby(col).neg.sum().reindex(g.index).fillna(0)
        # 可避免負評：延遲訂單的負評數 − 若它們像準時訂單一樣的預期負評數
        g["可避免負評數"] = (late_neg - g.延遲或未送達數 * g.準時訂單負評率_收縮).clip(lower=0)
        # 超額負評：實際負評數 − 以全站平均負評率估計的負評數（收縮後）
        g["負評率_收縮"] = shrink(g.負評數, g.訂單數, d.neg.mean())
        g["超額負評數"] = (g.負評率_收縮 - d.neg.mean()) * g.訂單數
        g = g[g.訂單數 >= min_n].sort_values("可避免負評數", ascending=False)
        out[key] = g.round(3)
        g.reset_index().to_csv(RESULTS / f"rq4_priority_{key}.csv", index=False)
    total_avoid = out["state"].可避免負評數.sum()
    log(f"全部負評 {int(d.neg.sum()):,} 則；若延遲／未送達的訂單都準時，估計約可少 {total_avoid:,.0f} 則（{total_avoid / d.neg.sum():.1%}）")
    s = out["seller"]
    top10 = s.head(10).可避免負評數.sum()
    log(f"賣家共 {len(s):,} 位（≥30 筆）；可避免負評最多的前 10 位賣家合計 {top10:,.0f} 則，"
        f"占所有賣家可避免負評的 {top10 / s.可避免負評數.sum():.1%}")
    log_text("\n改善優先：品類（依可避免負評數）\n" + out["category"].head(12).to_string())
    log_text("\n改善優先：州\n" + out["state"].head(10).to_string())
    log_text("\n改善優先：賣家（前 10）\n" + s.head(10).to_string())

    # 圖：品類的可避免負評 vs 超額負評
    c = out["category"].head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(c.index, c.可避免負評數, color="#d9822b", label="可避免負評（延遲造成）")
    ax.set_xlabel("則數")
    ax.set_title("改善優先順序：品類（延遲／未送達造成的可避免負評）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "rq4_priority_category.png", dpi=200)
    plt.close(fig)
    return out


def category_themes(d):
    p = RESULTS / "rq3_topics.csv"
    t = pd.read_csv(p) if p.exists() else None
    if t is None or len(t) != 9 or not (RESULTS / "rq3_topic_assign.csv").exists():
        log("rq3_topics 的主題數不是 9，略過「品類 × 主題」分析（THEMES 需要重新命名）")
        return
    a = pd.read_csv(RESULTS / "rq3_topic_assign.csv").merge(d[["order_id", "category"]], on="order_id")
    a["主題"] = a.topic.map(THEMES)
    a["類型"] = a.主題.map(THEME_TYPE)
    top = d.category.value_counts().head(15).index
    ct = pd.crosstab(a.category, a.主題, normalize="index").loc[lambda x: x.index.isin(top)].round(3)
    typ = pd.crosstab(a.category, a.類型, normalize="index").loc[lambda x: x.index.isin(top)].round(3)
    typ["建議方向"] = np.where(typ.get("商品", 0) + typ.get("商品／出貨", 0) > 0.30, "商品品質／出貨檢查", "物流配送")
    ct = ct.join(typ[["建議方向"]]).join(a.category.value_counts().rename("有留言負評數"))
    ct.to_csv(RESULTS / "rq4_category_theme.csv")
    overall = a.主題.value_counts(normalize=True).round(3)
    log_text("\n負評主題整體占比：\n" + overall.to_string())
    log_text("\n前 15 大品類的負評主題占比：\n" + ct.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact_cost", type=float, default=10, help="主動聯繫一位顧客的成本（R$，例如客服時間＋折價券）")
    ap.add_argument("--neg_cost", type=float, default=60, help="一則負評的損失（R$，例如流失的未來消費與商譽）")
    ap.add_argument("--success", type=float, default=0.3, help="主動聯繫後，能避免負評的比例")
    a = ap.parse_args()
    d = add_features(rq3_data.load())
    te, fitted = warning_and_cost(d, a)
    watchlist(te, fitted)
    priorities(d)
    category_themes(d)
    log(f"結果已存到 results/rq4_*.csv、figures/rq4_*.png；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
