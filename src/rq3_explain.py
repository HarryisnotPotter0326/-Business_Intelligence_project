"""RQ3：負評的原因是什麼？（要先跑 rq3_model.py，產生 LightGBM 模型）

1. SHAP：LightGBM（full／preship，依時間切分）在測試集上，各特徵對負評機率的貢獻
2. 淺層決策樹（深度 3）：轉成主管看得懂的 if-then 規則
3. 描述統計：各因素分組後的負評率（延遲天數、品類、州、運費比例、賣家歷史負評率…）

輸出：results/rq3_shap_*.csv、rq3_tree_rules.txt、rq3_negrate_*.csv；figures/rq3_*.png
用法：python src/rq3_explain.py
"""
import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree  # noqa: E402

from common import CACHE, FIGURES, RESULTS  # noqa: E402
from monitor import LOG_PATH, log, log_text  # noqa: E402
import rq3_data  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
FIGURES.mkdir(exist_ok=True)

ZH = {
    "n_items": "品項數", "n_products": "商品種類數", "n_sellers": "賣家數", "price_total": "商品總價",
    "freight_total": "運費", "freight_ratio": "運費占比", "price_max": "最高單價", "payment_value": "付款金額",
    "installments": "分期數", "n_payment_types": "付款方式數", "est_days": "預計配送天數", "approve_hours": "付款確認時數",
    "purchase_hour": "下單時刻", "purchase_dow": "下單星期", "purchase_month": "下單月份", "weight_g": "商品重量",
    "volume_cm3": "商品體積", "photos_qty": "商品照片數", "desc_len": "商品描述長度", "name_len": "商品名稱長度",
    "distance_km": "買賣家距離", "same_state": "同州", "seller_prev_n": "賣家過去評論數",
    "seller_prev_neg_rate": "賣家過去負評率", "customer_prev_orders": "顧客過去訂單數",
    "delivered": "已送達", "delivery_days": "實際配送天數", "delay_days": "延遲天數（實際−預計）", "is_late": "是否延遲",
    "carrier_days": "交給物流天數", "seller_late_handover": "賣家逾期出貨",
    "category": "商品品類", "customer_state": "顧客所在州", "seller_state": "賣家所在州", "payment_type": "付款方式",
}


def group_of(name, cats):
    """把 one-hot 欄位名稱（cat__category_xxx）對回原始特徵；補缺值的指示欄（missingindicator_xxx）也併回原特徵。"""
    n = name.split("__", 1)[1]
    for c in cats:
        if n.startswith(c + "_"):
            return c
    return n.replace("missingindicator_", "")


def shap_analysis(feat, d):
    path = CACHE / f"rq3_lgbm_{feat}_time_neg.pkl"
    pipe = joblib.load(path)
    num, cat = rq3_data.FEATURES[feat]
    _, _, te = rq3_data.split(d, "time")
    te = te.sample(min(5000, len(te)), random_state=42)
    X = pipe["pre"].transform(te[num + cat])
    names = pipe["pre"].get_feature_names_out()
    sv = shap.TreeExplainer(pipe["model"]).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    groups = np.array([group_of(n, cat) for n in names])
    imp = (pd.DataFrame(np.abs(sv), columns=groups).T.groupby(level=0).sum().T.mean()
           .sort_values(ascending=False).rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}))
    imp["特徵"] = imp.feature.map(ZH).fillna(imp.feature)
    imp.to_csv(RESULTS / f"rq3_shap_{feat}.csv", index=False)

    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top["特徵"], top.mean_abs_shap, color="#3b6ea5")
    ax.set_xlabel("平均 |SHAP|（對負評機率 log-odds 的影響）")
    ax.set_title(f"負評的重要因素（LightGBM，{'含配送結果' if feat == 'full' else '出貨前特徵'}）")
    fig.tight_layout()
    fig.savefig(FIGURES / f"rq3_shap_bar_{feat}.png", dpi=200)
    plt.close(fig)

    # 數值特徵的 beeswarm（顏色 = 特徵值高低）
    num_idx = [i for i, n in enumerate(names) if n.startswith("num__") and "missingindicator" not in n]
    ex = shap.Explanation(sv[:, num_idx], data=X[:, num_idx],
                          feature_names=[ZH.get(names[i][5:], names[i][5:]) for i in num_idx])
    plt.figure()
    shap.plots.beeswarm(ex, max_display=12, show=False)
    plt.title(f"SHAP 分布（{'含配送結果' if feat == 'full' else '出貨前特徵'}）")
    plt.tight_layout()
    plt.savefig(FIGURES / f"rq3_shap_beeswarm_{feat}.png", dpi=200)
    plt.close("all")
    log(f"SHAP（{feat}）前 8 名：" + "、".join(f"{r['特徵']} {r.mean_abs_shap:.3f}" for _, r in imp.head(8).iterrows()))
    return imp


def tree_rules(d):
    num, _ = rq3_data.FEATURES["full"]
    tr, _, te = rq3_data.split(d, "time")
    X = tr[num].fillna(999)  # 999 代表缺值：未送達視為「無限延遲」、新賣家視為「負評率未知（最高）」
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=300, random_state=42).fit(X, tr.neg)
    names = [ZH.get(c, c) for c in num]
    text = export_text(t, feature_names=names, show_weights=False, decimals=1)
    # 每個葉節點的負評率與筆數
    leaf = t.apply(X)
    stats = pd.DataFrame({"leaf": leaf, "neg": tr.neg.values}).groupby("leaf").neg.agg(["mean", "size"])
    lines = [text, "\n葉節點（訓練資料）：負評率、筆數"]
    lines += [f"  葉 {i}: 負評率 {r['mean']:.1%}，{int(r['size']):,} 筆" for i, r in stats.iterrows()]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(te.neg, t.predict_proba(te[num].fillna(999))[:, 1])
    lines.append(f"\n這棵深度 3 的樹在測試集（依時間切分）的 AUC：{auc:.4f}")
    out = "\n".join(lines)
    (RESULTS / "rq3_tree_rules.txt").write_text(out, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(t, feature_names=names, class_names=["非負評", "負評"], filled=True, impurity=False,
              proportion=True, rounded=True, fontsize=8, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_tree.png", dpi=200)
    plt.close(fig)
    log_text("\n淺層決策樹規則：\n" + out)


def negrate_tables(d):
    out = {}
    bins = [-np.inf, -15, -10, -5, -2, 0, 2, 5, 10, 20, np.inf]
    lab = ["提早>15天", "提早10–15天", "提早5–10天", "提早2–5天", "提早0–2天", "延遲0–2天", "延遲2–5天", "延遲5–10天", "延遲10–20天", "延遲>20天"]
    g = d.assign(延遲分組=pd.cut(d.delay_days, bins, labels=lab, right=True).cat.add_categories("未送達").fillna("未送達"))
    out["delay"] = g.groupby("延遲分組", observed=True).neg.agg(負評率="mean", 訂單數="size").reset_index()
    cat = d.groupby("category").neg.agg(負評率="mean", 訂單數="size")
    out["category"] = cat[cat.訂單數 >= 500].sort_values("負評率", ascending=False).reset_index()
    out["customer_state"] = d.groupby("customer_state").neg.agg(負評率="mean", 訂單數="size").sort_values("負評率", ascending=False).reset_index()
    out["freight_ratio"] = d.groupby(pd.qcut(d.freight_ratio, 10), observed=True).neg.agg(負評率="mean", 訂單數="size").reset_index()
    sb = pd.cut(d.seller_prev_neg_rate, [-0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0])
    sb = sb.cat.add_categories("新賣家（無紀錄）").fillna("新賣家（無紀錄）")
    out["seller_hist"] = d.groupby(sb, observed=True).neg.agg(負評率="mean", 訂單數="size").reset_index()
    out["n_sellers"] = d.groupby(d.n_sellers.clip(upper=3)).neg.agg(負評率="mean", 訂單數="size").reset_index()
    out["handover"] = d.groupby("seller_late_handover", dropna=False).neg.agg(負評率="mean", 訂單數="size").reset_index()
    for k, v in out.items():
        v["負評率"] = v.負評率.round(4)
        v.to_csv(RESULTS / f"rq3_negrate_{k}.csv", index=False)

    t = out["delay"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#3b6ea5"] * 5 + ["#d9822b"] * 5 + ["#b03a2e"]
    ax.bar(t.延遲分組.astype(str), 100 * t.負評率, color=colors[:len(t)])
    for i, (r, n) in enumerate(zip(t.負評率, t.訂單數)):
        ax.text(i, 100 * r + 1, f"{100 * r:.0f}%\n(n={n:,})", ha="center", fontsize=7)
    ax.axhline(100 * d.neg.mean(), ls="--", c="gray", lw=1)
    ax.set_ylabel("負評率（%）")
    ax.set_title("實際送達日 vs 預計送達日：延遲越久，負評率越高")
    plt.xticks(rotation=30, ha="right")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_delay_negrate.png", dpi=200)
    plt.close(fig)
    for k in ["delay", "seller_hist", "n_sellers", "handover"]:
        log_text(f"\n負評率：{k}\n" + out[k].to_string(index=False))
    log_text("\n負評率最高的品類（≥500 筆）：\n" + out["category"].head(10).to_string(index=False))


def main():
    d = rq3_data.load()
    for feat in ["full", "preship"]:
        shap_analysis(feat, d)
    tree_rules(d)
    negrate_tables(d)
    log(f"圖已存到 figures/rq3_*.png，表已存到 results/rq3_*.csv；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
