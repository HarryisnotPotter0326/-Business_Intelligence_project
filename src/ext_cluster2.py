"""加深 6：用「購物經驗」重新分群（RFM 在 Olist 上效果有限：97% 顧客只買過 1 次）

以顧客為單位，取每位顧客訂單的平均值：延遲天數、實際配送天數、運費占比、商品總價（log）、品項數、
分期數、買賣家距離（log）、是否同州、賣家過去負評率。評論結果（評分、負評）不放進分群，只拿來描述各群。
K-means（K = 3–8，silhouette 選），另以高斯混合模型（GMM，BIC）作穩健性比較。

輸出：results/ext_cluster2_*.csv；figures/ext_cluster2.png
用法：python src/ext_cluster2.py [--k 5]
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402
from monitor import LOG_PATH, Bar, log, log_text  # noqa: E402
import rq3_data  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
FEATS = {"delay_days": "延遲天數", "delivery_days": "配送天數", "freight_ratio": "運費占比", "log_price": "商品總價(log)",
         "n_items": "品項數", "installments": "分期數", "log_distance": "距離(log)", "same_state": "同州",
         "seller_prev_neg_rate": "賣家過去負評率"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int)
    a = ap.parse_args()
    d = rq3_data.load()
    d = d.assign(log_price=np.log1p(d.price_total), log_distance=np.log1p(d.distance_km),
                 delay_days=d.delay_days.clip(-40, 60), delivery_days=d.delivery_days.clip(upper=90))
    c = d.groupby("customer_unique_id").agg(**{k: (k, "mean") for k in FEATS},
                                            neg_rate=("neg", "mean"), avg_score=("review_score", "mean"),
                                            undelivered=("delivered", lambda x: 1 - x.mean()),
                                            late_rate=("is_late", "mean"), orders=("order_id", "nunique"),
                                            category=("category", lambda x: x.mode().iat[0] if len(x.mode()) else "unknown"),
                                            state=("customer_state", "first"))
    X = c[list(FEATS)].copy()
    X = X.fillna(X.median())
    Z = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(42)
    sample = rng.choice(len(Z), size=min(20000, len(Z)), replace=False)

    rows = []
    for k in Bar(range(3, 9), desc="選 K", unit="個"):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
        gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42).fit(Z[sample])
        rows.append(dict(k=k, silhouette=round(silhouette_score(Z[sample], km.labels_[sample]), 4),
                         inertia=round(km.inertia_, 0), gmm_bic=round(gm.bic(Z[sample]), 0)))
        log(f"K={k}：silhouette {rows[-1]['silhouette']:.4f}  GMM BIC {rows[-1]['gmm_bic']:,.0f}")
    ks = pd.DataFrame(rows)
    ks.to_csv(RESULTS / "ext_cluster2_k.csv", index=False)
    k = a.k or int(ks.sort_values("silhouette", ascending=False).k.iloc[0])
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42).fit(Z)
    c["cluster"] = km.labels_
    ari = adjusted_rand_score(km.labels_, gm.predict(Z))
    log(f"使用 K = {k}；K-means 與 GMM 分群的一致性（ARI）{ari:.3f}")

    prof = c.groupby("cluster").agg(顧客數=("neg_rate", "size"), **{v: (k_, "mean") for k_, v in FEATS.items()},
                                    負評率=("neg_rate", "mean"), 平均評分=("avg_score", "mean"), 延遲率=("late_rate", "mean"),
                                    未送達率=("undelivered", "mean"),
                                    最常見品類=("category", lambda x: x.value_counts().index[0]),
                                    最常見州=("state", lambda x: x.value_counts().index[0]))
    prof["顧客占比"] = prof.顧客數 / prof.顧客數.sum()
    prof = prof.round(3).sort_values("負評率", ascending=False)
    prof.to_csv(RESULTS / "ext_cluster2_profile.csv")
    log_text("\n各群畫像（依負評率排序）：\n" + prof.T.to_string())
    spread = prof.負評率.max() - prof.負評率.min()
    log(f"各群負評率差距（最高 − 最低）：{100 * spread:.1f} 個百分點（RFM 分群為 7.6 個百分點）")

    # 圖：各群的標準化特徵平均（熱力圖）＋負評率
    zc = pd.DataFrame(Z, columns=list(FEATS.values())).groupby(km.labels_).mean().loc[prof.index]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.6 * k + 2.5), gridspec_kw=dict(width_ratios=[3, 1]))
    im = axes[0].imshow(zc.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    axes[0].set_xticks(range(zc.shape[1]), zc.columns, rotation=35, ha="right")
    axes[0].set_yticks(range(k), [f"群 {i}（{100 * prof.loc[i, '顧客占比']:.0f}%）" for i in zc.index])
    for i in range(zc.shape[0]):
        for j in range(zc.shape[1]):
            axes[0].text(j, i, f"{zc.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    axes[0].set_title("各群特徵（標準化平均；紅＝高於整體）")
    fig.colorbar(im, ax=axes[0], fraction=0.03)
    axes[1].barh(range(k), 100 * prof.負評率.values, color="#b03a2e")
    axes[1].set_yticks(range(k), [""] * k)
    axes[1].set(xlabel="負評率（%）", title="各群負評率")
    axes[1].set_ylim(k - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "ext_cluster2.png", dpi=200)
    plt.close(fig)
    log(f"結果已存到 results/ext_cluster2_*.csv、figures/ext_cluster2.png；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
