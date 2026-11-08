"""RQ3：顧客分群（RFM ＋ K-means），並描述每一群的購物與評價行為。

R = 距離資料最後一天的天數（越小越近）、F = 訂單數、M = 總消費金額。
三個值先取 log1p 再標準化，K 從 2 到 8，用 silhouette 分數選（抽樣 2 萬人計算）。
每一群再描述：平均評分、負評率、延遲率、平均運費占比、最常買的品類、所在州。

輸出：results/rq3_cluster_k.csv、rq3_cluster_profile.csv；figures/rq3_cluster_*.png
用法：python src/rq3_cluster.py          （K 自動選）
      python src/rq3_cluster.py --k 4    （指定 K）
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402
from monitor import LOG_PATH, Bar, log, log_text  # noqa: E402
import rq3_data  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def customers(d):
    ref = d[rq3_data.TIME_COL].max() + pd.Timedelta(days=1)
    c = d.groupby("customer_unique_id").agg(
        last=(rq3_data.TIME_COL, "max"), F=("order_id", "nunique"), M=("payment_value", "sum"),
        avg_score=("review_score", "mean"), neg_rate=("neg", "mean"), late_rate=("is_late", "mean"),
        undelivered_rate=("delivered", lambda x: 1 - x.mean()), avg_delay=("delay_days", "mean"),
        freight_ratio=("freight_ratio", "mean"), state=("customer_state", "first"),
        top_category=("category", lambda x: x.mode().iat[0] if len(x.mode()) else "unknown"))
    c["R"] = (ref - c["last"]).dt.days
    return c.dropna(subset=["M"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int)
    a = ap.parse_args()
    d = rq3_data.load()
    c = customers(d)
    log(f"顧客 {len(c):,} 人；只買過 1 次的比例 {(c.F == 1).mean():.1%}")
    X = StandardScaler().fit_transform(np.log1p(c[["R", "F", "M"]]))
    rng = np.random.default_rng(42)
    sample = rng.choice(len(X), size=min(20000, len(X)), replace=False)

    rows = []
    for k in Bar(range(2, 9), desc="K-means 選 K", unit="個"):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        s = silhouette_score(X[sample], km.labels_[sample])
        rows.append(dict(k=k, silhouette=round(s, 4), inertia=round(km.inertia_, 1)))
        log(f"K={k}：silhouette {s:.4f}，inertia {km.inertia_:,.0f}")
    ks = pd.DataFrame(rows)
    ks.to_csv(RESULTS / "rq3_cluster_k.csv", index=False)
    k = a.k or int(ks[ks.k >= 3].sort_values("silhouette", ascending=False).k.iloc[0])  # 至少 3 群，比較好解讀
    log(f"使用 K = {k}")

    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    c["cluster"] = km.labels_
    prof = c.groupby("cluster").agg(
        顧客數=("R", "size"), R_中位數天=("R", "median"), F_平均=("F", "mean"), M_中位數=("M", "median"),
        平均評分=("avg_score", "mean"), 負評率=("neg_rate", "mean"), 延遲率=("late_rate", "mean"),
        未送達率=("undelivered_rate", "mean"), 平均運費占比=("freight_ratio", "mean"),
        最常見品類=("top_category", lambda x: x.value_counts().index[0]),
        最常見州=("state", lambda x: x.value_counts().index[0]))
    prof["顧客占比"] = (prof.顧客數 / prof.顧客數.sum()).round(3)
    prof = prof.round(3).sort_values("M_中位數", ascending=False)
    prof.to_csv(RESULTS / "rq3_cluster_profile.csv")
    log_text("\n各群畫像：\n" + prof.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ks.k, ks.silhouette, "o-")
    axes[0].set(xlabel="K", ylabel="silhouette", title="選擇群數")
    axes[1].plot(ks.k, ks.inertia, "o-", c="#d9822b")
    axes[1].set(xlabel="K", ylabel="inertia", title="手肘法")
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_cluster_k.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    s = c.sample(min(15000, len(c)), random_state=42)
    for cl, g in s.groupby("cluster"):
        ax.scatter(g.R, g.M, s=4, alpha=0.4, label=f"群 {cl}")
    ax.set(xscale="log", yscale="log", xlabel="R：距今天數", ylabel="M：總消費金額（R$）", title="RFM 分群（抽樣 1.5 萬人）")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_cluster_scatter.png", dpi=200)
    plt.close(fig)
    log(f"結果已存到 results/rq3_cluster_*.csv、figures/rq3_cluster_*.png；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
