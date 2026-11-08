"""加深 7：成本不確定時，預警名單該多長？（蒙地卡羅模擬）

RQ4 的成本（聯繫 R$10、負評 R$60、成功率 30%）只是假設。這裡讓三個數字都隨機變動 10,000 次：
  聯繫成本 ~ 均勻分布 R$5–20；負評損失 ~ 三角分布（R$30、最可能 R$60、R$150）；成功率 ~ Beta(3, 7)（平均 30%）
每一次都用驗證集選出最佳名單長度、再套到測試集；另外比較幾種固定政策（聯繫風險最高的 1%／2%／5%／10%）。
回報：淨效益的中位數與 90% 區間、淨效益為正的機率。

輸出：results/ext_montecarlo*.csv；figures/ext_montecarlo.png
用法：python src/ext_montecarlo.py
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402
from monitor import LOG_PATH, Bar, log, log_text  # noqa: E402
import rq3_data  # noqa: E402
import rq4_decision as R4  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
N = 10000
POLICIES = [0.01, 0.02, 0.05, 0.10]


def main():
    rng = np.random.default_rng(42)
    contact = rng.uniform(5, 20, N)
    neg = rng.triangular(30, 60, 150, N)
    succ = rng.beta(3, 7, N)
    ratio = neg * succ / contact  # 每則負評的期望可挽回價值 ÷ 聯繫成本

    d = R4.add_features(rq3_data.load())
    tr, va, te = rq3_data.split(d, "time")
    rows, dist = [], {}
    for name, num in Bar(list(R4.MOMENTS.items()), desc="預警時點", unit="個"):
        _, pva, pte, _ = R4.fit_moment(num, tr, va, te)
        tpv = np.cumsum(va.neg.values[np.argsort(-pva)])
        tpt = np.cumsum(te.neg.values[np.argsort(-pte)])
        nv, nt = np.arange(1, len(tpv) + 1), np.arange(1, len(tpt) + 1)
        best = np.empty(N)
        for i in range(N):  # 以聯繫成本為 1 單位：淨效益 = ratio × 抓到的負評 − 聯繫人數
            g = ratio[i] * tpv - nv
            j = int(np.argmax(g))
            frac = (j + 1) / len(tpv) if g[j] > 0 else 0.0
            k = int(round(frac * len(tpt)))
            best[i] = (ratio[i] * tpt[k - 1] - k) * contact[i] / len(tpt) * 1000 if k > 0 else 0.0
        dist[(name, "依成本調整名單")] = best
        for f in POLICIES:
            k = int(round(f * len(tpt)))
            dist[(name, f"固定聯繫前 {f:.0%}")] = (ratio * tpt[k - 1] - k) * contact / len(tpt) * 1000
    for (name, pol), v in dist.items():
        rows.append(dict(時點=name, 政策=pol, 淨效益中位數_每千筆=round(np.median(v), 1),
                         P5=round(np.percentile(v, 5), 1), P95=round(np.percentile(v, 95), 1),
                         淨效益為正的機率=round((v > 0).mean(), 3), 平均=round(v.mean(), 1)))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ext_montecarlo.csv", index=False)
    pd.DataFrame(dict(聯繫成本=contact, 負評損失=neg, 成功率=succ, 價值成本比=ratio)).describe().round(3).to_csv(
        RESULTS / "ext_montecarlo_inputs.csv")
    log(f"價值成本比的分布：中位數 {np.median(ratio):.2f}，90% 區間 {np.percentile(ratio, 5):.2f}–{np.percentile(ratio, 95):.2f}，"
        f"> 1 的機率 {(ratio > 1).mean():.1%}")
    log_text("\n各政策的淨效益（R$／每 1,000 筆訂單；成本隨機變動 10,000 次）：\n" + out.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, name in zip(axes, R4.MOMENTS):
        for pol, col in [("依成本調整名單", "#3b6ea5"), ("固定聯繫前 5%", "#d9822b"), ("固定聯繫前 10%", "#b03a2e")]:
            v = dist[(name, pol)]
            ax.hist(np.clip(v, -400, 400), bins=60, alpha=0.5, color=col, label=pol)
        ax.axvline(0, c="k", lw=1)
        ax.set(title=name, xlabel="淨效益（R\\$／每千筆，截斷在 ±400）")
    axes[0].set_ylabel("模擬次數")
    axes[0].legend(fontsize=8)
    fig.suptitle("成本不確定下的淨效益分布：依成本調整名單長度，幾乎不會虧錢")
    fig.tight_layout()
    fig.savefig(FIGURES / "ext_montecarlo.png", dpi=200)
    plt.close(fig)
    log(f"結果已存到 results/ext_montecarlo*.csv、figures/ext_montecarlo.png；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
