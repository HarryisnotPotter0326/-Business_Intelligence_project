"""加深 4：依時間滾動回測（模擬模型每個月上線一次）

- 對 2017-07 ～ 2018-08 的每一個月 M：用「M 的前一個月以前」下單的訂單訓練（留 1 個月空檔，
  因為剛下單的訂單還沒有評論），預測 M 月下單的訂單
- 比較兩種做法：每月重新訓練 vs 只在 2017-07 訓練一次之後就不更新
- 特徵組：完整（含配送結果）與出貨前
輸出：results/ext_backtest.csv；figures/ext_backtest.png
用法：python src/ext_backtest.py
"""
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402
from monitor import LOG_PATH, Bar, log, log_text  # noqa: E402
import rq3_data  # noqa: E402
from rq3_model import preprocessor  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
T = rq3_data.TIME_COL


def model(num, cat):
    return Pipeline([("pre", preprocessor(num, cat, impute=False)),
                     ("m", lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                              subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1))])


def main():
    d = rq3_data.load()
    months = pd.period_range("2017-07", "2018-08", freq="M")
    rows = []
    bar = Bar(total=len(months) * 2, desc="滾動回測", unit="次")
    for feat in ["full", "preship"]:
        num, cat = rq3_data.FEATURES[feat]
        cols = num + cat
        static = None
        for m in months:
            start, gap = m.start_time, (m - 1).start_time
            tr, te = d[d[T] < gap], d[(d[T] >= start) & (d[T] < (m + 1).start_time)]
            rolling = model(num, cat).fit(tr[cols], tr.neg)
            if static is None:  # 只在第一個月訓練一次
                static = model(num, cat).fit(tr[cols], tr.neg)
            y = te.neg.values
            pr, ps = rolling.predict_proba(te[cols])[:, 1], static.predict_proba(te[cols])[:, 1]
            rows.append(dict(features=feat, month=str(m), n_train=len(tr), n_test=len(te), 負評率=round(y.mean(), 4),
                             auc_每月重訓=round(roc_auc_score(y, pr), 4), auc_不更新=round(roc_auc_score(y, ps), 4),
                             pr_auc_每月重訓=round(average_precision_score(y, pr), 4),
                             pr_auc_不更新=round(average_precision_score(y, ps), 4)))
            bar.update(1)
    bar.close()
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ext_backtest.csv", index=False)
    summ = out.groupby("features")[["auc_每月重訓", "auc_不更新", "pr_auc_每月重訓", "pr_auc_不更新"]].agg(["mean", "std", "min", "max"]).round(4)
    log_text("\n滾動回測：\n" + out.to_string(index=False))
    log_text("\n摘要：\n" + summ.to_string())

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax2 = ax.twinx()
    f = out[out.features == "full"]
    ax2.bar(f.month, 100 * f.負評率, color="#dddddd", label="當月負評率（右軸）")
    for feat, ls in [("full", "-"), ("preship", "--")]:
        g = out[out.features == feat]
        name = "完整特徵" if feat == "full" else "出貨前特徵"
        ax.plot(g.month, g.auc_每月重訓, "o" + ls, c="#3b6ea5", label=f"{name}：每月重訓")
        ax.plot(g.month, g.auc_不更新, "s" + ls, c="#d9822b", label=f"{name}：不更新")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylabel("測試 AUC")
    ax2.set_ylabel("負評率（%）")
    ax.set_title("依時間滾動回測：每個月的預測表現")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "ext_backtest.png", dpi=200)
    plt.close(fig)
    log(f"結果已存到 results/ext_backtest.csv、figures/ext_backtest.png；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
