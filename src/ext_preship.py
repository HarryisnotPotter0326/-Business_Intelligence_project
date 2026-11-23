"""加深 12（11/20 補做）：出貨前預警的特徵改良。

第一次的出貨前模型（LightGBM 測試 AUC 0.635）有兩個問題：
  1. SHAP 第一名是「下單月份」。資料只有兩年、又依時間切分，月份其實是在記「那段時期的物流好壞」，
     換到新的月份不一定成立。
  2. 缺少最直接的出貨前訊號：賣家與路線過去的延遲紀錄、最近全站與各州的物流狀況、賣家最近的訂單量
     （rq3_data.NUM_HIST）。這些都只用「下單當下已經發生」的事件計算，不會偷看未來。

步驟：
  1. 消融比較（依時間切分，5 種模型）：原本／拿掉月份／加入延遲歷史／兩者都做（preship2）
  2. SHAP（preship2）
  3. RQ4 三個預警時點、成本名單、敏感度（時點 A 的特徵改用 preship2）
  4. 蒙地卡羅（同上）
  5. 每月滾動回測：preship vs preship2

輸出（原本的結果檔都不覆蓋）：
  results/ext_preship_models.csv、ext_preship_warning.csv、ext_preship_backtest.csv、rq3_shap_preship2.csv、
  rq4_warning_v2.csv、rq4_cost_v2.csv、rq4_cost_sensitivity_v2.csv、ext_montecarlo_v2.csv；
  figures/rq3_shap_*_preship2.png、rq4_warning_v2.png、ext_montecarlo_v2.png
用法：python src/ext_preship.py [--quick]（--quick：抽 15,000 筆快速測試，輸出檔名加 _quick）
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import RESULTS
from monitor import LOG_PATH, Bar, log, log_text
import rq3_data
import rq3_explain
import rq3_model
import rq4_decision as R4
import ext_backtest
import ext_montecarlo

T = rq3_data.TIME_COL
NAMES = {"preship": "原本", "preship_nomonth": "拿掉下單月份", "preship_hist": "加入延遲歷史",
         "preship2": "加入延遲歷史＋拿掉下單月份"}
KS = [0.01, 0.05, 0.10]


def ablation(d, quick, q):
    feats = ["preship_nomonth", "preship_hist", "preship2"]
    if quick:
        feats = ["preship"] + feats  # 抽樣資料要重算原本的設定才能比較
    bar = Bar(total=len(feats) * len(rq3_model.candidates(False)), desc="出貨前特徵比較", unit="個")
    rows = []
    for f in feats:
        rows += rq3_model.run_setting(f, "time", "neg", d, False, bar)
    bar.close()
    out = pd.DataFrame(rows)
    if not quick:  # 原本的設定直接沿用 rq3_models.csv（同一支程式、同一份資料算出來的）
        old = pd.read_csv(RESULTS / "rq3_models.csv")
        out = pd.concat([old[(old.features == "preship") & (old.split == "time") & (old.label == "neg")], out])
    _, _, te = rq3_data.split(d, "time")
    y = te.neg.values
    for i, r in out.iterrows():
        p = np.load(rq3_model.PRED / f"rq3_{r.features}_time_neg_{r.model}.npy")
        for k in KS:
            out.loc[i, f"P@{k:.0%}"] = round(R4.at_k(y, p, k)[0], 3)
    out.insert(1, "特徵組", out.features.map(NAMES))
    out.insert(2, "特徵數", [len(sum(rq3_data.FEATURES[f], [])) for f in out.features])
    out.to_csv(RESULTS / f"ext_preship_models{q}.csv", index=False)
    show = out.pivot(index="model", columns="特徵組", values="test_auc")[[NAMES[f] for f in NAMES if NAMES[f] in out.特徵組.values]]
    log_text("\n測試 AUC（依時間切分，出貨前特徵）：\n" + show.to_string())
    log_text("\nLightGBM：\n" + out[out.model == "LightGBM"][["特徵組", "特徵數", "val_auc", "test_auc", "test_ap",
                                                          "P@1%", "P@5%", "P@10%"]].to_string(index=False))
    return out


def backtest(d, q):
    months = pd.period_range("2017-07", "2018-08", freq="M")
    feats = ["preship", "preship2"]
    rows = []
    bar = Bar(total=len(months) * len(feats), desc="滾動回測", unit="次")
    for feat in feats:
        num, cat = rq3_data.FEATURES[feat]
        cols = num + cat
        for m in months:
            tr = d[d[T] < (m - 1).start_time]  # 留 1 個月空檔，和 ext_backtest.py 相同
            te = d[(d[T] >= m.start_time) & (d[T] < (m + 1).start_time)]
            p = ext_backtest.model(num, cat).fit(tr[cols], tr.neg).predict_proba(te[cols])[:, 1]
            rows.append(dict(features=feat, 特徵組=NAMES[feat], month=str(m), n_test=len(te),
                             負評率=round(te.neg.mean(), 4), auc_每月重訓=round(roc_auc_score(te.neg, p), 4)))
            bar.update(1)
    bar.close()
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"ext_preship_backtest{q}.csv", index=False)
    wide = out.pivot(index="month", columns="特徵組", values="auc_每月重訓")
    wide["差距"] = wide[NAMES["preship2"]] - wide[NAMES["preship"]]
    log_text("\n每月滾動回測（每月重訓的測試 AUC）：\n" + wide.round(4).to_string())
    log(f"平均：原本 {wide[NAMES['preship']].mean():.4f}、改良 {wide[NAMES['preship2']].mean():.4f}；"
        f"改良版較好的月份 {int((wide.差距 > 0).sum())}／{len(wide)}")
    return out


def compare(q, sfx):
    """原本 vs 改良版的預警時點與蒙地卡羅結果並排。"""
    rows = []
    for tag, s in [("原本", ""), ("加入延遲歷史", sfx)]:
        w = pd.read_csv(RESULTS / f"rq4_warning{s}.csv")
        c = pd.read_csv(RESULTS / f"rq4_cost{s}.csv")
        w = w.merge(c[["時點", "驗證集選的名單比例", "測試集淨效益_每千筆"]], on="時點")
        w.insert(0, "版本", tag)
        rows.append(w)
    out = pd.concat(rows)
    out.to_csv(RESULTS / f"ext_preship_warning{q}.csv", index=False)
    log_text("\n預警時點：原本 vs 加入延遲歷史\n" + out[["版本", "時點", "test_auc", "test_pr_auc", "P@1%", "P@5%",
                                                    "Lift@5%", "驗證集選的名單比例", "測試集淨效益_每千筆"]].to_string(index=False))
    mc = []
    for tag, s in [("原本", ""), ("加入延遲歷史", sfx)]:
        p = RESULTS / f"ext_montecarlo{s}.csv"
        if p.exists():
            m = pd.read_csv(p)
            m.insert(0, "版本", tag)
            mc.append(m[m.政策 == "依成本調整名單"])
    if mc:
        log_text("\n蒙地卡羅（依成本調整名單）：原本 vs 加入延遲歷史\n" + pd.concat(mc).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    q = "_quick" if a.quick else ""
    sfx = "_v2" + q
    d = rq3_data.load()  # 舊快取沒有延遲歷史欄位時會自動補算
    if a.quick:
        d = d.sample(15000, random_state=42).sort_values(T)
        tmp = Path(tempfile.mkdtemp())  # 快速測試的模型與預測放暫存資料夾，不覆蓋正式結果
        rq3_model.PRED = rq3_model.CACHE = rq3_explain.CACHE = tmp
    log(f"訂單 {len(d):,} 筆；新增的延遲歷史特徵：{'、'.join(rq3_explain.ZH[c] for c in rq3_data.NUM_HIST)}")
    log_text("新特徵的缺值比例：\n" + d[rq3_data.NUM_HIST].isna().mean().round(4).to_string())

    log("步驟 1／5：消融比較")
    ablation(d, a.quick, q)
    log("步驟 2／5：SHAP（preship2）")
    rq3_explain.shap_analysis("preship2", d)
    log("步驟 3／5：RQ4 預警時點與成本名單（preship2）")
    args = argparse.Namespace(contact_cost=10, neg_cost=60, success=0.3)  # 和 RQ4 原本的預設相同
    R4.warning_and_cost(R4.add_features(d), args, moments=R4.moments("preship2"), sfx=sfx)
    log("步驟 4／5：蒙地卡羅（preship2）")
    ext_montecarlo.main(moments=R4.moments("preship2"), sfx=sfx)  # 用全部資料
    log("步驟 5／5：滾動回測")
    backtest(d, q)
    compare(q, sfx)
    log(f"結果已存到 results/ext_preship_*{q}.csv、results/rq4_*{sfx}.csv；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
