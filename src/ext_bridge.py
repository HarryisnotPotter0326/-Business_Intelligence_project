"""第五晚（11/26 補強）：把兩條主線接起來——用「星等或文字」定義不滿，重估延遲的因果效果（只用 CPU，約 2 分鐘）

7.4 節以星等（1–2 星）為結果，估計延遲使負評率增加約 45 個百分點；7.6 節用 BERT 交叉評分發現，
有些顧客給了 3–5 星，文字卻是負面的（隱性不滿），而且延遲越久越多。但 7.6 節只是描述，沒有控制訂單條件。
本程式把兩者合在一起：結果變數改為
  (1) 星等負評：1–2 星（與 7.4 節相同，作為對照）
  (2) 星等未反映的不滿：3–5 星，但留言經 BERT 判為負面（正面機率 < 0.5）
  (3) 任一不滿：(1) 或 (2)
以同樣的 AIPW（2 折交叉擬合、共變數為全部出貨前特徵）估計各延遲組的 ATT，
並估計「只看星等」會漏掉多少可避免的不滿。
注意：沒有留言的評論無法判斷文字情緒，只能以星等判斷，因此 (2)、(3) 是下限。

輸出：results/ext_bridge_aipw.csv、ext_bridge_avoidable.csv
用法：python src/ext_bridge.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from common import RESULTS
from ext_causal import NUM, CAT, design
from monitor import LOG_PATH, Bar, log, log_text
import rq3_data

OUTCOMES = {"星等負評（1–2 星）": "y_star", "星等未反映的不滿（3–5 星、文字負面）": "y_hidden", "任一不滿（星等或文字）": "y_any"}


def aipw_multi(df, cols):
    """同一組傾向分數、各結果變數各自的結果模型；回傳 {欄位: (ATT, 標準誤)}。"""
    t = df.treat.values
    e = np.zeros(len(df))
    m0 = {c: np.zeros(len(df)) for c in cols}
    for tr, te in KFold(2, shuffle=True, random_state=42).split(df):
        Xtr, pre = design(df.iloc[tr])
        Xte = pre.transform(df.iloc[te][NUM + CAT])
        e[te] = LogisticRegression(C=0.1, max_iter=3000).fit(Xtr, t[tr]).predict_proba(Xte)[:, 1]
        ctrl = t[tr] == 0
        for c in cols:
            y = df[c].values[tr][ctrl]
            m0[c][te] = LogisticRegression(C=0.1, max_iter=3000).fit(Xtr[ctrl], y).predict_proba(Xte)[:, 1]
    e = np.clip(e, 0.01, 0.99)
    p1 = t.mean()
    out = {}
    for c in cols:
        y = df[c].values.astype(float)
        psi = (t * (y - m0[c]) - (1 - t) * e / (1 - e) * (y - m0[c])) / p1
        att = psi.mean()
        se = np.sqrt(((psi - att * t / p1) ** 2).mean() / len(df))
        out[c] = (att, se)
    return out


def main():
    d = rq3_data.load()
    s = pd.read_csv(RESULTS / "ext_apply_scores.csv")[["order_id", "p_pos"]]
    n0 = len(d)
    d = d.merge(s, on="order_id", how="left")
    assert len(d) == n0, "合併 BERT 分數後筆數改變"
    text_neg = (d.p_pos < 0.5).fillna(False).values
    d["y_star"] = d.neg.astype(int)
    d["y_hidden"] = ((d.review_score >= 3) & text_neg).astype(int)
    d["y_any"] = ((d.y_star == 1) | (d.y_hidden == 1)).astype(int)
    has = d.p_pos.notna()
    log(f"訂單 {len(d):,} 筆，其中有留言（有 BERT 分數）{has.sum():,} 筆（{has.mean():.1%}）")
    log(f"星等負評 {d.y_star.sum():,}（{d.y_star.mean():.1%}）；星等未反映的不滿 {d.y_hidden.sum():,}（{d.y_hidden.mean():.1%}）；"
        f"任一不滿 {d.y_any.sum():,}（{d.y_any.mean():.1%}）")

    ontime = d[(d.delivered == 1) & (d.delay_days <= 0)]
    groups = {
        "延遲（任何天數）": d[(d.delivered == 1) & (d.delay_days > 0)],
        "延遲 0–2 天": d[(d.delivered == 1) & (d.delay_days > 0) & (d.delay_days <= 2)],
        "延遲 2–5 天": d[(d.delivered == 1) & (d.delay_days > 2) & (d.delay_days <= 5)],
        "延遲 5–10 天": d[(d.delivered == 1) & (d.delay_days > 5) & (d.delay_days <= 10)],
        "延遲 >10 天": d[(d.delivered == 1) & (d.delay_days > 10)],
        "未送達": d[d.delivered == 0],
    }
    cols = list(OUTCOMES.values())
    rows, att = [], {}
    for name, g in Bar(list(groups.items()), desc="AIPW", unit="組"):
        df = pd.concat([g.assign(treat=1), ontime.assign(treat=0)]).reset_index(drop=True)
        res = aipw_multi(df, cols)
        for lab, c in OUTCOMES.items():
            a, se = res[c]
            att[(name, c)] = a
            naive = g[c].mean() - ontime[c].mean()
            rows.append(dict(組別=name, 處理組訂單數=len(g), 結果變數=lab, 處理組比例=round(g[c].mean(), 4),
                             對照組比例=round(ontime[c].mean(), 4), 單純差距_百分點=round(100 * naive, 2),
                             AIPW_ATT_百分點=round(100 * a, 2), CI低=round(100 * (a - 1.96 * se), 2), CI高=round(100 * (a + 1.96 * se), 2)))
        log(f"{name}：星等 {100 * res['y_star'][0]:+.1f}、星等未反映 {100 * res['y_hidden'][0]:+.1f}、"
            f"任一不滿 {100 * res['y_any'][0]:+.1f} 百分點")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ext_bridge_aipw.csv", index=False)
    log_text("\nAIPW（ATT：延遲訂單若準時送達，比例會降低多少）：\n" + out.to_string(index=False))

    # 可避免的不滿：各延遲組 ATT × 組內訂單數
    av = []
    for name in ["延遲 0–2 天", "延遲 2–5 天", "延遲 5–10 天", "延遲 >10 天", "未送達"]:
        r = dict(組別=name, 訂單數=len(groups[name]))
        for lab, c in OUTCOMES.items():
            r[lab] = round(att[(name, c)] * len(groups[name]), 0)
        av.append(r)
    av = pd.DataFrame(av)
    tot = av.drop(columns=["組別"]).sum()
    av = pd.concat([av, pd.DataFrame([dict(組別="合計", **tot.to_dict())])], ignore_index=True)
    share = {lab: tot[lab] / d[c].sum() for lab, c in OUTCOMES.items()}
    av = pd.concat([av, pd.DataFrame([dict(組別="占該類全部的比例", 訂單數=np.nan, **{k: round(v, 4) for k, v in share.items()})])],
                   ignore_index=True)
    av.to_csv(RESULTS / "ext_bridge_avoidable.csv", index=False)
    log_text("\n可避免的不滿（AIPW 的 ATT × 各組訂單數）：\n" + av.to_string(index=False))
    log(f"只看星等：可避免 {tot['星等負評（1–2 星）']:,.0f} 則；加入文字後：可避免 {tot['任一不滿（星等或文字）']:,.0f} 則，"
        f"其中星等未反映的不滿 {tot['星等未反映的不滿（3–5 星、文字負面）']:,.0f} 則")
    log(f"結果已存到 results/ext_bridge_*.csv；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
