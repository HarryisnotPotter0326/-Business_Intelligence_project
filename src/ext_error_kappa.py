"""加深 2d（11/20 補做）：錯誤原因歸類的一致性（Cohen's kappa）。

兩種比較：
  - 作者人工歸類 vs AI 第一次歸類（預設）：讀作者填好的 錯誤歸類_盲測.xlsx「歸類」工作表
  - AI 第二次獨立歸類 vs AI 第一次歸類（--ai2，重測信度）：讀 ext_error_recode.RECODE
輸出（{sfx} 是空字串或 _ai2）：
  results/ext_error_kappa{sfx}.csv（一致率、Cohen's kappa）、ext_error_confusion{sfx}.csv（第一次 × 第二位）、
  ext_error_human.csv 或 ext_error_recode.csv（逐則對照）
用法：python src/ext_error_kappa.py [--ai2]
"""
import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import RESULTS
from ext_error_blind import OUT
from ext_error_coding import CODE, NAMES
from monitor import LOG_PATH, log, log_text


def read_human():
    x = pd.read_excel(OUT, sheet_name="歸類")
    col = "你的歸類（1–6）"
    miss = x[x[col].isna()].編號.tolist()
    if miss:
        raise SystemExit(f"還有 {len(miss)} 則沒有歸類，編號：{miss[:20]}{' …' if len(miss) > 20 else ''}")
    bad = x[~x[col].isin(range(1, 7))].編號.tolist()
    if bad:
        raise SystemExit(f"歸類要填 1–6 的整數，有問題的編號：{bad}")
    return x.編號.tolist(), x[col].astype(int).tolist(), x.get("備註")


def main(second="human"):
    if second == "ai2":
        from ext_error_recode import RECODE
        ids, other, note = sorted(RECODE), [RECODE[i] for i in sorted(RECODE)], None
        who, sfx, out_name = "AI 第二次", "_ai2", "ext_error_recode.csv"
    else:
        ids, other, note = read_human()
        who, sfx, out_name = "作者", "", "ext_error_human.csv"
    assert sorted(ids) == sorted(CODE), "編號和第一次歸類對不起來"
    first = [CODE[i][0] for i in ids]
    same = [a == b for a, b in zip(first, other)]
    agree = sum(same) / len(ids)
    kappa = cohen_kappa_score(first, other)
    log(f"{len(ids)} 則：{who} vs AI 第一次，一致 {sum(same)} 則（{agree:.1%}）；Cohen's kappa = {kappa:.3f}")

    lab = [NAMES[k] for k in sorted(NAMES)]
    conf = pd.crosstab(pd.Categorical([NAMES[k] for k in first], lab), pd.Categorical([NAMES[k] for k in other], lab),
                       rownames=["AI 第一次"], colnames=[who], dropna=False)
    conf.to_csv(RESULTS / f"ext_error_confusion{sfx}.csv", encoding="utf-8-sig")
    log_text(f"\n交叉表（列＝AI 第一次、欄＝{who}）：\n" + conf.to_string())

    per = pd.DataFrame({"類別": lab, "AI 第一次則數": [first.count(k) for k in sorted(NAMES)],
                        f"{who}則數": [other.count(k) for k in sorted(NAMES)]})
    # 各類別的一致性：兩次都歸到這類的則數 ÷ 至少一次歸到這類的則數
    per["兩次都歸這類"] = [sum(a == b == k for a, b in zip(first, other)) for k in sorted(NAMES)]
    per["類別一致率"] = (per.兩次都歸這類 / [sum(a == k or b == k for a, b in zip(first, other)) or 1
                                       for k in sorted(NAMES)]).round(3)
    log_text("\n各類別則數：\n" + per.to_string(index=False))
    per.to_csv(RESULTS / f"ext_error_percat{sfx}.csv", index=False, encoding="utf-8-sig")

    # 依錯誤方向分開看（假正評＝1–2 星判正面、假負評＝4–5 星判負面）
    side = ["假正評" if i < 60 else "假負評" for i in ids]
    rows = [dict(比較=f"{who} vs AI 第一次", 範圍="全部", 則數=len(ids), 一致則數=sum(same), 一致率=round(agree, 4),
                 cohen_kappa=round(kappa, 4))]
    for s in ["假正評", "假負評"]:
        f2 = [a for a, t in zip(first, side) if t == s]
        o2 = [b for b, t in zip(other, side) if t == s]
        k2 = cohen_kappa_score(f2, o2)
        n_same = sum(a == b for a, b in zip(f2, o2))
        rows.append(dict(比較=f"{who} vs AI 第一次", 範圍=s, 則數=len(f2), 一致則數=n_same,
                         一致率=round(n_same / len(f2), 4), cohen_kappa=round(k2, 4)))
        log(f"  {s}：一致 {n_same}／{len(f2)}；kappa = {k2:.3f}")
    pd.DataFrame(rows).to_csv(RESULTS / f"ext_error_kappa{sfx}.csv", index=False, encoding="utf-8-sig")

    out = pd.DataFrame({"編號": ids, "AI第一次": [NAMES[k] for k in first], who: [NAMES[k] for k in other],
                        "一致": same})
    if note is not None:
        out["備註"] = note.values
    out.to_csv(RESULTS / out_name, index=False, encoding="utf-8-sig")
    log(f"結果已存到 results/ext_error_kappa{sfx}.csv、ext_error_confusion{sfx}.csv、ext_error_percat{sfx}.csv、"
        f"{out_name}；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai2", action="store_true", help="比較 AI 第二次獨立歸類（ext_error_recode.py）")
    main("ai2" if ap.parse_args().ai2 else "human")
