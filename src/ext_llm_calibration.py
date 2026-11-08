"""加深 11：LLM 零樣本／少樣本的門檻校正

零樣本 LLM 的 AUC 不差，但「正面 vs 負面」的分界（logit 差 > 0）不一定對：例如 Qwen-1.5B 只把 63.5% 判成正面
（實際 70%）。這裡用驗證集重新選門檻，再套到測試集，比較準確率與負評 F1。
兩種校正：(a) 讓驗證集準確率最高的門檻；(b) 讓判為正面的比例等於驗證集的實際正面比例。
輸出：results/ext_llm_calibration.csv
用法：python src/ext_llm_calibration.py（要先用 rq2_llm.py --score_val 產生驗證集分數）
"""
import glob
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from common import RESULTS, load_olist
from monitor import LOG_PATH, log, log_text

PRED = RESULTS / "preds"


def metrics(y, s, t):
    return accuracy_score(y, s > t), f1_score(y, s > t, pos_label=0), float((s > t).mean())


def main():
    _, val, test = load_olist()
    yv, yt = val.label.values, test.label.values
    rows = []
    for fv in sorted(glob.glob(str(PRED / "llm_*_val.npy"))):
        ft = fv.replace("_val.npy", ".npy")
        m = re.search(r"llm_(.+?)_(zeroshot|fewshot\d+|lora)_n(\d+)_s(\d+)_val", fv)
        sv, st = np.load(fv), np.load(ft)
        grid = np.quantile(sv, np.linspace(0.01, 0.99, 197))
        t_acc = float(grid[int(np.argmax([accuracy_score(yv, sv > t) for t in grid]))])
        t_rate = float(np.quantile(sv, 1 - yv.mean()))
        for how, t in [("不校正（門檻 0）", 0.0), ("驗證集準確率最高", t_acc), ("正面比例對齊", t_rate)]:
            acc, f1, rate = metrics(yt, st, t)
            rows.append(dict(model=m.group(1), mode=m.group(2), train_size=int(m.group(3)), 校正方式=how, 門檻=round(t, 3),
                             test_auc=round(roc_auc_score(yt, st), 4), test_acc=round(acc, 4), test_f1_neg=round(f1, 4),
                             判為正面比例=round(rate, 4)))
    if not rows:
        log("找不到驗證集分數（llm_*_val.npy），請先用 rq2_llm.py --score_val 跑零樣本／少樣本")
        return
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ext_llm_calibration.csv", index=False)
    log_text(f"\n門檻校正（測試集實際正面比例 {yt.mean():.3f}）：\n" + out.to_string(index=False))
    log(f"結果已存到 results/ext_llm_calibration.csv；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
