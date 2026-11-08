"""加深 3：把不同模型的預測合起來，會不會更好？（集成）

- 用已存的測試集預測（不重新訓練），每個模型的分數先轉成排名百分位再平均（不同模型的分數尺度不同）
- 權重一律相等（沒有另外的資料可以調權重，避免偷看測試集）
- 與單一最佳模型（BERTimbau Large 微調）做成對 bootstrap（2,000 次）
輸出：results/ext_ensemble.csv
用法：python src/ext_ensemble.py（第二批 GPU 實驗跑完後再跑一次，會自動納入新的 LLM 預測）
"""
import glob
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from common import RESULTS, load_olist
from monitor import LOG_PATH, Bar, log, log_text

PRED = RESULTS / "preds"
B = 2000


def avg(pattern):
    f = sorted(glob.glob(str(PRED / pattern)))
    f = [x for x in f if not x.endswith("_val.npy")]
    return (np.mean([np.load(x) for x in f], axis=0), len(f)) if f else (None, 0)


def collect():
    src = {
        "TF-IDF": "tfidf_paper.npy",
        "Base 凍結": "frozen_bertimbau-base_valbest.npy",
        "Large 凍結": "frozen_bertimbau-large_valbest.npy",
        "Base 微調": "finetune_bertimbau-base_n30464_lr2.5e-05_do0.1_E1_s*_ep1.npy",
        "Large 微調": "finetune_bertimbau-large_n30464_lr2.5e-05_do0.1_E1_s*_ep1.npy",
        "m-BERT 微調": "finetune_mbert_n30464_lr2.5e-05_do0_E2_s*_ep2.npy",
        "Qwen-0.5B LoRA": "llm_qwen-0.5b_lora_n30464_s*.npy",
        "Qwen-1.5B LoRA": "llm_qwen-1.5b_lora_n30464_s*.npy",
        "Qwen-3B LoRA": "llm_qwen-3b_lora_n30464_s*.npy",
        "Qwen-1.5B 零樣本": "llm_qwen-1.5b_zeroshot_n0_s42.npy",
        "Qwen-3B 零樣本": "llm_qwen-3b_zeroshot_n0_s42.npy",
    }
    p, n = {}, {}
    for k, pat in src.items():
        v, c = avg(pat)
        if v is not None:
            p[k], n[k] = rankdata(v) / len(v), c
    return p, n


def main():
    _, _, test = load_olist()
    y = test.label.values
    p, nseed = collect()
    log("可用的模型：" + "、".join(f"{k}（{nseed[k]} 個種子）" for k in p))
    combos = [(k,) for k in p]
    strong = [k for k in ["Large 微調", "Base 微調", "Qwen-1.5B LoRA", "Qwen-3B LoRA", "TF-IDF", "Large 凍結"] if k in p]
    for r in [2, 3]:
        combos += [c for c in combinations(strong, r) if "Large 微調" in c]
    ft = [k for k in ["Base 微調", "Large 微調", "m-BERT 微調"] if k in p]
    combos.append(tuple(ft))
    combos.append(tuple(k for k in p if "零樣本" not in k))
    combos = list(dict.fromkeys(combos))

    rng = np.random.default_rng(42)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    ref = p["Large 微調"]
    ref_boot = np.array([roc_auc_score(y[i], ref[i]) for i in Bar(idx, desc="bootstrap 基準", unit="次")])
    rows = []
    for c in Bar(combos, desc="集成組合", unit="組"):
        s = np.mean([p[k] for k in c], axis=0)
        auc = roc_auc_score(y, s)
        row = dict(組合=" ＋ ".join(c), 模型數=len(c), test_auc=round(auc, 4))
        if len(c) > 1:
            boot = np.array([roc_auc_score(y[i], s[i]) for i in idx])
            dd = boot - ref_boot
            pv = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
            row.update(比Large微調_百分點=round(100 * (auc - roc_auc_score(y, ref)), 3),
                       CI低=round(100 * np.percentile(dd, 2.5), 3), CI高=round(100 * np.percentile(dd, 97.5), 3),
                       p_value=round(min(pv, 1.0), 4))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    out.to_csv(RESULTS / "ext_ensemble.csv", index=False)
    log_text("\n集成結果（依測試 AUC 排序）：\n" + out.to_string(index=False))
    log(f"結果已存到 results/ext_ensemble.csv；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
