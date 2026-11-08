"""加深 10：各方法的推論速度與資源（給「效能 vs 成本」的圖與企業選型建議用）

在同一台電腦（RTX 2080 Ti、i7-9700KF）上，對 Olist 測試集量測：
  - TF-IDF ＋ LR（CPU）：訓練時間、每 1,000 則的預測時間
  - BERT（GPU、fp16、batch 256、max_len 60）：每 1,000 則的推論時間、顯示卡記憶體峰值、參數量
  - Qwen2.5 0.5B／1.5B／3B（GPU、fp16、batch 32）：同上（只量前 1,000 則）
每項先暖機一次再計時。凍結 BERT 的推論成本 ≈ 同模型的 BERT 推論（再加一個很小的 LR）。
輸出：results/ext_speed.csv
用法：python src/ext_speed.py
"""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import bert_utils as BU
from common import RESULTS, load_olist
from monitor import LOG_PATH, log, log_text
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import rq2_llm as L


def timed(fn, n_rep=3):
    fn()  # 暖機
    torch.cuda.synchronize()
    t = []
    for _ in range(n_rep):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        t.append(time.time() - t0)
    return float(np.median(t))


def main():
    train, val, test = load_olist()
    trval = pd.concat([train, val])
    rows = []

    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=6, max_features=10000)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(vec.fit_transform(trval.text_tokens), trval.label)
    fit_s = time.time() - t0
    t0 = time.time()
    for _ in range(5):
        clf.predict_proba(vec.transform(test.text_tokens))
    pred_s = (time.time() - t0) / 5
    rows.append(dict(method="TF-IDF ＋ LR", device="CPU", params_M=round(clf.coef_.size / 1e6, 3),
                     train_seconds=round(fit_s, 1), sec_per_1k=round(pred_s / len(test) * 1000, 4), peak_vram_GB=0))
    log(f"TF-IDF：訓練 {fit_s:.1f} 秒，每千則 {rows[-1]['sec_per_1k']:.4f} 秒")

    for name in ["bertimbau-base", "bertimbau-large", "mbert"]:
        tok = AutoTokenizer.from_pretrained(BU.MODELS[name])
        model = AutoModelForSequenceClassification.from_pretrained(BU.MODELS[name], num_labels=2).cuda().eval()
        enc = BU.encode(tok, test.review_text, 60)
        torch.cuda.reset_peak_memory_stats()
        s = timed(lambda: BU.predict(model, enc))
        rows.append(dict(method=name, device="GPU", params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                         sec_per_1k=round(s / len(test) * 1000, 3), peak_vram_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2)))
        log(f"{name}：每千則 {rows[-1]['sec_per_1k']:.3f} 秒，VRAM 峰值 {rows[-1]['peak_vram_GB']} GB")
        BU.free(model)

    texts = test.review_text.tolist()[:1000]
    for name in ["qwen-0.5b", "qwen-1.5b", "qwen-3b"]:
        try:
            tok = AutoTokenizer.from_pretrained(L.MODELS[name])
        except OSError:
            log(f"{name} 尚未下載，略過")
            continue
        tok.padding_side = "left"
        ids = [tok.encode(w, add_special_tokens=False)[0] for w in L.LABELS]
        model = AutoModelForCausalLM.from_pretrained(L.MODELS[name], dtype=torch.float16).cuda()
        torch.cuda.reset_peak_memory_stats()
        s = timed(lambda: L.score(model, tok, ids, texts, desc=f"{name} 計時"), n_rep=2)
        rows.append(dict(method=name, device="GPU", params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                         sec_per_1k=round(s / len(texts) * 1000, 3), peak_vram_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2)))
        log(f"{name}：每千則 {rows[-1]['sec_per_1k']:.3f} 秒，VRAM 峰值 {rows[-1]['peak_vram_GB']} GB")
        del model
        torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ext_speed.csv", index=False)
    log_text("\n推論速度：\n" + out.to_string(index=False))
    log(f"結果已存到 results/ext_speed.csv；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
