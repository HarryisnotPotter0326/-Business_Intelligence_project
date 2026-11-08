"""RQ1：BERT 凍結（不訓練）取向量，再接邏輯斯迴歸。

用法：python rq1_frozen.py [模型代號 ...]，代號見 MODELS。
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CACHE, RESULTS, load_olist, set_seed  # 要在 transformers 之前匯入（設定離線模式）
from transformers import AutoModel, AutoTokenizer
from monitor import LOG_PATH, Bar, log, log_text

MODELS = {
    "bertimbau-base": "neuralmind/bert-base-portuguese-cased",
    "bertimbau-large": "neuralmind/bert-large-portuguese-cased",
    "mbert": "google-bert/bert-base-multilingual-cased",
}
MAX_LEN = 60


@torch.no_grad()
def embed(tok, model, texts, desc, batch=64):
    """回傳最後一層與倒數第 2–4 層的 [CLS]、平均、最大值、標準差。"""
    feats = {k: [] for k in ["cls", "mean", "max", "std", "cls_last4", "mean_last4"]}
    bar = Bar(total=len(texts), desc=desc, unit="則")
    for i in range(0, len(texts), batch):
        bar.update(len(texts[i:i + batch]))
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").to("cuda")
        out = model(**enc, output_hidden_states=True)
        h = out.last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        n = m.sum(1)
        mean = (h * m).sum(1) / n
        std = (((h - mean.unsqueeze(1)) ** 2 * m).sum(1) / n).sqrt()
        mx = h.masked_fill(m == 0, -1e4).max(1).values
        last4 = torch.stack(out.hidden_states[-4:]).float().mean(0)
        feats["cls"].append(h[:, 0])
        feats["mean"].append(mean)
        feats["max"].append(mx)
        feats["std"].append(std)
        feats["cls_last4"].append(last4[:, 0])
        feats["mean_last4"].append((last4 * m).sum(1) / n)
    bar.close()
    return {k: torch.cat(v).cpu().numpy().astype(np.float32) for k, v in feats.items()}


STRATEGIES = {
    "[CLS]": ["cls"],
    "mean": ["mean"],
    "max": ["max"],
    "[CLS]+mean": ["cls", "mean"],
    "mean+max": ["mean", "max"],
    "[CLS]+mean+std": ["cls", "mean", "std"],
    "[CLS]+mean+max": ["cls", "mean", "max"],
    "[CLS] last4": ["cls_last4"],
    "mean last4": ["mean_last4"],
}


C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]


def fit_eval(Xtr, ytr, Xva, yva, Xte, yte):
    """用驗證集選 C，再以訓練＋驗證集重新訓練，回報測試 AUC。"""
    best = None
    for C in C_GRID:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(Xtr, ytr)
        a = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if best is None or a > best[1]:
            best = (C, a)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], max_iter=3000))
    clf.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    return best[0], best[1], roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def save(rows):
    """把這次的結果合併進 results/rq1_frozen.csv（同一模型＋聚合方式會覆蓋舊結果）。"""
    out = pd.DataFrame(rows)
    path = RESULTS / "rq1_frozen.csv"
    if path.exists():
        old = pd.read_csv(path)
        keep = ~old.set_index(["model", "strategy"]).index.isin(out.set_index(["model", "strategy"]).index)
        out = pd.concat([old[keep], out])
    out.to_csv(path, index=False)
    return out, path


def main(names):
    set_seed()
    train, val, test = load_olist()
    splits = {"train": train, "val": val, "test": test}
    rows = []
    for name in names:
        t0 = time.time()
        emb = {}
        tok = model = None
        for s, df in splits.items():
            path = CACHE / f"frozen_{name}_{s}.npz"
            if path.exists():
                emb[s] = dict(np.load(path))
                log(f"{name} {s}: 讀取已存的向量 {path.name}")
            else:
                if model is None:  # 只載入一次
                    log(f"載入模型 {MODELS[name]}（fp16，GPU：{torch.cuda.get_device_name(0)}）")
                    tok = AutoTokenizer.from_pretrained(MODELS[name])
                    model = AutoModel.from_pretrained(MODELS[name], dtype=torch.float16).cuda().eval()
                emb[s] = embed(tok, model, df.review_text.tolist(), desc=f"{name} 取向量 {s}")
                np.savez(path, **emb[s])
        del model
        torch.cuda.empty_cache()
        log(f"{name}: 向量完成，花了 {time.time() - t0:.0f} 秒；開始訓練邏輯斯迴歸")
        for strat, parts in Bar(STRATEGIES.items(), desc=f"{name} 聚合方式", unit="種"):
            X = {s: np.hstack([emb[s][p] for p in parts]) for s in splits}
            C, va, te = fit_eval(X["train"], train.label, X["val"], val.label, X["test"], test.label)
            rows.append(dict(model=name, strategy=strat, dim=X["train"].shape[1], C=C,
                             val_auc=round(va, 4), test_auc=round(te, 4)))
            log(f"{name:16s} {strat:16s} 維度 {rows[-1]['dim']:5d}  C={C:<5}  "
                f"驗證 AUC {va:.4f}  測試 AUC {te:.4f}")
            out, path = save(rows)  # 每完成一種就存，中途中斷也不會遺失
    log(f"結果已存到 {path}")
    log_text(out.sort_values("test_auc", ascending=False).head(10).to_string(index=False))
    log(f"記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
