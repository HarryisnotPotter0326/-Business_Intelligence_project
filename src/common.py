"""共用函式：讀取作者切分好的 Olist 評論資料。"""
import ast
import json
import os
import random

# 模型都已下載：改離線模式，避免背景偷偷下載 safetensors、也不顯示未登入警告
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = ROOT / "cache"
for p in (RESULTS, FIGURES, CACHE):
    p.mkdir(exist_ok=True)

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_olist():
    """回傳 train（折 1–8）、val（折 9）、test（折 10），已排除 3 分評論。"""
    d = pd.read_csv(DATA / "olist.csv")
    d = d[d.kfold_polarity > 0].copy()
    d["review_text"] = d.review_text.fillna("")
    d["review_text_processed"] = d.review_text_processed.fillna("")
    d["text_tokens"] = d.review_text_tokenized.fillna("[]").map(lambda s: " ".join(ast.literal_eval(s)))
    d["label"] = d.polarity.astype(int)
    train = d[d.kfold_polarity <= 8].reset_index(drop=True)
    val = d[d.kfold_polarity == 9].reset_index(drop=True)
    test = d[d.kfold_polarity == 10].reset_index(drop=True)
    return train, val, test


def subsample_idx(train, n, seed):
    """分層抽樣 n 筆訓練資料，回傳列索引（RQ2 學習曲線用；n 為 None 或超過總數時回傳全部）。"""
    if n is None or n >= len(train):
        return train.index.to_numpy()
    return train.groupby("label").sample(frac=n / len(train), random_state=seed).index.to_numpy()


def merge_save(rows, name, key):
    """把這次的結果合併進 results/<name>（key 相同的舊結果會被覆蓋）。"""
    out = pd.DataFrame(rows)
    path = RESULTS / name
    if path.exists():
        old = pd.read_csv(path)
        keep = ~old.set_index(key).index.isin(out.set_index(key).index)
        out = pd.concat([old[keep], out])
    out.to_csv(path, index=False)
    return out, path


def save_json(obj, name):
    path = RESULTS / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
