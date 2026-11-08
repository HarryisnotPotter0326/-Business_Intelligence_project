"""BERT 微調與預測的共用函式（給 ext_apply_bert.py、ext_crossdomain.py、ext_speed.py 用）。

訓練設定和 rq1_finetune.py 相同：AdamW（weight decay 0.01）、warm-up 10% 後線性下降、fp16、梯度裁剪 1.0。
"""
import time

import numpy as np
import torch

import common  # noqa: F401  要在 transformers 之前匯入（設定離線模式）
from common import set_seed
from monitor import Bar, log
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODELS = {
    "bertimbau-base": "neuralmind/bert-base-portuguese-cased",
    "bertimbau-large": "neuralmind/bert-large-portuguese-cased",
    "mbert": "google-bert/bert-base-multilingual-cased",
}


def encode(tok, texts, max_len):
    return tok([str(t) for t in texts], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")


@torch.no_grad()
def predict(model, enc, batch=256, desc=None):
    model.eval()
    out = []
    n = len(enc["input_ids"])
    bar = Bar(total=n, desc=desc, unit="則") if desc else None
    for i in range(0, n, batch):
        b = {k: v[i:i + batch].cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**b).logits.float()
        out.append(torch.softmax(logits, -1)[:, 1].cpu())
        if bar:
            bar.update(len(b["input_ids"]))
    if bar:
        bar.close()
    return torch.cat(out).numpy()


def train(name, texts, labels, max_len=60, lr=2.5e-5, dropout=0.1, epochs=1, batch=32, seed=42, desc="BERT"):
    """回傳（模型, tokenizer, 訓練分鐘數）。"""
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODELS[name])
    enc = encode(tok, texts, max_len)
    y = torch.tensor(np.asarray(labels, dtype=int))
    model = AutoModelForSequenceClassification.from_pretrained(MODELS[name], num_labels=2, classifier_dropout=dropout).cuda()
    model.train()  # from_pretrained 載入後是 eval 模式，一定要切回 train 模式（dropout 才會開啟）
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 0.01},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    steps = (len(y) + batch - 1) // batch
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps * epochs), steps * epochs)
    scaler = torch.amp.GradScaler("cuda")
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    log(f"[{desc}] 訓練 {name}：{len(y):,} 筆，{epochs} epoch，max_len {max_len}")
    for ep in range(epochs):
        order = torch.randperm(len(y), generator=g)
        bar = Bar(total=steps, desc=f"{desc} epoch {ep + 1}/{epochs}", unit="步")
        run = None
        for i in range(0, len(y), batch):
            idx = order[i:i + batch]
            b = {k: v[idx].cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**b, labels=y[idx].cuda()).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
            bar.extra = {"loss": f"{run:.4f}"}
            bar.update(1)
        bar.close()
    return model, tok, (time.time() - t0) / 60


def free(model):
    """釋放顯示卡記憶體。呼叫端的變數仍指向模型，所以要把參數和梯度搬回 CPU，光 del 不夠。"""
    import gc
    for p in model.parameters():
        p.grad = None
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
