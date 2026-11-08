"""RQ1／RQ2：BERT 微調（[CLS] + 線性層），照論文的超參數。

論文設定（2201.03382）：Adam with weight decay、slanted triangular LR（warm-up 10% 後線性下降）、
最長 60 tokens；BERTimbau 最佳 LR 2.5e-5、dropout 10%、1 epoch；m-BERT 最佳 LR 2.5e-5、無 dropout、2 epochs。

用法（預設就是論文的最佳設定，3 個隨機種子）：
  python src/rq1_finetune.py bertimbau-base
  python src/rq1_finetune.py mbert
  python src/rq1_finetune.py bertimbau-large
其他參數：--seeds 42 43 44 --epochs 2 --lr 2e-5 --dropout 0.1 --batch 32 --train_size 5000（RQ2 學習曲線）
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from common import RESULTS, load_olist, merge_save, set_seed, subsample_idx  # 要在 transformers 之前匯入（設定離線模式）
from monitor import LOG_PATH, Bar, log, log_text
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

MODELS = {
    "bertimbau-base": "neuralmind/bert-base-portuguese-cased",
    "bertimbau-large": "neuralmind/bert-large-portuguese-cased",
    "mbert": "google-bert/bert-base-multilingual-cased",
}
# 論文的最佳設定
PAPER = {
    "bertimbau-base": dict(lr=2.5e-5, dropout=0.1, epochs=1),
    "bertimbau-large": dict(lr=2.5e-5, dropout=0.1, epochs=1),
    "mbert": dict(lr=2.5e-5, dropout=0.0, epochs=2),
}
MAX_LEN = 60
PRED_DIR = RESULTS / "preds"
PRED_DIR.mkdir(exist_ok=True)


@torch.no_grad()
def predict(model, enc, batch=256):
    model.eval()
    probs = []
    for i in range(0, len(enc["input_ids"]), batch):
        b = {k: v[i:i + batch].cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**b).logits.float()
        probs.append(torch.softmax(logits, -1)[:, 1].cpu())
    model.train()
    return torch.cat(probs).numpy()


def metrics(y, p):
    return dict(auc=roc_auc_score(y, p), acc=accuracy_score(y, p > 0.5), f1_neg=f1_score(y, p > 0.5, pos_label=0))


def run(name, seed, a, train, val, test):
    set_seed(seed)
    tr = train.loc[subsample_idx(train, a.train_size, seed)].reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(MODELS[name])
    enc = {s: tok(df.review_text.tolist(), padding="max_length", truncation=True, max_length=MAX_LEN,
                  return_tensors="pt") for s, df in [("train", tr), ("val", val), ("test", test)]}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[name], num_labels=2, classifier_dropout=a.dropout).cuda()
    model.train()  # from_pretrained 載入後是 eval 模式（dropout 關閉），訓練前一定要切回 train 模式

    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=a.lr)
    steps_per_epoch = (len(tr) + a.batch - 1) // a.batch
    total = steps_per_epoch * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)  # slanted triangular
    scaler = torch.amp.GradScaler("cuda")
    y = torch.tensor(tr.label.values)

    log(f"[{name} seed={seed}] 訓練 {len(tr)} 筆，{a.epochs} epoch × {steps_per_epoch} 步，"
        f"LR {a.lr}，dropout {a.dropout}，batch {a.batch}")
    rows, t0 = [], time.time()
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, a.epochs + 1):
        order = torch.randperm(len(tr), generator=g)
        bar = Bar(total=steps_per_epoch, desc=f"{name} s{seed} epoch {ep}/{a.epochs}", unit="步")
        run_loss = None
        for i in range(0, len(tr), a.batch):
            idx = order[i:i + a.batch]
            b = {k: v[idx].cuda() for k, v in enc["train"].items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**b, labels=y[idx].cuda()).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
            bar.extra = {"loss": f"{run_loss:.4f}", "lr": f"{sched.get_last_lr()[0]:.1e}"}
            bar.update(1)
        bar.close()
        pv, pt = predict(model, enc["val"]), predict(model, enc["test"])
        mv, mt = metrics(val.label, pv), metrics(test.label, pt)
        rows.append(dict(model=name, seed=seed, train_size=len(tr), lr=a.lr, dropout=a.dropout, epochs=a.epochs,
                         epoch=ep, train_loss=round(run_loss, 4), val_auc=round(mv["auc"], 4),
                         test_auc=round(mt["auc"], 4), test_acc=round(mt["acc"], 4),
                         test_f1_neg=round(mt["f1_neg"], 4), minutes=round((time.time() - t0) / 60, 2)))
        np.save(PRED_DIR / f"finetune_{name}_n{len(tr)}_lr{a.lr:g}_do{a.dropout:g}_E{a.epochs}_s{seed}_ep{ep}.npy", pt)
        log(f"[{name} seed={seed}] epoch {ep}: loss {run_loss:.4f}  驗證 AUC {mv['auc']:.4f}  "
            f"測試 AUC {mt['auc']:.4f}  準確率 {mt['acc']:.4f}  負評 F1 {mt['f1_neg']:.4f}  "
            f"累計 {(time.time() - t0) / 60:.1f} 分")
    del model, opt
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", choices=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--lr", type=float)
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train_size", type=int, help="只用部分訓練資料（RQ2 學習曲線）")
    ap.add_argument("--out", default="rq1_finetune.csv")
    args = ap.parse_args()

    log(f"GPU：{torch.cuda.get_device_name(0)}")
    train, val, test = load_olist()
    all_rows = []
    for name in args.models:
        a = argparse.Namespace(**vars(args))
        for k, v in PAPER[name].items():
            if getattr(a, k) is None:
                setattr(a, k, v)
        for seed in args.seeds:
            all_rows += run(name, seed, a, train, val, test)
            out, path = merge_save(all_rows, args.out, ["model", "seed", "train_size", "lr", "dropout", "epochs", "epoch"])  # 每個種子跑完就存
    log(f"結果已存到 {path}")
    last = out.sort_values("epoch").groupby(["model", "train_size", "lr", "dropout", "epochs", "seed"]).tail(1)
    summary = (last.groupby(["model", "train_size", "lr", "dropout", "epochs"])
               .agg(seeds=("seed", "count"), test_auc_mean=("test_auc", "mean"), test_auc_std=("test_auc", "std"),
                    test_acc_mean=("test_acc", "mean"), minutes=("minutes", "mean")).round(4).reset_index())
    log_text(summary.to_string(index=False))
    log(f"記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
