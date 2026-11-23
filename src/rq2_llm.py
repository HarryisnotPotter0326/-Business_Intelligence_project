"""RQ2：小型 LLM（Qwen2.5-Instruct）零樣本 vs LoRA 微調。

做法：把評論放進提示詞，請模型回答 positive／negative。
  - 分數 = 模型在答案位置「positive」與「negative」兩個詞的 logit 差，用來算 AUC；大於 0 判為正面。
  - LoRA：只訓練答案那一個 token 的 loss，fp16 混合精度。

用法：
  python src/rq2_llm.py qwen-1.5b --mode zeroshot
  python src/rq2_llm.py qwen-1.5b --mode lora --train_size 5000
  （11/20 補做）每個 epoch 結束都評估驗證集與測試集，之後用驗證集挑 epoch：
  python src/rq2_llm.py qwen-1.5b --mode lora --train_size 1000 --epochs 3 --eval_each_epoch --out rq2_llm_v2.csv
  （11/20 補做）學習率改用「種子 42、指定訓練筆數」中驗證 AUC 最高的那個：
  python src/rq2_llm.py qwen-3b --mode lora --batch 8 --seed 43 --eval_each_epoch --lr_from 30464 --out rq2_llm_v2.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from common import RESULTS, load_olist, merge_save, set_seed, subsample_idx  # 要在 transformers 之前匯入
from monitor import LOG_PATH, Bar, log, log_text
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
}
LABELS = ["negative", "positive"]
MAX_CHARS = 400  # 評論太長時截斷（Olist 評論平均只有 7 個字）
MAX_BAD_RUN = 20  # 連續這麼多步 loss 為 NaN／inf 就判定發散、中止訓練
PRED_DIR = RESULTS / "preds"
PRED_DIR.mkdir(exist_ok=True)


EXAMPLES = []  # 少樣本提示的範例 [(評論, 標籤)]，由 main() 依 --shots 設定


def user_msg(t):
    return {"role": "user", "content": "Classify the sentiment of this Brazilian Portuguese e-commerce review "
                                       "as positive or negative.\n\nReview: " + str(t)[:MAX_CHARS] +
                                       "\n\nAnswer with one word: positive or negative."}


def prompts(tok, texts):
    shots = []
    for t, y in EXAMPLES:  # 少樣本：先放幾組「問題 → 正確答案」的對話
        shots += [user_msg(t), {"role": "assistant", "content": LABELS[y]}]
    out = []
    for t in texts:
        msg = [{"role": "system", "content": "You are a sentiment classifier for customer reviews."}, *shots, user_msg(t)]
        out.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    return out


def pick_examples(train, k, seed):
    """從訓練集挑 k 則範例（正負各半、長度 3–20 個字），正負交錯排列。"""
    n_words = train.review_text.str.split().str.len()
    pool = train[(n_words >= 3) & (n_words <= 20)]
    neg = pool[pool.label == 0].sample(k // 2, random_state=seed)
    pos = pool[pool.label == 1].sample(k - k // 2, random_state=seed)
    ex = []
    for i in range(k):
        src = neg if i % 2 == 0 else pos
        r = src.iloc[i // 2]
        ex.append((r.review_text, int(r.label)))
    return ex


def batch_inputs(tok, texts):
    enc = tok(texts, padding=True, return_tensors="pt", add_special_tokens=False)
    enc["position_ids"] = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return {k: v.cuda() for k, v in enc.items()}


@torch.no_grad()
def score(model, tok, label_ids, texts, desc, batch=32):
    """回傳每則評論的 logit(positive) − logit(negative)。"""
    model.eval()
    ps = prompts(tok, texts)
    order = np.argsort([len(p) for p in ps])  # 依長度排序，減少 padding
    scores = np.zeros(len(ps), dtype=np.float32)
    bar = Bar(total=len(ps), desc=desc, unit="則")
    for i in range(0, len(ps), batch):
        idx = order[i:i + batch]
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**batch_inputs(tok, [ps[j] for j in idx]), logits_to_keep=1).logits[:, -1].float()
        scores[idx] = (logits[:, label_ids[1]] - logits[:, label_ids[0]]).cpu().numpy()
        bar.update(len(idx))
    bar.close()
    return scores


def train_lora(model, tok, label_ids, df, a, on_epoch=None):
    """on_epoch(ep, model, loss, n_train)：每個 epoch 結束時呼叫（逐 epoch 評估用）。"""
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # 11 GB 顯示卡放不下 1.5B 的完整中間結果：用 gradient checkpointing（多花約 30% 時間，省大量記憶體）
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"LoRA r={a.r}：可訓練參數 {n_train / 1e6:.1f}M（占全部 {100 * n_train / n_all:.2f}%）")

    texts = [p + LABELS[y] for p, y in zip(prompts(tok, df.review_text.tolist()), df.label)]
    targets = torch.tensor([label_ids[y] for y in df.label])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = (len(texts) + a.batch - 1) // a.batch * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    g = torch.Generator().manual_seed(a.seed)
    model.train()
    run_loss = None
    bad_total = bad_run = 0  # NaN 步數：累計、連續
    for ep in range(1, a.epochs + 1):
        model.train()  # 逐 epoch 評估時 score() 會切成 eval 模式，每個 epoch 開始要切回來
        order = torch.randperm(len(texts), generator=g)
        bar = Bar(total=(len(texts) + a.batch - 1) // a.batch, desc=f"LoRA epoch {ep}/{a.epochs}", unit="步")
        for i in range(0, len(texts), a.batch):
            idx = order[i:i + a.batch]
            with torch.autocast("cuda", dtype=torch.float16):
                # 輸入最後一個 token 是答案，所以用倒數第 2 個位置的輸出預測它
                logits = model(**batch_inputs(tok, [texts[j] for j in idx]), logits_to_keep=2).logits[:, 0].float()
            loss = torch.nn.functional.cross_entropy(logits, targets[idx].cuda())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            # fp16 可能溢位：偶爾一步 NaN 時 GradScaler 會跳過那一步；連續 NaN 代表已經發散，
            # 直接中止，讓批次流程立刻改用備援設定（11/20 第四晚：種子 42、學習率 1e-4、batch 8 曾空轉 26 分鐘）
            if not torch.isfinite(loss):
                bad_total += 1
                bad_run += 1
                if bad_total in (1, 10, 100):
                    log(f"警告：loss 出現 NaN／inf（第 {ep} 個 epoch 第 {i // a.batch + 1} 步，累計 {bad_total} 步）")
                if bad_run >= MAX_BAD_RUN:
                    bar.close()
                    raise RuntimeError(f"訓練發散：連續 {bad_run} 步 loss 為 NaN／inf（學習率 {a.lr:g}、batch {a.batch}）")
                bar.update(1)
                continue  # NaN 不計入平均 loss
            bad_run = 0
            run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
            bar.extra = {"loss": f"{run_loss:.4f}"}
            bar.update(1)
        bar.close()
        log(f"LoRA epoch {ep} 完成，loss {run_loss:.4f}")
        if on_epoch:
            on_epoch(ep, model, run_loss, n_train)
    return model, n_train, run_loss


def run_each_epoch(model, tok, label_ids, tr, val, test, a, t0):
    """LoRA 訓練，每個 epoch 結束都評估驗證集與測試集，逐 epoch 存檔（之後用驗證 AUC 挑 epoch）。"""
    rows, eval_sec = [], [0.0]
    keys = ["model", "mode", "train_size", "seed", "epochs", "epoch", "lr", "batch", "n_test"]

    def on_epoch(ep, m, loss, n_train):
        t1 = time.time()
        sv = score(m, tok, label_ids, val.review_text.tolist(), desc=f"epoch {ep} 評估驗證集")
        t2 = time.time()
        st = score(m, tok, label_ids, test.review_text.tolist(), desc=f"epoch {ep} 評估測試集")
        infer = time.time() - t2
        eval_sec[0] += time.time() - t1
        y = test.label.values
        tag = f"llm_{a.model}_lora_n{len(tr)}_s{a.seed}_lr{a.lr:g}_E{a.epochs}_ep{ep}"
        np.save(PRED_DIR / f"{tag}.npy", st)
        np.save(PRED_DIR / f"{tag}_val.npy", sv)
        rows.append(dict(model=a.model, mode="lora", train_size=len(tr), seed=a.seed, epochs=a.epochs, epoch=ep,
                         lr=a.lr, batch=a.batch, lora_r=a.r, trainable_params=n_train, train_loss=round(loss, 4),
                         val_auc=round(roc_auc_score(val.label.values, sv), 4), test_auc=round(roc_auc_score(y, st), 4),
                         test_acc=round(accuracy_score(y, st > 0), 4), test_f1_neg=round(f1_score(y, st > 0, pos_label=0), 4),
                         pred_pos_rate=round(float((st > 0).mean()), 4),
                         train_minutes=round((time.time() - t0 - eval_sec[0]) / 60, 2),
                         infer_sec_per_1k=round(infer / len(test) * 1000, 1), n_test=len(test)))
        r = rows[-1]
        log(f"{a.model} LoRA n={len(tr)} seed={a.seed} epoch {ep}/{a.epochs}：loss {loss:.4f}  驗證 AUC {r['val_auc']:.4f}  "
            f"測試 AUC {r['test_auc']:.4f}  準確率 {r['test_acc']:.4f}  累計訓練 {r['train_minutes']:.1f} 分")
        merge_save(rows, a.out, keys)  # 每個 epoch 都存，中斷也不會白跑

    train_lora(model, tok, label_ids, tr, a, on_epoch=on_epoch)
    out, path = merge_save(rows, a.out, keys)
    log(f"結果已存到 {path}")
    log_text(pd.DataFrame(rows)[["epoch", "train_loss", "val_auc", "test_auc", "test_acc", "train_minutes"]].to_string(index=False))
    log(f"記錄檔 {LOG_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("--mode", choices=["zeroshot", "lora"], default="zeroshot")
    ap.add_argument("--train_size", type=int, help="LoRA 訓練筆數（預設全部 30,464）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="只評估前 N 則測試資料（快速測試用）")
    ap.add_argument("--out", default="rq2_llm.csv")
    ap.add_argument("--shots", type=int, default=0, help="零樣本模式下，在提示裡放幾則範例（少樣本提示）")
    ap.add_argument("--score_val", action="store_true", help="另外對驗證集評分並存檔（給門檻校正用）")
    ap.add_argument("--eval_each_epoch", action="store_true", help="LoRA：每個 epoch 結束都評估驗證集與測試集")
    ap.add_argument("--lr_from", type=int, help="LoRA：學習率改用 --out 檔中「種子 42、這個訓練筆數」驗證 AUC 最高者")
    a = ap.parse_args()
    if a.lr_from:
        r = pd.read_csv(RESULTS / a.out)
        r = r[(r.model == a.model) & (r["mode"] == "lora") & (r.train_size == a.lr_from) & (r.seed == 42)
              & r.val_auc.notna()]
        if r.empty:
            raise SystemExit(f"--lr_from：{a.out} 裡找不到 {a.model}、{a.lr_from} 筆、種子 42 的驗證結果")
        cand = r.groupby("lr").val_auc.max()
        a.lr = float(cand.idxmax())
        log(f"依驗證 AUC 選學習率：{a.lr:g}（候選 {', '.join(f'{k:g}→{v:.4f}' for k, v in cand.items())}）")

    set_seed(a.seed)
    train, val, test = load_olist()
    if a.limit:
        test = test.iloc[:a.limit]
        val = val.iloc[:a.limit]
    if a.shots and a.mode == "zeroshot":
        EXAMPLES[:] = pick_examples(train, a.shots, a.seed)
        a.mode = f"fewshot{a.shots}"
        log(f"少樣本提示：{a.shots} 則範例（正負各半）")
    log(f"GPU：{torch.cuda.get_device_name(0)}；載入 {MODELS[a.model]}（fp16）")
    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    tok.padding_side = "left"
    label_ids = [tok.encode(w, add_special_tokens=False)[0] for w in LABELS]
    assert label_ids[0] != label_ids[1], label_ids
    model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], dtype=torch.float16).cuda()

    t0 = time.time()
    n_train, loss, size = 0, None, 0
    if a.mode == "lora":
        tr = train.loc[subsample_idx(train, a.train_size, a.seed)].reset_index(drop=True)
        size = len(tr)
        log(f"LoRA 訓練 {size} 筆，{a.epochs} epoch，LR {a.lr}，batch {a.batch}")
        if a.eval_each_epoch:
            run_each_epoch(model, tok, label_ids, tr, val, test, a, t0)
            return
        model, n_train, loss = train_lora(model, tok, label_ids, tr, a)
    train_min = (time.time() - t0) / 60

    t1 = time.time()
    s = score(model, tok, label_ids, test.review_text.tolist(), desc=f"{a.model} {a.mode} 評估測試集")
    infer_sec = time.time() - t1
    y = test.label.values
    row = dict(model=a.model, mode=a.mode, train_size=size, seed=a.seed, epochs=a.epochs if size else 0,
               lr=a.lr if size else None, lora_r=a.r if size else None, trainable_params=n_train,
               train_loss=round(loss, 4) if loss else None,
               test_auc=round(roc_auc_score(y, s), 4), test_acc=round(accuracy_score(y, s > 0), 4),
               test_f1_neg=round(f1_score(y, s > 0, pos_label=0), 4), pred_pos_rate=round(float((s > 0).mean()), 4),
               train_minutes=round(train_min, 2), infer_sec_per_1k=round(infer_sec / len(test) * 1000, 1),
               n_test=len(test))
    np.save(PRED_DIR / f"llm_{a.model}_{a.mode}_n{size}_s{a.seed}.npy", s)
    if a.score_val:
        sv = score(model, tok, label_ids, val.review_text.tolist(), desc=f"{a.model} {a.mode} 評估驗證集")
        np.save(PRED_DIR / f"llm_{a.model}_{a.mode}_n{size}_s{a.seed}_val.npy", sv)
        row["val_auc"] = round(roc_auc_score(val.label.values, sv), 4)
    log(f"{a.model} {a.mode} n={size}: 測試 AUC {row['test_auc']:.4f}  準確率 {row['test_acc']:.4f}  "
        f"負評 F1 {row['test_f1_neg']:.4f}  判為正面比例 {row['pred_pos_rate']:.3f}")
    out, path = merge_save([row], a.out, ["model", "mode", "train_size", "seed", "epochs", "n_test"])
    log(f"結果已存到 {path}")
    log_text(out.to_string(index=False))
    log(f"記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
