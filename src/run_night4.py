"""第四晚（11/20 補做）：重跑第一次沒做好的實驗，約 6 小時。

1. 出貨前預警（CPU）：加入賣家／路線的延遲歷史、最近的物流狀況，並比較拿掉「下單月份」（ext_preship.py）
2. 學習曲線的小資料點訓練不足：
   - BERTimbau Base 微調 1k／5k：原本 3 epoch 時驗證 AUC 還在上升 → 改成最多 8 epoch，依驗證集挑 epoch
   - Qwen-1.5B LoRA 1k／5k：原本只跑 1 epoch（訓練損失 0.35–0.55）→ 改成 3 epoch；補上 10k（2 epoch）
     都是每個 epoch 評估驗證集與測試集，3 個種子
3. Qwen-3B LoRA 設定沒調好（batch 只能 8，學習率卻沿用 batch 16 的 2e-4，全部資料反而比 5k 差）：
   - 全部資料、種子 42：學習率 1e-4（依 batch 等比例縮小）與 5e-5 各跑一次
   - 依驗證 AUC 選定學習率後，補種子 43、44
   - 1k／5k：3 epoch、種子 42、同一個學習率
   - 11/22 補跑：學習率 5e-5 的種子 43、44（1e-4 的種子 42 在 batch 8 發散，改用 batch 4，設定不一致）
     已完成的段落會自動跳過，重新執行這支程式只會跑這 2 段（約 1.5 小時）
結果：results/ext_preship_*.csv、rq4_*_v2.csv、rq2_finetune.csv（epochs＝8 的列）、rq2_llm_v2.csv
狀態存在 logs/night4_state.json；每段完成會更新 自動結果摘要.md 的「第四晚」段落。

用法（和 run_all.py 一樣，可加 --list／--only／--redo）：
  .venv/Scripts/python.exe src/run_night4.py
"""
import run_all as R

FT, LLM, OUT = "src/rq1_finetune.py", "src/rq2_llm.py", "rq2_llm_v2.csv"
stages, multi = [], {}


def add(sid, title, cmds, minutes, fallback=None):
    stages.append((sid, title, cmds[0], minutes, fallback))
    if len(cmds) > 1:
        multi[sid] = cmds


# 1. 出貨前預警（CPU）
add("preship_v2", "出貨前預警：加入延遲歷史特徵（CPU）", [["src/ext_preship.py"]], 10)

# 2. 學習曲線：BERT Base 微調（3 個種子在同一個指令裡）
for n, m in [(1000, 3), (5000, 8)]:
    add(f"curve_ft_{n}_8ep", f"學習曲線：BERT Base 微調 {n} 筆，8 epoch × 3 種子",
        [[FT, "bertimbau-base", "--train_size", str(n), "--epochs", "8", "--out", "rq2_finetune.csv"]], m)

# 2. 學習曲線：Qwen-1.5B LoRA（每個種子一段，失敗不影響其他種子）
for n, ep, m in [(1000, 3, 5), (5000, 3, 14), (10000, 2, 15)]:
    for s in ["42", "43", "44"]:
        add(f"lora15_{n}_s{s}", f"學習曲線：Qwen-1.5B LoRA {n} 筆，{ep} epoch，種子 {s}",
            [[LLM, "qwen-1.5b", "--mode", "lora", "--train_size", str(n), "--epochs", str(ep), "--seed", s,
              "--eval_each_epoch", "--out", OUT]], m)

# 3. Qwen-3B LoRA（顯示卡記憶體不夠時改用 batch 4）
B3 = [LLM, "qwen-3b", "--mode", "lora", "--batch", "8", "--eval_each_epoch", "--out", OUT]
for lr in ["1e-4", "5e-5"]:
    add(f"lora3b_30k_lr{lr}", f"Qwen-3B LoRA 全部資料，學習率 {lr}，種子 42", [B3 + ["--lr", lr]], 45,
        B3 + ["--lr", lr, "--batch", "4"])
for s in ["43", "44"]:
    add(f"lora3b_30k_s{s}", f"Qwen-3B LoRA 全部資料，種子 {s}（學習率依驗證 AUC 選定）",
        [B3 + ["--seed", s, "--lr_from", "30464"]], 45, B3 + ["--seed", s, "--lr_from", "30464", "--batch", "4"])
for n, m in [(1000, 12), (5000, 28)]:
    add(f"lora3b_{n}", f"Qwen-3B LoRA {n} 筆，3 epoch，種子 42（學習率同上）",
        [B3 + ["--train_size", str(n), "--epochs", "3", "--lr_from", "30464"]], m,
        B3 + ["--train_size", str(n), "--epochs", "3", "--lr_from", "30464", "--batch", "4"])

# 11/22 補跑：學習率 1e-4 的種子 42 在 batch 8 發散，改用 batch 4 才跑完，3 個種子的設定不一致；
# 補上學習率 5e-5 的種子 43、44（batch 8），讓 5e-5 有 3 個種子、設定完全一致的結果
for s in ["43", "44"]:
    add(f"lora3b_30k_lr5e-5_s{s}", f"Qwen-3B LoRA 全部資料，學習率 5e-5，種子 {s}（11/22 補跑）",
        [B3 + ["--lr", "5e-5", "--seed", s]], 45, B3 + ["--lr", "5e-5", "--seed", s, "--batch", "4"])

R.STAGES = stages
R.MULTI = multi
R.STATE = R.LOGS / "night4_state.json"

if __name__ == "__main__":
    R.main()
