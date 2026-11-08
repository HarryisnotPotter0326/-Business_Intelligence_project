"""加深實驗・晚上批次（GPU，約 6 小時）。

開始前要先下載 Qwen2.5-3B（約 6 GB）：.venv/Scripts/python.exe src/download_models.py qwen-3b
用法和 run_all.py 一樣（--list／--only／--redo），狀態存在 logs/extra_gpu_state.json：
  .venv/Scripts/python.exe src/run_extra_gpu.py
"""
import run_all as R

LLM = "src/rq2_llm.py"
stages, multi = [], {}


def add(sid, title, cmds, minutes, fallback=None):
    stages.append((sid, title, cmds[0], minutes, fallback))
    if len(cmds) > 1:
        multi[sid] = cmds


# 1. 把 BERT 用在真實業務資料（隱性不滿、3 分評論）
add("ext_apply_bert", "BERT 用在業務資料：3 分評論、隱性不滿", [["src/ext_apply_bert.py"]], 15)
# 2. 跨資料集
add("ext_crossdomain", "跨資料集：5 個資料集 × TF-IDF／BERT／LLM 零樣本", [["src/ext_crossdomain.py"]], 45)
# 3. 推論速度
add("ext_speed", "推論速度與顯示卡記憶體", [["src/ext_speed.py"]], 10)
# 4. LLM 零樣本、少樣本（同時對驗證集評分，給門檻校正用）
QW = ["qwen-0.5b", "qwen-1.5b", "qwen-3b"]
add("llm_zeroshot_val", "LLM 零樣本（含驗證集）：0.5B／1.5B／3B", [[LLM, m, "--score_val"] for m in QW], 15)
for k in [4, 8]:
    add(f"llm_fewshot{k}", f"LLM 少樣本 {k} 則範例：0.5B／1.5B／3B", [[LLM, m, "--shots", str(k), "--score_val"] for m in QW], 25)
# 5. LoRA 多種子（原本只有種子 42）
add("lora_0.5b_seeds", "LoRA 0.5B 全部資料：種子 43、44",
    [[LLM, "qwen-0.5b", "--mode", "lora", "--seed", s] for s in ["43", "44"]], 20)
for n, m in [(1000, 6), (5000, 12), (30464, 40)]:
    add(f"lora_1.5b_{n}_seeds", f"LoRA 1.5B，{n} 筆：種子 43、44",
        [[LLM, "qwen-1.5b", "--mode", "lora", "--train_size", str(n), "--seed", s] for s in ["43", "44"]], m,
        [LLM, "qwen-1.5b", "--mode", "lora", "--train_size", str(n), "--seed", "43", "--batch", "8"])
# 6. LoRA 3B
for n, m in [(1000, 8), (5000, 20), (30464, 75)]:
    add(f"lora_3b_{n}", f"LoRA 3B，{n} 筆", [[LLM, "qwen-3b", "--mode", "lora", "--train_size", str(n), "--batch", "8"]], m,
        [LLM, "qwen-3b", "--mode", "lora", "--train_size", str(n), "--batch", "4"])
# 7. 整理（CPU）
add("ext_llm_calibration", "LLM 門檻校正", [["src/ext_llm_calibration.py"]], 1)
add("ext_ensemble_v2", "集成（納入新的 LLM 預測）", [["src/ext_ensemble.py"]], 3)

R.STAGES = stages
R.MULTI = multi
R.STATE = R.LOGS / "extra_gpu_state.json"

if __name__ == "__main__":
    R.main()
