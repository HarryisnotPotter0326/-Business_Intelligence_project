"""第二晚的自動實驗：RQ1 加深（更多種子＋統計檢定＋微調超參數搜尋）。

用法和 run_all.py 一樣（--list／--only／--redo），狀態另存在 logs/night2_state.json：
  .venv/Scripts/python.exe src/run_night2.py
"""
import run_all as R

FT = "src/rq1_finetune.py"
SEEDS = ["--seeds", "42", "43", "44"]
EXTRA = ["--seeds", "45", "46", "47", "48", "49"]

stages, multi = [], {}

# 1. 論文設定多跑 5 個種子（共 8 個），結果併入 rq1_finetune.csv
stages += [
    ("seeds_base_mbert", "論文設定多跑 5 種子：BERTimbau Base + m-BERT", [FT, "bertimbau-base", "mbert", *EXTRA], 20, None),
    ("seeds_large", "論文設定多跑 5 種子：BERTimbau Large", [FT, "bertimbau-large", *EXTRA], 20,
     [FT, "bertimbau-large", *EXTRA, "--batch", "16"]),
    # 2. 統計檢定
    ("significance", "成對 bootstrap 信賴區間與顯著性檢定", ["src/rq1_significance.py"], 5, None),
]

# 3. 超參數搜尋：Base 學習率 3 種 × dropout 2 種 × epoch 數 1/2/4，各 3 種子
for lr in ["1e-5", "2.5e-5", "5e-5"]:
    for do in ["0", "0.1"]:
        sid = f"sweep_base_lr{lr}_do{do}"
        cmds = [[FT, "bertimbau-base", "--lr", lr, "--dropout", do, "--epochs", E, *SEEDS, "--out", "rq1_sweep.csv"]
                for E in ["1", "2", "4"]]
        stages.append((sid, f"超參數搜尋 Base：LR {lr}、dropout {do}、epoch 1/2/4", cmds[0], 25, None))
        multi[sid] = cmds

# Large：學習率 2 種 × epoch 1/2（dropout 0.1），各 3 種子
for lr in ["1e-5", "2.5e-5"]:
    sid = f"sweep_large_lr{lr}"
    cmds = [[FT, "bertimbau-large", "--lr", lr, "--dropout", "0.1", "--epochs", E, *SEEDS, "--out", "rq1_sweep.csv"]
            for E in ["1", "2"]]
    stages.append((sid, f"超參數搜尋 Large：LR {lr}、epoch 1/2", cmds[0], 35, None))
    multi[sid] = cmds

R.STAGES = stages
R.MULTI = multi
R.STATE = R.LOGS / "night2_state.json"

if __name__ == "__main__":
    R.main()
