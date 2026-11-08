"""第三晚：修正 dropout bug 後，重跑所有 BERT 微調實驗。

bug：from_pretrained 載入的模型是 eval 模式，舊版 rq1_finetune.py 第 1 個 epoch 沒有切回 train 模式，
所以第 1 個 epoch 完全沒有 dropout。修正前的結果備份在 results/nodropout_bug/（可當「無 dropout」對照組）。

用法和 run_all.py 一樣（--list／--only／--redo），狀態存在 logs/night3_state.json：
  .venv/Scripts/python.exe src/run_night3.py
"""
import run_all as R

FT = "src/rq1_finetune.py"
SEEDS8 = ["--seeds", *[str(s) for s in range(42, 50)]]
rq2 = [s for s in R.STAGES if s[0].startswith("rq2_curve_ft")]  # RQ2 學習曲線（Base 微調）

import run_night2  # noqa: E402  匯入後 R.STAGES／R.MULTI 會換成第二晚的內容
sweep = [s for s in R.STAGES if s[0].startswith("sweep")]
multi = {k: v for k, v in R.MULTI.items() if k.startswith("sweep")}

R.STAGES = [
    # 1. 論文設定，8 個種子（最重要，先跑）
    ("ft_base_mbert", "論文設定 8 種子：BERTimbau Base + m-BERT", [FT, "bertimbau-base", "mbert", *SEEDS8], 35, None),
    ("ft_large", "論文設定 8 種子：BERTimbau Large", [FT, "bertimbau-large", *SEEDS8], 25,
     [FT, "bertimbau-large", *SEEDS8, "--batch", "16"]),
    ("significance", "成對 bootstrap 信賴區間與顯著性檢定", ["src/rq1_significance.py"], 5, None),
    # 2. RQ2 學習曲線
    *rq2,
    # 3. 超參數搜尋
    *sweep,
]
R.MULTI = multi
R.STATE = R.LOGS / "night3_state.json"

if __name__ == "__main__":
    R.main()
