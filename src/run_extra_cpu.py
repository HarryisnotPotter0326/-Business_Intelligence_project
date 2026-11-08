"""加深實驗・白天批次（只用 CPU，約 15–25 分鐘）。

用法和 run_all.py 一樣（--list／--only／--redo），狀態存在 logs/extra_cpu_state.json：
  .venv/Scripts/python.exe src/run_extra_cpu.py
"""
import run_all as R

R.STAGES = [
    ("ext_tfidf_gap", "TF-IDF 與論文差 0.9 個百分點的原因", ["src/ext_tfidf_gap.py"], 3, None),
    ("ext_error_analysis", "錯誤分析：最好的模型錯在哪裡", ["src/ext_error_analysis.py"], 1, None),
    ("ext_ensemble", "集成：把不同模型的預測合起來", ["src/ext_ensemble.py"], 3, None),
    ("ext_backtest", "依時間滾動回測（每月重訓 vs 不更新）", ["src/ext_backtest.py"], 3, None),
    ("ext_causal", "因果推論：斷點迴歸、傾向分數配對、AIPW", ["src/ext_causal.py"], 5, None),
    ("ext_cluster2", "用購物經驗重新分群", ["src/ext_cluster2.py"], 3, None),
    ("ext_montecarlo", "成本不確定下的蒙地卡羅模擬", ["src/ext_montecarlo.py"], 2, None),
]
R.MULTI = {}
R.STATE = R.LOGS / "extra_cpu_state.json"

if __name__ == "__main__":
    R.main()
