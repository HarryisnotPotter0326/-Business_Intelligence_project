"""RQ3 資料探勘：一次跑完（只用 CPU，約 10–15 分鐘）。

用法和 run_all.py 一樣（--list／--only／--redo），狀態存在 logs/rq3_state.json：
  .venv/Scripts/python.exe src/run_rq3.py
"""
import run_all as R

R.STAGES = [
    ("rq3_data", "串接 9 張原始表、建立訂單特徵", ["src/rq3_data.py"], 1, None),
    ("rq3_model", "負評預測：5 種模型 × 特徵組 × 切分方式", ["src/rq3_model.py"], 10, None),
    ("rq3_explain", "SHAP、淺層決策樹規則、各因素負評率", ["src/rq3_explain.py"], 2, None),
    ("rq3_cluster", "顧客分群：RFM ＋ K-means", ["src/rq3_cluster.py"], 2, None),
    ("rq3_topics", "負評文字主題：TF-IDF ＋ NMF", ["src/rq3_topics.py"], 2, None),
]
R.MULTI = {}
R.STATE = R.LOGS / "rq3_state.json"

if __name__ == "__main__":
    R.main()
