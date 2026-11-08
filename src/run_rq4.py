"""RQ4 智慧決策：一次跑完（只用 CPU，約 2–3 分鐘）。

會先重跑 RQ3 的兩段小修正（決策樹缺值處理、存下每則負評的主題），再跑 RQ4。
用法和 run_all.py 一樣（--list／--only／--redo），狀態存在 logs/rq4_state.json：
  .venv/Scripts/python.exe src/run_rq4.py
"""
import run_all as R

R.STAGES = [
    ("rq3_explain_v2", "RQ3 重跑：決策樹改用「未送達＝無限延遲」", ["src/rq3_explain.py"], 1, None),
    ("rq3_topics_v2", "RQ3 重跑：負評主題（存下每則負評的主題給 RQ4）", ["src/rq3_topics.py"], 1, None),
    ("rq4_decision", "RQ4：預警時點、成本敏感門檻、預警名單、改善優先順序", ["src/rq4_decision.py"], 2, None),
]
R.MULTI = {}
R.STATE = R.LOGS / "rq4_state.json"

if __name__ == "__main__":
    R.main()
