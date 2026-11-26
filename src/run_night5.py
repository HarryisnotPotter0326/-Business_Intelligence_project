"""第五晚（11/26 補強）：和同學的報告比較後補上的檢定與連結實驗，只用 CPU，約 10 分鐘。

1. ext_robust.py：統計檢定與樣本選擇
   - RQ1 的 21 組成對比較做 Holm 多重比較校正
   - RQ3 五種模型的 AUC 差距：成對 bootstrap（2,000 次）＋ Holm 校正；出貨前特徵的三種改良也一起檢定
   - RQ4 三個預警時點：AUC、Precision@1%、Precision@5% 的 95% 信賴區間，以及時點之間的差距檢定
   - 樣本選擇：有留言 vs 沒有留言的評論、延遲是否影響留言率、有評論 vs 沒有評論的訂單
2. ext_bridge.py：把兩條主線接起來
   - 以「星等或 BERT 判斷的文字」定義不滿，用 AIPW 重估延遲的因果效果，估計只看星等會漏掉多少可避免的不滿
結果：results/ext_robust_*.csv、results/ext_bridge_*.csv
狀態存在 logs/night5_state.json；每段完成會更新 自動結果摘要.md 的「第五晚」段落。

用法（和 run_all.py 一樣，可加 --list／--only／--redo）：
  .venv/Scripts/python.exe src/run_night5.py
"""
import run_all as R

R.STAGES = [
    ("robust", "統計檢定補強：RQ1 多重比較校正、RQ3 模型差距、RQ4 預警時點的信賴區間、樣本選擇（CPU）",
     ["src/ext_robust.py"], 6, None),
    ("bridge", "兩條主線的連結：以「星等或文字」定義不滿，重估延遲的因果效果（CPU）",
     ["src/ext_bridge.py"], 3, None),
]
R.MULTI = {}
R.STATE = R.LOGS / "night5_state.json"

if __name__ == "__main__":
    R.main()
