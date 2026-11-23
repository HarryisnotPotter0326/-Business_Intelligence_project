#! 附錄　程式清單

本研究的程式都放在專案的 `src/` 資料夾（表 A-1），實驗結果放在 `results/`，執行記錄放在 `logs/`，圖放在 `figures/`；本報告的表格與圖都由程式從結果檔產生，各次實驗的完整數值見 `results/` 中的結果檔與 `實驗紀錄.md`。重現步驟為：以 `download_models.py` 下載模型；依序執行 `run_all.py`、`run_night2.py`、`run_night3.py`（RQ1、RQ2）、`run_rq3.py`、`run_rq4.py`（RQ3、RQ4）、`run_extra_cpu.py`、`run_extra_gpu.py`（加深實驗）與 `run_night4.py`、`ext_error_recode.py`（第二輪補做）；最後以 `fig_eda.py`、`fig_rq12.py` 與 `build_report.py` 產生圖與本文件。

> GEN files | 表 A-1　程式清單
