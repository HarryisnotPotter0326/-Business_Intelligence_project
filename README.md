# Olist 評論情感分析與負評決策：程式碼

國立中央大學「商業智慧」期中報告《以預訓練語言模型重現與延伸電商評論情感分析：Olist 巴西電商的負評成因、預警與改善決策》的程式碼、實驗結果與報告。

- 報告：[`報告/`](報告/)（完整版 Word 與 PDF，98 頁）；[`報告/精簡版/`](報告/精簡版/)（精簡版，40 頁，只留主要結果）
- 資料：因檔案太大（最大 1.1 GB）未放入 repo，請依 [`datasets/README.md`](datasets/README.md) 下載

## 內容

| 資料夾／檔案 | 內容 |
|---|---|
| `src/` | 全部程式，清單見本文最後一節，與報告附錄表 A-8 相同 |
| `results/` | 實驗結果（CSV），報告中的數字都來自這裡。`preds/` 是各模型在測試集上的預測機率，供顯著性檢定使用；`nodropout_bug/` 是修正 dropout 問題前的舊結果（報告 5.5 節） |
| `figures/` | 報告中的圖 |
| `logs/` | 每次執行的記錄檔，含當時的 CPU／GPU 使用情形。`*_state.json` 記錄批次流程完成到哪一段 |
| `report/` | 報告原稿（Markdown，每章一個檔），`build_report.py` 由此產生 Word 與 PDF |
| `report_精簡版/` | 精簡版報告原稿（格式同上） |
| `實驗紀錄.md` | 每段實驗的結果與解讀 |
| `自動結果摘要.md` | 批次流程每段完成後自動彙整的結果 |
| `requirements.txt` | Python 套件與版本 |
| `報告/` | 期中報告的 Word 與 PDF |
| `datasets/` | 資料說明與下載方式（資料檔本身不在 repo 中） |

## 執行環境

- Windows 11、Python 3.11
- GPU：NVIDIA GeForce RTX 2080 Ti（11 GB）、CUDA 12.6。BERT 微調與大型語言模型需要 GPU；TF-IDF、RQ3、RQ4 只需要 CPU。
- 安裝套件：

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

## 準備資料與模型

1. 依 [`datasets/README.md`](datasets/README.md) 從 Kaggle 下載資料，把 CSV 檔放進 `datasets/`：

```
-Business_Intelligence_project/
├── datasets/      ← 放下載的 CSV
├── src/
├── results/
├── figures/
└── ...
```

2. 下載預訓練模型。程式預設為離線模式（`common.py` 設定 `HF_HUB_OFFLINE=1`），所以要先下載模型才能執行實驗：

```
python src/download_models.py                                  # BERTimbau Base／Large、m-BERT，約 2.4 GB
python src/download_models.py qwen-0.5b qwen-1.5b qwen-3b      # Qwen2.5，約 10 GB
```

## 重現步驟

附上的 `results/` 與 `logs/` 是已經跑完的結果。批次流程會依 `logs/*_state.json` 跳過已完成的段落，**要從頭重跑請加 `--redo`**。所有指令都在專案資料夾執行。

| 步驟 | 指令 | 內容 |
|---|---|---|
| 1 | `python src/run_all.py --redo` | RQ1、RQ2 主要實驗 |
| 2 | `python src/run_night2.py --redo`<br>`python src/run_night3.py --redo` | RQ1、RQ2 補充實驗（超參數搜尋、多種子、LLM） |
| 3 | `python src/run_rq3.py --redo`<br>`python src/run_rq4.py --redo` | RQ3 資料探勘、RQ4 決策分析 |
| 4 | `python src/run_extra_cpu.py --redo`<br>`python src/run_extra_gpu.py --redo` | 加深實驗（因果推論、回測、跨資料集、錯誤分析等） |
| 5 | `python src/run_night4.py --redo`<br>`python src/ext_error_recode.py` | 第二輪補做（學習曲線延長訓練、Qwen-3B 學習率、出貨前特徵改良、錯誤歸類的重測信度） |
| 6 | `python src/fig_eda.py`<br>`python src/fig_rq12.py` | 報告用圖 |
| 7 | `python src/build_report.py`<br>`python src/build_report.py --src report_精簡版 --out 報告/精簡版 --name 商業智慧期中報告_Olist評論情感分析與負評決策_精簡版` | 產生報告的 Word 與 PDF（完整版、精簡版；PDF 需要 Word 或 LibreOffice） |

說明：

- 每段完成後，結果寫入 `results/`，執行記錄寫入 `logs/`，並更新 `自動結果摘要.md`。
- 隨機種子：單次實驗用 42；BERT 微調的多種子實驗用 42–49。
- GPU 實驗（BERT 微調、LLM）耗時最長，本研究分幾個晚上才跑完。
- 在 Windows 把輸出導向檔案時，請先設定 `PYTHONIOENCODING=utf-8`，以免中文訊息出錯。

## 程式清單（src/）

| 程式 | 用途 |
|---|---|
| `common.py`、`monitor.py` | 共用設定（路徑、離線模式、隨機種子）；進度條與 CPU／GPU 監控、記錄檔 |
| `bert_utils.py` | BERT 共用函式：文字編碼、預測、微調訓練、釋放顯示卡記憶體 |
| `run_all.py`、`run_night2.py`、`run_night3.py` | RQ1／RQ2 批次流程（每段完成自動更新結果摘要） |
| `rq1_tfidf.py` | TF-IDF＋邏輯斯迴歸基準與設定敏感度 |
| `rq1_frozen.py` | BERT 凍結向量（9 種聚合）＋邏輯斯迴歸 |
| `rq1_finetune.py` | BERT 微調（論文設定、多種子、超參數搜尋、學習曲線） |
| `rq1_significance.py` | 成對 bootstrap 信賴區間與顯著性檢定 |
| `rq2_learning_curve.py` | 學習曲線（TF-IDF、凍結 BERT） |
| `rq2_llm.py` | Qwen2.5 零樣本、少樣本與 LoRA 微調 |
| `download_models.py` | 下載 Hugging Face 模型到本機快取 |
| `rq3_data.py` | 串接 Olist 9 張表，建立訂單層級資料與特徵 |
| `rq3_model.py` | 負評預測（5 種模型 × 2 組特徵 × 2 種切分） |
| `rq3_explain.py` | SHAP、決策樹規則、各因素負評率 |
| `rq3_cluster.py`、`rq3_topics.py` | RFM＋K-means 分群；負評文字 NMF 主題 |
| `rq4_decision.py` | 預警時點、成本敏感名單、改善優先順序、預警名單 |
| `run_rq3.py`、`run_rq4.py` | RQ3／RQ4 批次流程 |
| `ext_tfidf_gap.py`、`ext_error_analysis.py`、`ext_error_coding.py` | TF-IDF 差距分析；錯誤分析與錯誤原因歸類 |
| `ext_ensemble.py` | 集成與成對 bootstrap |
| `ext_backtest.py` | 依月份滾動回測 |
| `ext_causal.py` | 斷點迴歸、傾向分數配對、AIPW |
| `ext_cluster2.py` | 購物經驗分群（K-means、GMM） |
| `ext_montecarlo.py` | 成本不確定下的蒙地卡羅模擬 |
| `run_night4.py` | 第二輪補做的批次流程 |
| `ext_preship.py` | 出貨前特徵改良：延遲歷史特徵的消融比較、SHAP、預警時點、蒙地卡羅、滾動回測 |
| `ext_error_blind.py`、`ext_error_recode.py`、`ext_error_kappa.py` | 錯誤歸類的盲測表、第二次獨立歸類與 Cohen's κ |
| `ext_apply_bert.py` | BERT 交叉評分有留言的評論（3 星、隱性不滿） |
| `ext_crossdomain.py` | 跨資料集 5×5（TF-IDF、BERT、Qwen 零樣本） |
| `ext_speed.py`、`ext_llm_calibration.py` | 推論速度與記憶體；LLM 門檻校正 |
| `run_extra_cpu.py`、`run_extra_gpu.py` | 加深實驗批次流程 |
| `fig_eda.py`、`fig_rq12.py` | 報告用圖（研究流程、EDA、RQ1／RQ2） |
| `summarize.py`、`build_report.py` | 自動結果摘要；產生報告的 Word 與 PDF |
