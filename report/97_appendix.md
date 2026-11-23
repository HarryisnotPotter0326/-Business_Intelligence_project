#! 附錄 A　補充資料

## A.1　完整實驗結果表

本附錄的表格都由程式直接從 `results/` 中的結果檔產生，數值與正文一致。

> GEN frozen | 表 A-1　BERT 凍結向量：9 種聚合方式的驗證／測試 AUC（%）
> NOTE 註：每格為「驗證 AUC／測試 AUC」；正則化參數 C 以驗證集選擇後，以訓練集加驗證集重新訓練。

> GEN sweep | 表 A-2　BERT 微調超參數搜尋（3 個種子）
> NOTE 註：AUC 為最後一個 epoch 的結果；訓練分鐘為每個種子的平均。

> GEN llm | 表 A-3　Qwen2.5 各次執行的完整結果
> NOTE 註：判斷門檻為 0（未校正）。少樣本與零樣本不需訓練；LoRA 的訓練筆數為抽樣後的訓練資料量。Qwen2.5-3B 的 LoRA 為第一次（學習率 2e-4）的結果，第二輪的結果見表 A-9。

> GEN calib | 表 A-4　LLM 判斷門檻校正的完整結果
> NOTE 註：門檻為 logit 差 s(x) 的切點；AUC 不受門檻影響，因此未列出。

> GEN cross_tfidf | 表 A-5　跨資料集測試 AUC（%）：TF-IDF＋邏輯斯迴歸
> NOTE 註：列為訓練資料集、欄為測試資料集；設定與表 6-3 相同。

> GEN backtest | 表 A-6　滾動回測的逐月 AUC
> NOTE 註：2017-07 為第一個月，兩種做法相同。

## A.2　補充圖

> FIG figures/rq3_tree.png | 15.5 | 圖 A-1　負評預測的深度 3 決策樹規則（完整特徵）

> FIG figures/rq3_shap_bar_preship.png | 13 | 圖 A-2　出貨前特徵的重要性（LightGBM，平均絕對 SHAP 值）

> FIG figures/rq3_cluster_k.png | 13 | 圖 A-3　RFM 分群的群數選擇（輪廓係數）

> FIG figures/rq1_error_coding.png | 14 | 圖 A-4　120 則錯誤樣本的原因歸類

> FIG figures/rq3_shap_bar_preship2.png | 13 | 圖 A-5　加入延遲歷史後的出貨前特徵重要性（LightGBM，平均絕對 SHAP 值）

## A.3　錯誤樣本歸類表

> GEN errors | 表 A-7　120 則最有把握卻判錯的評論與歸類
> NOTE 註：p(正面) 為 BERTimbau Large 微調（8 個種子平均）估計的正面機率；假正評為 1–2 星被判成正面，假負評為 4–5 星被判成負面。原文超過 110 個字元者截斷。歸類由 AI 助理依表 5-4 的準則初分，作者複核後定稿。第二次獨立歸類與本表不同的 8 則為：#11（第二次歸為太短／無意義）、#22（模型真的判錯）、#35（隱含負面）、#73（中性敘述／詢問）、#88、#104、#114（正負混雜）、#112（評分與文字矛盾）；逐則對照見 results/ext_error_recode.csv。

## A.4　程式清單與執行環境

本研究的程式都放在專案的 `src/` 資料夾（表 A-8），實驗結果放在 `results/`，執行記錄放在 `logs/`，圖放在 `figures/`。重現步驟為：（1）以 `download_models.py` 下載模型；（2）依序執行 `run_all.py`、`run_night2.py`、`run_night3.py`（RQ1、RQ2）；（3）執行 `run_rq3.py`、`run_rq4.py`（RQ3、RQ4）；（4）執行 `run_extra_cpu.py`、`run_extra_gpu.py`（加深實驗）；（5）執行 `run_night4.py` 與 `ext_error_recode.py`（第二輪補做）；（6）執行 `fig_eda.py`、`fig_rq12.py` 與 `build_report.py` 產生報告的圖與本文件。

> GEN files | 表 A-8　程式清單

## A.5　第二輪補做實驗的完整結果

第二輪重跑第一次做得不夠好的實驗（3.4 節、3.5.1 節）。每個 epoch 都評估驗證集與測試集，「選用」欄為依各種子平均的驗證 AUC 選出的 epoch，選擇過程不使用測試集。

> GEN lora_v2 | 表 A-9　LoRA 第二輪：逐 epoch 結果
> NOTE 註：epoch 欄為「第幾個 epoch／排程的 epoch 數」；學習率依排程在 10% warm-up 後線性下降到 0，因此 3 個 epoch 排程的第 1 個 epoch 與只訓練 1 個 epoch 的結果不同。Qwen-3B 種子 42、學習率 1e-4 的批次為 4：以批次 8 訓練時半精度溢位、損失變成 NaN（6.2 節）。

> GEN ft8 | 表 A-10　BERTimbau Base 微調延長為 8 個 epoch（3 個種子）
> NOTE 註：學習率 2.5e-5、dropout 0.1，排程與表 6-1 的 3 個 epoch 版本相同，只是總長度不同。

> GEN preship_backtest | 表 A-11　出貨前特徵改良的滾動回測：每月重新訓練的測試 AUC
> NOTE 註：原本為 29 個出貨前特徵；改良版加入 8 個延遲歷史特徵並拿掉下單月份（36 個特徵）。每個月以前一個月以前下單的訂單訓練 LightGBM，預測當月訂單。
