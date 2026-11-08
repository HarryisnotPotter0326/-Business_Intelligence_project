# 資料說明

本研究使用的資料都是公開資料集，未經修改。檔案太大（最大 1.1 GB）無法放上 GitHub，請從下列 Kaggle 頁面下載，把 CSV 檔直接放在這個 `datasets/` 資料夾（與 `src/` 同一層）。檔名與下表相同即可，程式會自動讀取。

## 一、評論情感分類資料（RQ1、RQ2）

來源：Souza, F. D. (2021). *Brazilian Portuguese sentiment analysis datasets* [Data set]. Kaggle. https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets

這是重現對象 Souza and Souza Filho (2022) 公開的資料，已完成前處理並切分為 10 折。

| 檔案 | 筆數 | 大小（MB） | 用途 |
|---|---:|---:|---|
| `olist.csv` | 41,744 | 11.0 | RQ1 重現實驗、RQ2 學習曲線與 LLM、跨資料集 |
| `b2w.csv` | 132,373 | 67.0 | RQ2 跨資料集 |
| `buscape.csv` | 84,991 | 78.8 | RQ2 跨資料集 |
| `utlc_apps.csv` | 1,039,535 | 285.1 | RQ2 跨資料集 |
| `utlc_movies.csv` | 1,487,449 | 1,105.1 | RQ2 跨資料集 |

欄位：`original_index`、`review_text`（原文）、`review_text_processed`（前處理後文字）、`review_text_tokenized`、`polarity`（0＝負面，1＝正面；3 星為空值）、`rating`（1–5 星）、`kfold_polarity`、`kfold_rating`（折號；3 星評論的 `kfold_polarity` 為 −1）。

本研究使用作者的切分：`kfold_polarity` 第 1–8 折為訓練集、第 9 折為驗證集、第 10 折為測試集。跨資料集實驗從第 1–9 折抽 30,000 筆訓練、從第 10 折抽 5,000 筆測試（依標籤分層抽樣，隨機種子 42）。

同一個 Kaggle 頁面還有合併檔 `concatenated.csv`（約 1.5 GB），本研究沒有使用，因此未附。

## 二、Olist 原始訂單資料（RQ3、RQ4）

來源：Olist. (2018). *Brazilian e-commerce public dataset by Olist* [Data set]. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

授權：CC BY-NC-SA 4.0。期間為 2016-09 至 2018-10，共 9 張表，以 `order_id`、`customer_id`、`product_id`、`seller_id` 串接。

| 檔案 | 筆數 | 大小（MB） | 內容 |
|---|---:|---:|---|
| `olist_orders_dataset.csv` | 99,441 | 17.7 | 訂單狀態與各階段時間（下單、出貨、送達、預計送達日） |
| `olist_order_items_dataset.csv` | 112,650 | 15.4 | 訂單品項、賣家、價格、運費、出貨期限 |
| `olist_order_reviews_dataset.csv` | 99,224 | 14.5 | 評分（1–5 星）與評論文字 |
| `olist_order_payments_dataset.csv` | 103,886 | 5.8 | 付款方式、分期數、金額 |
| `olist_customers_dataset.csv` | 99,441 | 9.0 | 顧客代碼、郵遞區號、城市、州 |
| `olist_sellers_dataset.csv` | 3,095 | 0.2 | 賣家郵遞區號、城市、州 |
| `olist_products_dataset.csv` | 32,951 | 2.4 | 商品類別、名稱與描述長度、照片數、重量與尺寸 |
| `olist_geolocation_dataset.csv` | 1,000,163 | 61.3 | 郵遞區號前五碼的經緯度（計算買賣家距離） |
| `product_category_name_translation.csv` | 71 | 0.0 | 商品類別的葡萄牙文與英文對照 |

處理方式見報告 3.2.2 節與 `src/rq3_data.py`：以「一筆訂單一列」串接後，得到 98,673 筆有評分的訂單。
