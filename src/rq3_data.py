"""RQ3／RQ4：把 Olist 的 9 張原始表串成「一筆訂單一列」的資料，並建立特徵。

標籤：負評 = 評分 1–2（neg）；另一種定義 = 評分 1–3（neg3）。
特徵分兩組：
  PRESHIP：下單／付款確認當下就知道的資訊（RQ4 出貨前預警用，不會偷看未來）
  FULL   ：PRESHIP ＋ 實際配送結果（延遲天數等，RQ3 解釋負評原因用）
  PRESHIP2（11/20 補做，ext_preship.py）：PRESHIP 拿掉「下單月份」，加入延遲歷史（NUM_HIST）
賣家歷史評分、延遲歷史、顧客過去訂單數只用「這筆訂單下單之前」就已經發生的資料計算。

直接執行會建立快取並印出資料概況：python src/rq3_data.py
"""
import numpy as np
import pandas as pd

from common import CACHE, DATA

CACHE_FILE = CACHE / "rq3_orders.pkl"
TIME_COL = "purchase_ts"

NUM_PRESHIP = [
    "n_items", "n_products", "n_sellers", "price_total", "freight_total", "freight_ratio", "price_max",
    "payment_value", "installments", "n_payment_types",
    "est_days", "approve_hours", "purchase_hour", "purchase_dow", "purchase_month",
    "weight_g", "volume_cm3", "photos_qty", "desc_len", "name_len",
    "distance_km", "same_state", "seller_prev_n", "seller_prev_neg_rate", "customer_prev_orders",
]
CAT_PRESHIP = ["category", "customer_state", "seller_state", "payment_type"]
NUM_DELIVERY = ["delivered", "delivery_days", "delay_days", "is_late", "carrier_days", "seller_late_handover"]
# 延遲歷史：下單當下就查得到（只用已經送達／已經出貨的過去訂單）
NUM_HIST = ["seller_prev_deliv_n", "seller_prev_late_rate", "seller_prev_handover_late_rate", "route_prev_late_rate",
            "route_est_margin", "platform_late_rate_14d", "state_late_rate_30d", "seller_orders_7d"]
NO_MONTH = [c for c in NUM_PRESHIP if c != "purchase_month"]
FEATURES = {
    "preship": (NUM_PRESHIP, CAT_PRESHIP),
    "full": (NUM_PRESHIP + NUM_DELIVERY, CAT_PRESHIP),
    "preship_nomonth": (NO_MONTH, CAT_PRESHIP),
    "preship_hist": (NUM_PRESHIP + NUM_HIST, CAT_PRESHIP),
    "preship2": (NO_MONTH + NUM_HIST, CAT_PRESHIP),
}


def _read(name, **kw):
    return pd.read_csv(DATA / f"olist_{name}_dataset.csv", **kw)


def _haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build():
    zip_str = {"customer_zip_code_prefix": str, "seller_zip_code_prefix": str, "geolocation_zip_code_prefix": str}
    orders = _read("orders", parse_dates=["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                                          "order_delivered_customer_date", "order_estimated_delivery_date"])
    reviews = _read("order_reviews", parse_dates=["review_creation_date", "review_answer_timestamp"])
    items = _read("order_items", parse_dates=["shipping_limit_date"])
    pays = _read("order_payments")
    products = _read("products")
    sellers = _read("sellers", dtype=zip_str)
    customers = _read("customers", dtype=zip_str)
    geo = _read("geolocation", dtype=zip_str).groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
    trans = pd.read_csv(DATA / "product_category_name_translation.csv")

    # 一筆訂單只留一則評論（同一訂單有多則時取最後一則）
    rv = (reviews.sort_values("review_answer_timestamp").groupby("order_id").tail(1)
          [["order_id", "review_id", "review_score", "review_comment_message", "review_creation_date", "review_answer_timestamp"]])
    d = rv.merge(orders, on="order_id", how="inner")
    d = d.rename(columns={"order_purchase_timestamp": TIME_COL})

    # 品項：主要品項 = 價格最高的那一個（決定主要賣家、品類與商品屬性）
    products = products.merge(trans, on="product_category_name", how="left")
    products["category"] = products.product_category_name_english.fillna(products.product_category_name).fillna("unknown")
    products["volume_cm3"] = products.product_length_cm * products.product_height_cm * products.product_width_cm
    it = items.merge(products, on="product_id", how="left")
    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "count"), n_products=("product_id", "nunique"), n_sellers=("seller_id", "nunique"),
        price_total=("price", "sum"), freight_total=("freight_value", "sum"), price_max=("price", "max"),
        shipping_limit=("shipping_limit_date", "max"))
    main = (it.sort_values("price", ascending=False).groupby("order_id").head(1).set_index("order_id")
            [["seller_id", "category", "product_weight_g", "volume_cm3", "product_photos_qty",
              "product_description_lenght", "product_name_lenght"]]
            .rename(columns={"product_weight_g": "weight_g", "product_photos_qty": "photos_qty",
                             "product_description_lenght": "desc_len", "product_name_lenght": "name_len"}))
    d = d.merge(agg, on="order_id", how="left").merge(main, on="order_id", how="left")
    d["freight_ratio"] = d.freight_total / (d.price_total + d.freight_total)

    # 付款
    pay = pays.groupby("order_id").agg(payment_value=("payment_value", "sum"), installments=("payment_installments", "max"),
                                       n_payment_types=("payment_type", "nunique"))
    pay["payment_type"] = pays.sort_values("payment_value", ascending=False).groupby("order_id").payment_type.first()
    d = d.merge(pay, on="order_id", how="left")

    # 顧客、賣家、距離
    d = d.merge(customers, on="customer_id", how="left").merge(sellers, on="seller_id", how="left")
    c = geo.reindex(d.customer_zip_code_prefix).to_numpy()
    s = geo.reindex(d.seller_zip_code_prefix).to_numpy()
    d["distance_km"] = _haversine(c[:, 0], c[:, 1], s[:, 0], s[:, 1])
    d["same_state"] = (d.customer_state == d.seller_state).astype(float)

    # 時間特徵（下單當下就知道）
    day = pd.Timedelta(days=1)
    d["est_days"] = (d.order_estimated_delivery_date - d[TIME_COL]) / day
    d["approve_hours"] = (d.order_approved_at - d[TIME_COL]) / pd.Timedelta(hours=1)
    d["purchase_hour"] = d[TIME_COL].dt.hour
    d["purchase_dow"] = d[TIME_COL].dt.day_of_week
    d["purchase_month"] = d[TIME_COL].dt.month

    # 配送結果（事後才知道）
    d["delivered"] = d.order_delivered_customer_date.notna().astype(float)
    d["delivery_days"] = (d.order_delivered_customer_date - d[TIME_COL]) / day
    d["delay_days"] = (d.order_delivered_customer_date - d.order_estimated_delivery_date) / day
    d["is_late"] = (d.delay_days > 0).astype(float)
    d.loc[d.delivered == 0, "is_late"] = np.nan
    d["carrier_days"] = (d.order_delivered_carrier_date - d[TIME_COL]) / day
    d["seller_late_handover"] = (d.order_delivered_carrier_date > d.shipping_limit).astype(float)
    d.loc[d.order_delivered_carrier_date.isna(), "seller_late_handover"] = np.nan

    # 標籤
    d["neg"] = (d.review_score <= 2).astype(int)
    d["neg3"] = (d.review_score <= 3).astype(int)

    # 賣家歷史：這筆訂單下單之前，該賣家已經收到（已回覆）的評論數與負評率
    ev = (d[["seller_id", "review_answer_timestamp", "neg"]].dropna(subset=["seller_id"])
          .sort_values("review_answer_timestamp"))
    ev["cum_n"] = ev.groupby("seller_id").cumcount() + 1
    ev["cum_neg"] = ev.groupby("seller_id").neg.cumsum()
    left = d[["order_id", "seller_id", TIME_COL]].dropna(subset=["seller_id"]).sort_values(TIME_COL)
    hist = pd.merge_asof(left, ev[["seller_id", "review_answer_timestamp", "cum_n", "cum_neg"]],
                         left_on=TIME_COL, right_on="review_answer_timestamp", by="seller_id",
                         direction="backward", allow_exact_matches=False)
    hist["seller_prev_n"] = hist.cum_n.fillna(0)
    hist["seller_prev_neg_rate"] = hist.cum_neg / hist.cum_n
    d = d.merge(hist[["order_id", "seller_prev_n", "seller_prev_neg_rate"]], on="order_id", how="left")

    # 顧客過去的訂單數（同一個 customer_unique_id、更早下單）
    d = d.sort_values(TIME_COL).reset_index(drop=True)
    d["customer_prev_orders"] = d.groupby("customer_unique_id").cumcount()
    return add_history(d)


def _past(d, ev_time, value, by=None):
    """每筆訂單下單「之前」（ev_time 早於下單時間）已經發生的事件：累計筆數與累計值。by＝分組欄位，None＝全站。"""
    key = by or "_all"
    x = d.assign(_all=0)
    ev = x.loc[x[ev_time].notna() & x[value].notna() & x[key].notna(), [key, ev_time, value]].sort_values(ev_time)
    ev["cum_n"] = ev.groupby(key).cumcount() + 1
    ev["cum_v"] = ev.groupby(key)[value].cumsum()
    left = x.loc[x[key].notna(), ["order_id", key, TIME_COL]].sort_values(TIME_COL)
    h = pd.merge_asof(left, ev[[key, ev_time, "cum_n", "cum_v"]], left_on=TIME_COL, right_on=ev_time, by=key,
                      direction="backward", allow_exact_matches=False)
    h = h.set_index("order_id").reindex(d.order_id)
    return h.cum_n.to_numpy(dtype=float), h.cum_v.to_numpy(dtype=float)


def _window(d, ev_time, value, days, by=None):
    """下單前 days 天內（[t − days, t)）發生的事件筆數與事件值總和。"""
    key = by or "_all"
    x = d.assign(_all=0).reset_index(drop=True)
    n, v = np.zeros(len(x)), np.zeros(len(x))
    ev = x[x[ev_time].notna() & x[value].notna()]
    groups = {g: e.sort_values(ev_time) for g, e in ev.groupby(key)}
    t_all = x[TIME_COL].to_numpy()
    for g, idx in x.groupby(key).indices.items():
        if g not in groups:
            continue
        e = groups[g]
        T = e[ev_time].to_numpy()
        C = np.concatenate([[0.0], np.cumsum(e[value].to_numpy(dtype=float))])
        t = t_all[idx]
        i1 = np.searchsorted(T, t, "left")
        i0 = np.searchsorted(T, t - np.timedelta64(days, "D"), "left")
        n[idx], v[idx] = i1 - i0, C[i1] - C[i0]
    return n, v


def add_history(d, m=20):
    """延遲歷史特徵（11/20 補做）。都只用下單當下已經發生的事件；路線延遲率以全站延遲率做縮減（m 筆虛擬樣本）。"""
    d = d.copy()
    D, H = "order_delivered_customer_date", "order_delivered_carrier_date"
    d["_route"] = d.seller_state + ">" + d.customer_state  # 任一邊缺值時為 NaN
    # 賣家：過去已送達訂單的延遲率、過去已出貨訂單的逾期交物流比例
    n, v = _past(d, D, "is_late", "seller_id")
    d["seller_prev_deliv_n"] = np.nan_to_num(n)
    d["seller_prev_late_rate"] = v / n
    n, v = _past(d, H, "seller_late_handover", "seller_id")
    d["seller_prev_handover_late_rate"] = v / n
    # 路線（賣家州 → 顧客州）：過去的延遲率、過去的平均實際配送天數 vs 這次承諾的天數
    gn, gv = _past(d, D, "is_late")
    n, v = _past(d, D, "is_late", "_route")
    d["route_prev_late_rate"] = (np.nan_to_num(v) + m * gv / gn) / (np.nan_to_num(n) + m)
    gn, gs = _past(d, D, "delivery_days")
    n, s = _past(d, D, "delivery_days", "_route")
    route_days = (np.nan_to_num(s) + m * gs / gn) / (np.nan_to_num(n) + m)
    d["route_est_margin"] = d.est_days - route_days
    # 最近的物流狀況：全站近 14 天、顧客所在州近 30 天的延遲率（以送達日計）
    n, v = _window(d, D, "is_late", 14)
    d["platform_late_rate_14d"] = np.where(n > 0, v / np.maximum(n, 1), np.nan)
    n, v = _window(d, D, "is_late", 30, "customer_state")
    d["state_late_rate_30d"] = np.where(n >= 5, v / np.maximum(n, 1), np.nan)
    # 賣家最近 7 天的訂單量（忙碌程度）
    d["_one"] = 1.0
    n, _ = _window(d, TIME_COL, "_one", 7, "seller_id")
    d["seller_orders_7d"] = np.where(d.seller_id.notna(), n, np.nan)
    return d.drop(columns=["_route", "_one"])


def load(rebuild=False):
    if CACHE_FILE.exists() and not rebuild:
        d = pd.read_pickle(CACHE_FILE)
        need = {c for num, cat in FEATURES.values() for c in num + cat}
        if need <= set(d.columns):
            return d
        d = add_history(d)  # 舊快取沒有新特徵：補算後覆寫快取
        d.to_pickle(CACHE_FILE)
        return d
    d = build()
    CACHE_FILE.parent.mkdir(exist_ok=True)
    d.to_pickle(CACHE_FILE)
    return d


def split(d, how="time"):
    """time：依下單時間切（訓練 ≤ 2018-03、驗證 2018-04～05、測試 ≥ 2018-06，模擬上線）；random：分層隨機 70/15/15。"""
    if how == "time":
        t = d[TIME_COL]
        return d[t < "2018-04-01"], d[(t >= "2018-04-01") & (t < "2018-06-01")], d[t >= "2018-06-01"]
    rng = np.random.default_rng(42)
    u = rng.random(len(d))
    return d[u < 0.70], d[(u >= 0.70) & (u < 0.85)], d[u >= 0.85]


if __name__ == "__main__":
    d = load(rebuild=True)
    print(f"訂單數 {len(d):,}，期間 {d[TIME_COL].min():%Y-%m-%d} ～ {d[TIME_COL].max():%Y-%m-%d}")
    print(f"負評率（1–2 分）{d.neg.mean():.1%}；（1–3 分）{d.neg3.mean():.1%}")
    for how in ["time", "random"]:
        print(how, [f"{len(x):,}（負評 {x.neg.mean():.1%}）" for x in split(d, how)])
    cols = FEATURES["full"][0] + FEATURES["full"][1]
    print("\n缺值比例：")
    print(d[cols].isna().mean().round(3).sort_values(ascending=False).head(12).to_string())
    print("\n負評率：是否延遲")
    print(d.groupby("is_late", dropna=False).neg.agg(["mean", "size"]).to_string())
