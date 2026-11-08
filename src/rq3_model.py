"""RQ3：用訂單的結構化特徵預測負評（LR、決策樹、隨機森林、XGBoost、LightGBM）。

比較：特徵組（full＝含配送結果／preship＝出貨前就知道）× 切分方式（time＝依時間／random＝隨機）。
另外用 full＋time 再跑一次「1–3 分算負評」的定義。
超參數都在驗證集上選，選好後用 訓練＋驗證 重新訓練，再算測試集分數；
分類門檻也在驗證集上選（F1 最大），不偷看測試集。

用法：python src/rq3_model.py            （全部，約 5–10 分鐘，只用 CPU）
      python src/rq3_model.py --quick    （只用 1 萬筆、少量模型，快速測試用）
"""
import argparse
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import CACHE, RESULTS, merge_save
from monitor import LOG_PATH, Bar, log, log_text
import rq3_data

PRED = RESULTS / "preds"
PRED.mkdir(exist_ok=True)


def preprocessor(num, cat, impute, scale=False):
    num_steps = [SimpleImputer(strategy="median", add_indicator=True)] if impute else []
    if scale:
        num_steps.append(StandardScaler())
    num_pipe = make_pipeline(*num_steps) if num_steps else "passthrough"
    onehot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200, sparse_output=False)
    return ColumnTransformer([("num", num_pipe, num), ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"), onehot), cat)],
                             verbose_feature_names_out=True)


def candidates(quick):
    """每個模型：(名稱, 是否需要補缺值, 是否標準化, 超參數候選清單, 建立模型的函式)。"""
    rf_jobs = -1
    c = [
        ("LR", True, True, [dict(C=x) for x in [0.01, 0.1, 1.0]],
         lambda p: LogisticRegression(max_iter=3000, **p)),
        ("DecisionTree", True, False, [dict(max_depth=x) for x in [4, 6, 8, 12]],
         lambda p: DecisionTreeClassifier(min_samples_leaf=50, random_state=42, **p)),
        ("RandomForest", True, False, [dict(min_samples_leaf=x) for x in [5, 20]],
         lambda p: RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=rf_jobs, random_state=42, **p)),
        ("XGBoost", False, False, [dict(max_depth=x) for x in [4, 6]],
         lambda p: xgb.XGBClassifier(**{**dict(n_estimators=2000, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                     early_stopping_rounds=100, eval_metric="auc", n_jobs=-1, random_state=42), **p})),
        ("LightGBM", False, False, [dict(num_leaves=x) for x in [31, 63]],
         lambda p: lgb.LGBMClassifier(**{**dict(n_estimators=2000, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, random_state=42, verbose=-1), **p})),
    ]
    if quick:
        c = [x for x in c if x[0] in ("LR", "LightGBM")]
    return c


def fit_one(name, model, Xtr, ytr, Xva=None, yva=None):
    """XGBoost／LightGBM 用驗證集 early stopping；回傳模型與最佳迭代數。"""
    if name == "XGBoost" and Xva is not None:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return model, model.best_iteration + 1
    if name == "LightGBM" and Xva is not None:
        model.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(yva,), eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        return model, model.best_iteration_
    model.fit(Xtr, ytr)
    return model, None


def best_threshold(y, p):
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.99, 50)))
    f1s = [f1_score(y, p >= t) for t in qs]
    return float(qs[int(np.argmax(f1s))])


def run_setting(feat, how, label, d, quick, bar):
    num, cat = rq3_data.FEATURES[feat]
    tr, va, te = rq3_data.split(d, how)
    cols = num + cat
    rows = []
    for name, impute, scale, grid, make in candidates(quick):
        t0 = time.time()
        pre = preprocessor(num, cat, impute, scale).fit(tr[cols])
        Xtr, Xva = pre.transform(tr[cols]), pre.transform(va[cols])
        best = None
        for p in grid:
            m, it = fit_one(name, make(p), Xtr, tr[label], Xva, va[label])
            auc = roc_auc_score(va[label], m.predict_proba(Xva)[:, 1])
            if best is None or auc > best[0]:
                best = (auc, p, it, m.predict_proba(Xva)[:, 1])
        val_auc, p, it, pva = best
        thr = best_threshold(va[label].values, pva)
        # 用 訓練＋驗證 重新訓練（XGBoost／LightGBM 固定為 early stopping 找到的迭代數）
        trva = pd.concat([tr, va])
        p_final = dict(p, **({"n_estimators": it} if it else {}))
        model = make(p_final)
        if name == "XGBoost":
            model.set_params(early_stopping_rounds=None)
        pipe = Pipeline([("pre", preprocessor(num, cat, impute, scale)), ("model", model)]).fit(trva[cols], trva[label])
        pte = pipe.predict_proba(te[cols])[:, 1]
        y = te[label].values
        rows.append(dict(features=feat, split=how, label=label, model=name, params=str(p_final),
                         n_train=len(trva), n_test=len(te), test_pos_rate=round(y.mean(), 4),
                         val_auc=round(val_auc, 4), test_auc=round(roc_auc_score(y, pte), 4),
                         test_ap=round(average_precision_score(y, pte), 4),
                         threshold=round(thr, 4), test_precision=round(precision_score(y, pte >= thr), 4),
                         test_recall=round(recall_score(y, pte >= thr), 4), test_f1=round(f1_score(y, pte >= thr), 4),
                         seconds=round(time.time() - t0, 1)))
        tag = f"{feat}_{how}_{label}_{name}"
        np.save(PRED / f"rq3_{tag}.npy", pte)
        if name == "LightGBM" and not quick:
            joblib.dump(pipe, CACHE / f"rq3_lgbm_{feat}_{how}_{label}.pkl")
        r = rows[-1]
        log(f"[{feat}/{how}/{label}] {name:<12} 驗證 AUC {r['val_auc']:.4f}  測試 AUC {r['test_auc']:.4f}  "
            f"PR-AUC {r['test_ap']:.4f}  F1 {r['test_f1']:.4f}  {r['seconds']:.0f} 秒  {p_final}")
        bar.update(1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    d = rq3_data.load()
    if a.quick:
        d = d.sample(10000, random_state=42).sort_values(rq3_data.TIME_COL)
    log(f"訂單 {len(d):,} 筆，負評率 {d.neg.mean():.1%}")
    settings = [("full", "time", "neg"), ("preship", "time", "neg"), ("full", "random", "neg"),
                ("preship", "random", "neg"), ("full", "time", "neg3")]
    if a.quick:
        settings = settings[:2]
    out_name = "rq3_models_quick.csv" if a.quick else "rq3_models.csv"
    bar = Bar(total=len(settings) * len(candidates(a.quick)), desc="RQ3 模型", unit="個")
    for s in settings:
        out, path = merge_save(run_setting(*s, d, a.quick, bar), out_name, ["features", "split", "label", "model"])
    bar.close()
    show = out[["features", "split", "label", "model", "test_auc", "test_ap", "test_precision", "test_recall", "test_f1", "seconds"]]
    log_text("\n" + show.sort_values(["label", "split", "features", "test_auc"], ascending=[True, True, True, False]).to_string(index=False))
    log(f"結果已存到 {path}；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
