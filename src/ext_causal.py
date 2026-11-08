"""加深 5：延遲「造成」多少負評？（因果推論，補強 RQ4「36.9% 負評可避免」的相關性估計）

延遲的訂單本來就可能比較特別（偏遠地區、大型商品…），單純比較負評率會把這些差異算進去。三種方法：
1. 斷點迴歸（RD）：以「實際送達 − 預計送達」為跑分變數，比較剛好準時與剛好延遲的訂單
   （在斷點附近，兩邊的訂單條件幾乎一樣）。斷點 0 天與 1 天（預計日當天結束）都做，頻寬 1–5 天。
2. 傾向分數配對（PSM）：用出貨前特徵估計「會延遲的機率」，每筆延遲訂單配一筆條件相近的準時訂單；
   回報配對前後的共變數平衡（標準化平均差 SMD）。
3. 雙重穩健估計（AIPW，2 折交叉擬合）：同時建模「是否延遲」與「負評」，兩個模型任一正確即一致。
   也依延遲天數分組（劑量反應），並估計「未送達」的效果。
最後用因果效果估計「可避免負評數」，和 RQ4 的相關性估計比較。

輸出：results/ext_causal_*.csv；figures/ext_causal_rd.png
用法：python src/ext_causal.py
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402
from monitor import LOG_PATH, Bar, log, log_text  # noqa: E402
import rq3_data  # noqa: E402
from rq3_model import preprocessor  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
NUM, CAT = rq3_data.FEATURES["preship"]
rng = np.random.default_rng(42)


# ---------- 1. 斷點迴歸 ----------
def local_linear(r, y, c, h):
    m = np.abs(r - c) <= h
    x, yy = r[m] - c, y[m]
    w = 1 - np.abs(x) / h  # 三角核
    D = (x > 0).astype(float)
    X = np.column_stack([np.ones_like(x), D, x, D * x])
    XtW = X.T * w
    bread = np.linalg.inv(XtW @ X)
    beta = bread @ XtW @ yy
    e = yy - X @ beta
    meat = (X.T * (w * e) ** 2) @ X
    n, k = X.shape
    V = bread @ meat @ bread * n / (n - k)  # HC1 穩健標準誤
    return beta[1], np.sqrt(V[1, 1]), int(m.sum()), int((m & (r > c)).sum())


def rd(d):
    dl = d[d.delivered == 1]
    r, y = dl.delay_days.values, dl.neg.values.astype(float)
    rows = []
    for c in [0.0, 1.0]:
        for h in [1, 2, 3, 5]:
            jump, se, n, n_right = local_linear(r, y, c, h)
            rows.append(dict(斷點_天=c, 頻寬_天=h, 跳躍_百分點=round(100 * jump, 2), 標準誤=round(100 * se, 2),
                             CI低=round(100 * (jump - 1.96 * se), 2), CI高=round(100 * (jump + 1.96 * se), 2),
                             樣本數=n, 斷點右側樣本數=n_right))
    out = pd.DataFrame(rows)
    # 密度檢查：斷點兩側 1 天內的訂單數（若有人為操弄送達日，會在斷點左側堆積）
    dens = pd.DataFrame([dict(斷點_天=c, 左側1天內=int(((r > c - 1) & (r <= c)).sum()), 右側1天內=int(((r > c) & (r <= c + 1)).sum()))
                         for c in [0.0, 1.0]])
    out.to_csv(RESULTS / "ext_causal_rd.csv", index=False)
    log_text("\n斷點迴歸（局部線性、三角核、HC1 標準誤）：\n" + out.to_string(index=False))
    log_text("\n斷點兩側的樣本數（檢查是否有人為操弄）：\n" + dens.to_string(index=False))

    # 圖：每 0.5 天一組的負評率，加上斷點兩側的局部線性擬合（頻寬 5 天）
    bins = np.arange(-15, 20.5, 0.5)
    g = pd.DataFrame({"r": r, "y": y})
    g["b"] = pd.cut(g.r, bins)
    s = g.groupby("b", observed=True).agg(r=("r", "mean"), y=("y", "mean"), n=("y", "size"))
    s = s[s.n >= 30]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(s.r, 100 * s.y, s=np.clip(s.n / 40, 5, 80), c="#3b6ea5", alpha=0.7, label="每 0.5 天的負評率（點大小＝筆數）")
    for c, col in [(0.0, "#d9822b"), (1.0, "#b03a2e")]:
        ax.axvline(c, ls="--", c=col, lw=1)
        for side in [-1, 1]:
            m = (np.sign(r - c) == side) & (np.abs(r - c) <= 5)
            coef = np.polyfit(r[m], y[m], 1)
            xs = np.linspace(c, c + 5 * side, 20)
            ax.plot(xs, 100 * np.polyval(coef, xs), c=col, lw=2, label=f"斷點 {c:g} 天的局部線性擬合" if side == 1 else None)
    ax.set(xlabel="實際送達 − 預計送達（天）", ylabel="負評率（%）", title="斷點迴歸：送達日超過承諾日的那一刻，負評率跳升多少？")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "ext_causal_rd.png", dpi=200)
    plt.close(fig)
    return out


# ---------- 2.／3. 傾向分數配對與 AIPW ----------
def design(df):
    pre = preprocessor(NUM, CAT, impute=True, scale=True)
    return pre.fit_transform(df[NUM + CAT]), pre


def smd(x, t):
    a, b = x[t == 1], x[t == 0]
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)


def psm(df, label):
    X, pre = design(df)
    t = df.treat.values
    y = df.neg.values.astype(float)
    ps = LogisticRegression(C=0.1, max_iter=3000).fit(X, t).predict_proba(X)[:, 1]
    logit = np.log(ps / (1 - ps))
    cal = 0.2 * logit.std()
    nn = NearestNeighbors(n_neighbors=1).fit(logit[t == 0].reshape(-1, 1))
    dist, j = nn.kneighbors(logit[t == 1].reshape(-1, 1))
    ok = dist.ravel() <= cal
    yt, yc = y[t == 1][ok], y[t == 0][j.ravel()[ok]]
    diff = yt - yc
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(500)]
    # 共變數平衡（數值特徵）
    names = pre.get_feature_names_out()
    Xt, Xc = X[t == 1][ok], X[t == 0][j.ravel()[ok]]
    bal = []
    for i, nm in enumerate(names):
        if nm.startswith("num__") and "missingindicator" not in nm:
            xi = X[:, i]
            after = np.concatenate([Xt[:, i], Xc[:, i]])
            ta = np.r_[np.ones(len(Xt)), np.zeros(len(Xc))]
            bal.append(dict(對照=label, 特徵=nm[5:], SMD_配對前=round(smd(xi, t), 3), SMD_配對後=round(smd(after, ta), 3)))
    naive = y[t == 1].mean() - y[t == 0].mean()
    res = dict(對照=label, 處理組=int(t.sum()), 配對成功=int(ok.sum()), 單純差距_百分點=round(100 * naive, 2),
               PSM_ATT_百分點=round(100 * diff.mean(), 2), CI低=round(100 * np.percentile(boot, 2.5), 2),
               CI高=round(100 * np.percentile(boot, 97.5), 2))
    return res, pd.DataFrame(bal)


def aipw(df, label):
    """ATT 的雙重穩健估計，2 折交叉擬合；標準誤用影響函數。"""
    t = df.treat.values
    y = df.neg.values.astype(float)
    e, m0 = np.zeros(len(df)), np.zeros(len(df))
    for tr, te in KFold(2, shuffle=True, random_state=42).split(df):
        Xtr, pre = design(df.iloc[tr])
        Xte = pre.transform(df.iloc[te][NUM + CAT])
        e[te] = LogisticRegression(C=0.1, max_iter=3000).fit(Xtr, t[tr]).predict_proba(Xte)[:, 1]
        c = t[tr] == 0
        m0[te] = LogisticRegression(C=0.1, max_iter=3000).fit(Xtr[c], y[tr][c]).predict_proba(Xte)[:, 1]
    e = np.clip(e, 0.01, 0.99)
    p1 = t.mean()
    psi = (t * (y - m0) - (1 - t) * e / (1 - e) * (y - m0)) / p1
    att = psi.mean()
    se = np.sqrt(((psi - att * t / p1) ** 2).mean() / len(df))
    return dict(對照=label, 處理組=int(t.sum()), AIPW_ATT_百分點=round(100 * att, 2),
                CI低=round(100 * (att - 1.96 * se), 2), CI高=round(100 * (att + 1.96 * se), 2)), att


def main():
    d = rq3_data.load()
    rd(d)

    ontime = d[(d.delivered == 1) & (d.delay_days <= 0)]
    groups = {
        "延遲（任何天數）": d[(d.delivered == 1) & (d.delay_days > 0)],
        "延遲 0–2 天": d[(d.delivered == 1) & (d.delay_days > 0) & (d.delay_days <= 2)],
        "延遲 2–5 天": d[(d.delivered == 1) & (d.delay_days > 2) & (d.delay_days <= 5)],
        "延遲 5–10 天": d[(d.delivered == 1) & (d.delay_days > 5) & (d.delay_days <= 10)],
        "延遲 >10 天": d[(d.delivered == 1) & (d.delay_days > 10)],
        "未送達": d[d.delivered == 0],
    }
    psm_rows, bal, aipw_rows, att = [], [], [], {}
    for name, g in Bar(list(groups.items()), desc="PSM／AIPW", unit="組"):
        df = pd.concat([g.assign(treat=1), ontime.assign(treat=0)]).reset_index(drop=True)
        r, b = psm(df, name)
        psm_rows.append(r)
        bal.append(b)
        r2, a = aipw(df, name)
        aipw_rows.append(r2)
        att[name] = a
        log(f"{name}：單純差距 {r['單純差距_百分點']:+.1f}、PSM {r['PSM_ATT_百分點']:+.1f}、AIPW {r2['AIPW_ATT_百分點']:+.1f} 百分點")
    P, A = pd.DataFrame(psm_rows), pd.DataFrame(aipw_rows)
    Bal = pd.concat(bal)
    P.to_csv(RESULTS / "ext_causal_psm.csv", index=False)
    A.to_csv(RESULTS / "ext_causal_aipw.csv", index=False)
    Bal.to_csv(RESULTS / "ext_causal_balance.csv", index=False)
    log_text("\n傾向分數配對（ATT：延遲訂單若準時，負評率會差多少）：\n" + P.to_string(index=False))
    log_text("\nAIPW（雙重穩健）：\n" + A.to_string(index=False))
    b = Bal[Bal.對照 == "延遲（任何天數）"].sort_values("SMD_配對前", key=abs, ascending=False).head(10)
    log_text("\n共變數平衡（延遲 vs 準時，|SMD| < 0.1 視為平衡）：\n" + b.to_string(index=False))

    # 可避免負評：各延遲組 ATT × 組內訂單數
    rows = []
    for name in ["延遲 0–2 天", "延遲 2–5 天", "延遲 5–10 天", "延遲 >10 天", "未送達"]:
        rows.append(dict(組別=name, 訂單數=len(groups[name]), AIPW_ATT=round(att[name], 4),
                         可避免負評=round(att[name] * len(groups[name]), 0)))
    av = pd.DataFrame(rows)
    total = av.可避免負評.sum()
    av.to_csv(RESULTS / "ext_causal_avoidable.csv", index=False)
    log_text("\n用因果效果估計的可避免負評：\n" + av.to_string(index=False))
    log(f"因果估計：可避免負評約 {total:,.0f} 則，占全部負評 {total / d.neg.sum():.1%}（RQ4 相關性估計為 5,351 則、36.9%）")
    log(f"結果已存到 results/ext_causal_*.csv、figures/ext_causal_rd.png；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
