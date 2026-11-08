"""RQ3：負評在抱怨什麼？（負評文字的主題分析）

- 資料：評分 1–2 且有留言的評論
- 方法：TF-IDF（1–2 gram）＋ NMF；主題數 K 從 4 到 10，用 UMass coherence 參考選擇（也可用 --k 指定）
- 每個主題：關鍵詞、評論數、代表評論，並串回訂單資料（延遲率、未送達率、多賣家比例、常見品類）

輸出：results/rq3_topics.csv、rq3_topic_examples.csv、rq3_topic_k.csv；figures/rq3_topics.png
用法：python src/rq3_topics.py          （K 自動選）
      python src/rq3_topics.py --k 7    （指定 K）
"""
import argparse
import re
import unicodedata

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import NMF  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402
from monitor import LOG_PATH, Bar, log, log_text  # noqa: E402
import rq3_data  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]

# 葡萄牙文停用詞（已去除重音符號）
STOP = set("""
a o as os um uma uns umas de do da dos das no na nos nas em ao aos para pra pro por pelo pela pelos pelas com sem
e ou mas que se ja nao sim muito muita muitos muitas mais menos bem so tambem ate ainda quando como onde porque pois
eu me meu minha meus minhas voce voces ele ela eles elas lhe lhes nosso nossa isso isto esse essa este esta aquele
aquela foi ser sao era estou estava esta estao tem ter tinha tenho fui fez faz vai vou sera seria seja havia ha
dia dias mesmo agora entao assim aqui la tudo nada outro outra outros outras qual quais cada todo toda todos todas
sobre entre apos antes depois desde the q pq vc ne la lo
""".split())


def clean(s):
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(w for w in s.split() if w not in STOP and len(w) > 1)


def umass(W_top, Xbin):
    """UMass coherence：主題前 10 個詞兩兩在同一則評論出現的程度（越接近 0 越好）。"""
    df = np.asarray(Xbin.sum(0)).ravel()
    scores = []
    for words in W_top:
        s = []
        for i in range(1, len(words)):
            for j in range(i):
                co = Xbin[:, words[i]].multiply(Xbin[:, words[j]]).sum()
                s.append(np.log((co + 1) / df[words[j]]))
        scores.append(np.mean(s))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int)
    a = ap.parse_args()
    d = rq3_data.load()
    neg = d[(d.neg == 1) & d.review_comment_message.notna()].copy()
    neg["clean"] = neg.review_comment_message.map(clean)
    neg = neg[neg.clean.str.len() > 0].reset_index(drop=True)
    log(f"有留言的負評 {len(neg):,} 則（占全部負評 {len(neg) / d.neg.sum():.1%}）")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.5, sublinear_tf=True)
    X = vec.fit_transform(neg.clean)
    vocab = np.array(vec.get_feature_names_out())
    Xbin = (X > 0).astype(np.float32).tocsc()
    log(f"詞彙 {len(vocab):,} 個")

    rows, models = [], {}
    for k in Bar(range(4, 11), desc="NMF 選 K", unit="個"):
        nmf = NMF(n_components=k, init="nndsvda", random_state=42, max_iter=500).fit(X)
        top = [np.argsort(-w)[:10] for w in nmf.components_]
        coh = umass(top, Xbin)
        rows.append(dict(k=k, umass=round(coh, 4), reconstruction_err=round(nmf.reconstruction_err_, 2)))
        models[k] = nmf
        log(f"K={k}：UMass {coh:.3f}")
    ks = pd.DataFrame(rows)
    ks.to_csv(RESULTS / "rq3_topic_k.csv", index=False)
    k = a.k or int(ks[ks.k >= 5].sort_values("umass", ascending=False).k.iloc[0])  # 至少 5 個主題，避免太粗
    log(f"使用 K = {k}")

    nmf = models[k]
    Wdoc = nmf.transform(X)
    neg["topic"] = Wdoc.argmax(1)
    neg["topic_weight"] = Wdoc.max(1)
    topics = []
    for t in range(k):
        g = neg[neg.topic == t]
        words = vocab[np.argsort(-nmf.components_[t])[:15]]
        topics.append(dict(
            topic=t, 關鍵詞=", ".join(words), 評論數=len(g), 占比=round(len(g) / len(neg), 3),
            平均評分=round(g.review_score.mean(), 2), 延遲率=round(g.is_late.mean(), 3),
            未送達率=round(1 - g.delivered.mean(), 3), 多賣家訂單比例=round((g.n_sellers > 1).mean(), 3),
            多品項訂單比例=round((g.n_items > 1).mean(), 3), 平均延遲天數=round(g.delay_days.mean(), 1),
            常見品類=", ".join(g.category.value_counts().head(3).index)))
    tp = pd.DataFrame(topics).sort_values("評論數", ascending=False)
    tp.to_csv(RESULTS / "rq3_topics.csv", index=False)
    ex = (neg.sort_values("topic_weight", ascending=False).groupby("topic").head(5)
          [["topic", "review_score", "review_comment_message", "is_late", "delivered", "category"]].sort_values("topic"))
    ex.to_csv(RESULTS / "rq3_topic_examples.csv", index=False)
    neg[["order_id", "topic", "topic_weight"]].to_csv(RESULTS / "rq3_topic_assign.csv", index=False)  # RQ4 用
    log_text("\n負評主題：\n" + tp.to_string(index=False))
    log_text("\n各主題代表評論（每主題 5 則）：\n" + ex.to_string(index=False, max_colwidth=120))

    fig, ax = plt.subplots(figsize=(9, 0.6 * k + 1.5))
    lab = [f"主題 {r.topic}：{', '.join(r.關鍵詞.split(', ')[:4])}" for r in tp.itertuples()][::-1]
    ax.barh(lab, tp.評論數[::-1], color="#b03a2e")
    ax.set_xlabel("評論數")
    ax.set_title("負評主題（NMF）")
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_topics.png", dpi=200)
    plt.close(fig)
    log(f"結果已存到 results/rq3_topic*.csv、figures/rq3_topics.png；記錄檔 {LOG_PATH}")


if __name__ == "__main__":
    main()
