"""A3: LDA baseline with Gensim.

Two steps (run in order):
  python -m src.topics tune  --papers data/papers.parquet --out results   # pick k by c_v coherence
  python -m src.topics final --papers data/papers.parquet --out results --k 20   # train on all papers
"""
import os
import pickle
import multiprocessing as mp

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel

from src.config import SEED


def load_tokens(papers_path, n=None):
    df = pd.read_parquet(papers_path, columns=["id", "text_lemma"])
    if n and n < len(df):
        df = df.sample(n, random_state=SEED).sort_values("id")
    return df["id"].tolist(), [t.split() for t in df["text_lemma"]]


def build_corpus(tokens, no_below=20, no_above=0.4, keep_n=20000):
    """Dictionary + bag-of-words. Drops very rare words and words in >40% of papers."""
    dct = Dictionary(tokens)
    dct.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    corpus = [dct.doc2bow(t) for t in tokens]
    return dct, corpus


def train_lda(corpus, dct, k, passes=5, workers=None):
    workers = workers or max(1, mp.cpu_count() - 1)
    return LdaMulticore(corpus=corpus, id2word=dct, num_topics=k, passes=passes,
                        chunksize=4000, workers=workers, random_state=SEED,
                        eval_every=None)


def coherence_cv(model, tokens, dct, topn=10):
    cm = CoherenceModel(model=model, texts=tokens, dictionary=dct,
                        coherence="c_v", topn=topn, processes=max(1, mp.cpu_count() - 1))
    return cm.get_coherence()


def top_words_table(model, topn=15):
    rows = []
    for t in range(model.num_topics):
        words = [w for w, _ in model.show_topic(t, topn=topn)]
        rows.append({"lda_topic": t, "top_words": ", ".join(words), "topic_label": ""})
    return pd.DataFrame(rows)


def assign_topics(model, corpus, ids):
    """Dominant topic and its probability for every paper."""
    out = []
    for bow in corpus:
        dist = model.get_document_topics(bow, minimum_probability=0.0)
        t, p = max(dist, key=lambda x: x[1]) if dist else (-1, 0.0)
        out.append((int(t), float(p)))
    return pd.DataFrame({"id": ids, "lda_topic": [o[0] for o in out], "lda_prob": [o[1] for o in out]})


def cmd_tune(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    ids, tokens = load_tokens(a.papers, a.sample)
    dct, corpus = build_corpus(tokens)
    print(f"tuning on {len(tokens)} papers, vocabulary {len(dct)} words")
    rows = []
    for k in a.ks:
        m = train_lda(corpus, dct, k, passes=a.passes)
        c = coherence_cv(m, tokens, dct)
        rows.append({"k": k, "c_v": round(c, 4)})
        print(f"k={k:3d}  c_v={c:.4f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/lda_coherence.csv", index=False)
    plt.figure(figsize=(6, 4)); plt.plot(res.k, res.c_v, marker="o")
    plt.xlabel("number of topics (k)"); plt.ylabel("c_v coherence"); plt.title("LDA coherence vs k")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{a.out}/lda_coherence.png", dpi=150)
    print("best k by c_v:", int(res.loc[res.c_v.idxmax(), "k"]))


def cmd_final(a):
    os.makedirs(a.out, exist_ok=True)
    ids, tokens = load_tokens(a.papers)
    dct, corpus = build_corpus(tokens)
    print(f"training k={a.k} on {len(tokens)} papers, vocabulary {len(dct)} words")
    model = train_lda(corpus, dct, a.k, passes=a.passes)
    model.save(f"{a.out}/lda_model")
    with open(f"{a.out}/lda_dictionary.pkl", "wb") as f:
        pickle.dump(dct, f)
    words = top_words_table(model)
    words.to_csv(f"{a.out}/lda_topic_words.csv", index=False)
    print(words[["lda_topic", "top_words"]].to_string(index=False))
    assigned = assign_topics(model, corpus, ids)
    assigned.to_parquet(f"{a.out}/lda_topics.parquet", index=False)
    print("\nc_v coherence (all papers):", round(coherence_cv(model, tokens, dct), 4))
    print("papers per topic:\n", assigned.lda_topic.value_counts().sort_index().to_string())
    print("saved to", a.out)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("tune")
    t.add_argument("--papers", required=True); t.add_argument("--out", default="results")
    t.add_argument("--ks", type=int, nargs="+", default=[10, 15, 20, 30])
    t.add_argument("--sample", type=int, default=20000, help="papers used for tuning")
    t.add_argument("--passes", type=int, default=5)
    t.set_defaults(fn=cmd_tune)
    f = sp.add_parser("final")
    f.add_argument("--papers", required=True); f.add_argument("--out", default="results")
    f.add_argument("--k", type=int, required=True); f.add_argument("--passes", type=int, default=8)
    f.set_defaults(fn=cmd_final)
    a = p.parse_args(); a.fn(a)
