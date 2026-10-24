"""A2: clean the raw filtered papers and build the final papers.parquet.

Input : papers_raw.parquet  (from A1)
Output: papers.parquet and papers_sample.parquet with the agreed contract columns:
        id, title, abstract, text_raw, text_lemma, primary_category,
        categories, date, year, month, quarter
"""
import re
import pandas as pd

from src.config import SEED, SAMPLE_SIZE

MIN_ABSTRACT_WORDS = 30

# --- generic academic words that carry no topic information (extra stop words) ---
EXTRA_STOP = {
    "paper", "propose", "proposed", "present", "show", "result", "method", "approach",
    "use", "using", "used", "based", "novel", "work", "study", "new", "also", "however",
    "demonstrate", "achieve", "provide", "introduce", "address", "problem", "task",
    "different", "recent", "existing", "state", "art", "experiment", "experimental",
    "performance", "outperform", "improve", "improvement",
}


_URL = re.compile(r"https?://\S+|www\.\S+")
_MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.DOTALL)           # $...$ and $$...$$
_CMD_ARG = re.compile(r"\\[a-zA-Z]+\*?\{([^{}]*)\}")              # \textbf{x} -> x
_CMD = re.compile(r"\\[a-zA-Z]+\*?")                             # leftover \command
_JUNK = re.compile(r"[{}\\^_~]")                                  # stray LaTeX characters
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light cleaning: remove LaTeX, URLs and stray symbols. Keeps case and punctuation
    so the text still reads naturally (good for embeddings and for the LLM)."""
    text = _URL.sub(" ", text)
    text = _MATH.sub(" ", text)
    text = _CMD_ARG.sub(r"\1", text)
    text = _CMD.sub(" ", text)
    text = _JUNK.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def lemmatize_all(texts, batch_size=1000):
    """Heavy cleaning for TF-IDF / LDA: lowercase, letters only, no stop words, lemmatized."""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.max_length = 2_000_000
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        toks = []
        for t in doc:
            if not t.is_alpha or t.is_stop:
                continue
            lemma = t.lemma_.lower()
            if len(lemma) < 3 or lemma in EXTRA_STOP:
                continue
            toks.append(lemma)
        out.append(" ".join(toks))
    return out


def build_papers(raw: pd.DataFrame) -> pd.DataFrame:
    n0 = len(raw)
    df = raw.copy()

    # 1. drop missing / duplicate / withdrawn / very short
    df = df.dropna(subset=["id", "title", "abstract", "date", "primary_category"])
    df = df.drop_duplicates("id")
    df["title"] = df["title"].map(clean_text)                          # clean BEFORE comparing titles
    df["_tn"] = df["title"].map(_norm_title)
    df = df.drop_duplicates("_tn").drop(columns="_tn")                 # near-identical titles
    df = df[~df["abstract"].str.contains(r"\b(?:withdrawn|has been withdrawn)\b", case=False, regex=True)
            | (df["abstract"].str.split().str.len() > 80)]             # withdrawal notices are short
    print(f"after dropping missing/duplicates/withdrawn: {len(df)} (from {n0})")

    # 2. text_raw = title + abstract, lightly cleaned
    df["abstract"] = df["abstract"].map(clean_text)
    df["text_raw"] = df["title"] + ". " + df["abstract"]
    df = df[df["abstract"].str.split().str.len() >= MIN_ABSTRACT_WORDS]
    print(f"after dropping abstracts under {MIN_ABSTRACT_WORDS} words: {len(df)}")

    # 3. dates
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["quarter"] = df["year"].astype(str) + "Q" + df["date"].dt.quarter.astype(str)

    # 4. text_lemma
    df = df.reset_index(drop=True)
    print("lemmatizing (a few minutes for ~100k papers)...")
    df["text_lemma"] = lemmatize_all(df["text_raw"].tolist())
    df = df[df["text_lemma"].str.split().str.len() >= 5]               # drop papers with almost no content words

    # 5. final order. Sort ONCE by id and never re-sort afterwards
    #    (embeddings and everything else are aligned to this row order).
    cols = ["id", "title", "abstract", "text_raw", "text_lemma", "primary_category",
            "categories", "date", "year", "month", "quarter"]
    df = df[cols].sort_values("id").reset_index(drop=True)

    assert df[cols].isna().sum().sum() == 0, "nulls found in key columns"
    assert df["id"].is_unique
    return df


def report(df: pd.DataFrame):
    print("\nFINAL ROWS:", len(df))
    print(df.primary_category.value_counts().to_string())
    print(df.quarter.value_counts().sort_index().to_string())
    print("avg words text_raw:", round(df.text_raw.str.split().str.len().mean(), 1),
          "| avg words text_lemma:", round(df.text_lemma.str.split().str.len().mean(), 1))


if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True, help="papers_raw.parquet from A1")
    p.add_argument("--out", default="data")
    p.add_argument("--limit", type=int, default=None, help="only process this many rows (testing)")
    a = p.parse_args()

    raw = pd.read_parquet(a.inp)
    if a.limit:
        raw = raw.head(a.limit)
    papers = build_papers(raw)
    report(papers)
    os.makedirs(a.out, exist_ok=True)
    papers.to_parquet(f"{a.out}/papers.parquet", index=False)
    papers.sample(min(SAMPLE_SIZE, len(papers)), random_state=SEED).sort_values("id") \
          .reset_index(drop=True).to_parquet(f"{a.out}/papers_sample.parquet", index=False)
    print("saved papers.parquet and papers_sample.parquet in", a.out)
