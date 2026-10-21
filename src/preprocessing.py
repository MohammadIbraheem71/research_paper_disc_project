"""A1: stream the huge arXiv JSON file and keep only the papers we need."""
import json
import pandas as pd

from src.config import (CATEGORIES, YEAR_MIN, YEAR_MAX, MAX_PER_CATEGORY,
                        SEED, SAMPLE_SIZE)


def _squash(text: str) -> str:
    """Collapse newlines and repeated spaces into single spaces."""
    return " ".join(text.split())


def stream_filter(path, cats=CATEGORIES, y0=YEAR_MIN, y1=YEAR_MAX, limit=None):
    """Read the file line by line and return a DataFrame of matching papers.

    Keeps a paper only if its PRIMARY category (first one listed) is in `cats`
    and its first-version submission year is between y0 and y1.
    `limit` stops after that many lines (use it for quick tests).
    """
    cats = set(cats)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            cat_list = r["categories"].split()          # space-separated string
            if not cat_list or cat_list[0] not in cats:  # cheap check first
                continue
            d = pd.to_datetime(r["versions"][0]["created"], errors="coerce", utc=True)
            if pd.isna(d) or not (y0 <= d.year <= y1):
                continue
            rows.append({
                "id": r["id"],
                "title": _squash(r["title"]),
                "abstract": _squash(r["abstract"]),
                "categories": r["categories"],
                "primary_category": cat_list[0],
                "date": d.tz_localize(None),             # naive datetime
            })
    return pd.DataFrame(rows)


def cap_per_category(df, max_per_cat=MAX_PER_CATEGORY, seed=SEED):
    """Randomly keep at most `max_per_cat` papers per primary category."""
    if max_per_cat is None:
        return df
    parts = [g.sample(min(len(g), max_per_cat), random_state=seed)
             for _, g in df.groupby("primary_category")]
    return pd.concat(parts).sort_values("id").reset_index(drop=True)


def make_sample(df, n=SAMPLE_SIZE, seed=SEED):
    return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)


def summarize(df):
    print("rows:", len(df))
    print("duplicate ids:", df.id.duplicated().sum())
    print("date range:", df.date.min(), "->", df.date.max())
    print(df.primary_category.value_counts())
    print(df.date.dt.year.value_counts().sort_index())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="arxiv-metadata-oai-snapshot.json")
    p.add_argument("--out", default="data", help="output folder")
    p.add_argument("--limit", type=int, default=None, help="only read this many lines (testing)")
    a = p.parse_args()

    import os
    os.makedirs(a.out, exist_ok=True)
    df = stream_filter(a.path, limit=a.limit)
    summarize(df)
    df = cap_per_category(df)
    df = df.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    df.to_parquet(f"{a.out}/papers_raw.parquet", index=False)
    make_sample(df).to_parquet(f"{a.out}/papers_sample_raw.parquet", index=False)
    print("saved", len(df), "papers and a sample of", min(SAMPLE_SIZE, len(df)))
