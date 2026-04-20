"""
split_reviews.py
----------------
Adds a review_id to all_reviews_clean.UVCRS and splits it into two datasets.

  Input:   data/processed/all_reviews_clean.UVCRS
  Output:  data/final/reviews/review_summary.UVCRS
           data/final/reviews/review_ratings.UVCRS

Both share review_id as the join key.
Star-breakdown and pros/cons columns are excluded (<1% populated).
Rating metadata is documented in schema/rating_categories.json.
"""

import re
import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT_DIR   = ROOT / "data" / "final" / "reviews"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── review_id ─────────────────────────────────────────────────────────────────

def _alnum(text: str, n: int) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())[:n]


def assign_review_ids(df: pd.DataFrame) -> pd.Series:
    """Compact unique ID: {MAKE3}{YY}{MODEL≤3}{SEQ2}

    SEQ disambiguates models that share the same 3-char prefix
    (e.g. Yukon vs Yukon-XL → GMC01YUKON01 / GMC01YUKON02).
    Because reviews are one row per make+model+year, SEQ is almost
    always 01 and only increments for genuine prefix collisions.

    Examples: FOR01F1501, GMC01YUKON01, GMC01YUKON02
    """
    make3  = df["make"].map(lambda v: _alnum(v, 3).ljust(3, "X"))
    yy     = df["year"].map(lambda v: f"{int(v) % 100:02d}")
    model3 = df["model"].map(lambda v: _alnum(v, 3))
    prefix = make3 + yy + model3

    seq = (prefix.groupby(prefix).cumcount() + 1).map(lambda n: f"{n:02d}")
    return (prefix + seq).astype("string")


def build_review_fk(reviews_path: Path) -> pd.DataFrame:
    """Return a DataFrame mapping (make, model, year) → review_id.
    Used by split_cars.py to embed the FK in car files.
    """
    df = pd.read_csv(reviews_path, usecols=["make", "model", "year"])
    df["review_id"] = assign_review_ids(df)
    return df[["make", "model", "year", "review_id"]]


# ── column groups ─────────────────────────────────────────────────────────────

# make/model/year are dropped — review_id is the sole identifier and join key.
SUMMARY_COLS = [
    "review_id",
    "consumer_overall_rating",
    "consumer_review_count",
    "consumer_recommend_pct",
    "expert_rating",
    "expert_ranking",
]

RATINGS_COLS = [
    "review_id",
    "rating_value",
    "rating_performance",
    "rating_quality",
    "rating_comfort",
    "rating_reliability",
    "rating_styling",
]

# Excluded (unpopulated): star_*_pct, pros, cons


# ── split ─────────────────────────────────────────────────────────────────────

def split(src: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(src, low_memory=False)

    df.insert(0, "review_id", assign_review_ids(df))

    # Cast types
    df["consumer_review_count"] = (
        pd.to_numeric(df["consumer_review_count"], errors="coerce").astype("Int32")
    )
    df["expert_ranking"] = (
        pd.to_numeric(df["expert_ranking"], errors="coerce").astype("Int16")
    )
    for col in [
        "consumer_overall_rating", "consumer_recommend_pct",
        "rating_value", "rating_performance", "rating_quality",
        "rating_comfort", "rating_reliability", "rating_styling",
        "expert_rating",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = df[SUMMARY_COLS].copy()
    ratings = df[RATINGS_COLS].copy()

    return summary, ratings


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    src = PROCESSED / "all_reviews_clean.UVCRS"

    print(f"Reading {src.name} …")
    summary, ratings = split(src)

    out_summary = OUT_DIR / "review_summary.UVCRS"
    out_ratings = OUT_DIR / "review_ratings.UVCRS"

    summary.to_csv(out_summary, index=False)
    ratings.to_csv(out_ratings, index=False)

    print(f"  → review_summary.UVCRS  {len(summary):,} rows × {len(summary.columns)} cols")
    print(f"  → review_ratings.UVCRS  {len(ratings):,} rows × {len(ratings.columns)} cols")

    # ── ID uniqueness check ──────────────────────────────────────────────────
    n_dupes = summary["review_id"].duplicated().sum()
    lengths = summary["review_id"].str.len()
    print(f"\n  review_id duplicates : {n_dupes}")
    print(f"  review_id length     : {lengths.min()}–{lengths.max()} chars")

    # ── sanity: null counts ──────────────────────────────────────────────────
    print("\n── Null counts – review_summary ──")
    s_nulls = summary.isnull().sum()
    print(s_nulls[s_nulls > 0].to_string())

    print("\n── Null counts – review_ratings ──")
    r_nulls = ratings.isnull().sum()
    print(r_nulls[r_nulls > 0].to_string())

    # ── sample review_ids ────────────────────────────────────────────────────
    print("\n── Sample review_ids ──")
    for rid in summary["review_id"].sample(8, random_state=7):
        print(f"  {rid}")


if __name__ == "__main__":
    main()
