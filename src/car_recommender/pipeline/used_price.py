"""car_recommender.pipeline.used_price — used-market price estimates via KBB.

Optional pipeline stage, off by default (see runner.py's OPTIONAL_STAGES) —
network/browser dependent and, unlike the rest of the pipeline, cannot be
verified from a sandboxed dev environment. Treat this as a starting point:
run it yourself, inspect the output, and iterate.

Currently scrapes whatever `CARS` / `YEARS` are configured in
scraping/kbb.py — a small hand-picked sample, not the full catalog.
Expanding coverage to every make/model/year in the catalog is a deliberate
follow-up (real request volume against a live site), not something to turn
on silently by default.

Output: data/final/cars/used_price.csv (make, model, year, used_price_est)
— export_final.py joins this in on (make, model, year) when the file
exists, adding `used_price_est` / `used_price_missing` columns; the rest of
the pipeline runs unaffected when it's absent.

Usage:
    python -m car_recommender.pipeline.used_price
    python -m car_recommender.pipeline.runner --only scrape-used-price
"""

import pandas as pd

from car_recommender.core.paths import PROJECT_ROOT
from car_recommender.scraping.kbb import kbb_worker

OUT = PROJECT_ROOT / "data" / "final" / "cars" / "used_price.csv"


def _clean_price(raw: str | None) -> float | None:
    """KBB price text like '$24,500' -> 24500.0; unparsable -> None."""
    if not raw:
        return None
    digits = "".join(ch for ch in raw if ch.isdigit() or ch == ".")
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


def build_used_price_table() -> pd.DataFrame:
    raw = kbb_worker()
    if raw.empty:
        return pd.DataFrame(columns=["make", "model", "year", "used_price_est"])

    raw = raw.copy()
    raw["used_price_est"] = raw["Price"].apply(_clean_price)
    raw["make"] = raw["Brand"]
    raw["model"] = raw["Model"]
    raw["year"] = raw["Year"]

    # Multiple styles/trims per make+model+year get scraped — export_final.py
    # joins one price onto every trim of that make+model+year, so collapse
    # to a single point estimate (median) per group here.
    return (
        raw.dropna(subset=["used_price_est"])
        .groupby(["make", "model", "year"], as_index=False)["used_price_est"]
        .median()
    )


def main() -> None:
    table = build_used_price_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"Saved {len(table):,} make/model/year used-price estimates -> {OUT}")


if __name__ == "__main__":
    main()
