"""Filtering and recommendation-ranking logic against the car catalog.

Pure functions — no Streamlit, no module-level globals — moved out of the
old app.py so the FastAPI API (and anything else) shares exactly one
implementation of "what does a filter dict mean against the catalog".
"""

import pandas as pd

CATEGORICAL_FEATURES = ["make", "bodytype", "Fuel Type", "Drivetrain", "Transmission Type"]

NUMERIC_FILTERS = {
    "max_price":      ("price",      "<="),
    "min_price":      ("price",      ">="),
    "min_mpg_comb":   ("mpg_comb",   ">="),
    "max_mpg_comb":   ("mpg_comb",   "<="),
    "min_hp":         ("hp",         ">="),
    "max_hp":         ("hp",         "<="),
    "min_year":       ("year",       ">="),
    "max_year":       ("year",       "<="),
    "min_cargo_cuft": ("cargo_cuft", ">="),
}

NUMERIC_BOUNDS = {
    "max_price":      (0,    500_000),
    "min_price":      (0,    500_000),
    "min_mpg_comb":   (0,    200),
    "max_mpg_comb":   (0,    200),
    "min_hp":         (0,    2000),
    "max_hp":         (0,    2000),
    "min_year":       (2001, 2024),
    "max_year":       (2001, 2024),
    "min_cargo_cuft": (0,    500),
}

DISPLAY_COLS = [
    "make", "model", "year", "trim", "bodytype", "Fuel Type", "Drivetrain",
    "price", "mpg_comb", "hp", "rating_reliability", "predicted_consumer_rating",
]


def apply_filters(prefs: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Filter the catalog DataFrame down to rows matching every key present in `prefs`."""
    if "fuel_type" in prefs:
        prefs = dict(prefs)
        prefs["Fuel Type"] = prefs.pop("fuel_type")

    mask = pd.Series(True, index=df.index)
    for key in CATEGORICAL_FEATURES:
        if key not in prefs:
            continue
        val = prefs[key]
        vals = [v.lower() for v in (val if isinstance(val, list) else [val])]
        mask &= df[key].str.lower().isin(vals)
    if "is_ev" in prefs:
        mask &= df["is_ev"] == int(bool(prefs["is_ev"]))
    if "luxury" in prefs:
        mask &= df["luxury"] == int(bool(prefs["luxury"]))
    for key, (col, op) in NUMERIC_FILTERS.items():
        if key not in prefs:
            continue
        try:
            val = float(prefs[key])
            mask &= (df[col] <= val) if op == "<=" else (df[col] >= val)
        except (TypeError, ValueError):
            pass
    return df[mask]


def get_recommendations(prefs: dict, df: pd.DataFrame, top_n: int = 5) -> dict:
    """Filter + rank the catalog, returning the same JSON-serializable shape the
    advisor tool and the /recommendations endpoint both hand back to callers."""
    results = apply_filters(prefs, df)
    if results.empty:
        return {"count": 0, "message": "No cars matched those filters. Try relaxing one or more criteria."}

    display_cols = [c for c in DISPLAY_COLS if c in results.columns]
    top = (
        results[display_cols]
        .sort_values("predicted_consumer_rating", ascending=False)
        .drop_duplicates(subset=["make", "model", "year"])
        .head(top_n)
        .reset_index(drop=True)
    )
    top.index += 1
    records = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for _, row in top.iterrows()]
    return {
        "count": len(records),
        "total_matched": len(results),
        "cars": records,
        "note": "Prices shown are new-car MSRP — used market prices will be significantly lower (often 30-60% less).",
    }
