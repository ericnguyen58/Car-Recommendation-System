"""
split_cars.py
-------------
Splits all_cars_clean.csv into two focused datasets.

  Input:   data/processed/all_cars_clean.csv
           data/processed/all_reviews_clean.csv   (for review_id FK lookup)
  Output:  data/final/cars/car_stats.csv
           data/final/cars/car_features.csv

car_id   – primary key (trim-level, unique per row)
review_id – foreign key into review_summary / review_ratings (model-level)
Feature tier/group metadata is loaded from schema/feature_categories.json.
"""

import json
import sys
import pandas as pd
from pathlib import Path

ROOT       = Path(__file__).parent.parent
PROCESSED  = ROOT / "data" / "processed"
OUT_DIR    = ROOT / "data" / "final" / "cars"
SCHEMA_DIR = ROOT / "schema"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure the repo root is on sys.path so `scripts` is importable as a package
# whether this file is run directly or via main.py.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.split_reviews import build_review_fk

# ── column definitions ────────────────────────────────────────────────────────

STATS_COLS = [
    "car_id",
    # identity
    "make", "model", "year", "bodytype", "doors", "trim",
    # pricing
    "price",
    # fuel economy
    "mpg_city", "mpg_hwy", "mpg_comb",
    # powertrain
    "Fuel Type", "hp", "Engine", "torque_lbft",
    "Recommended Fuel", "Transmission Type", "Drivetrain",
    # performance
    "0 - 60",
    # dimensions & capacity
    "curb_weight_lbs", "cargo_cuft",
    "Fuel Capacity", "Wheel Base", "Overall Length",
    "Front Head Room", "Front Leg Room", "Front Shoulder Room",
    "Width with mirrors", "Turning Diameter",
]

FEATURE_COLS = [
    "car_id",
    # safety
    "Child Seat Anchors", "Child Door Locks",
    "Traction Control", "Stability Control", "Hill Start Assist",
    "Blind-Spot Alert", "Collision Warning System",
    # connectivity
    "Bluetooth Wireless Technology", "Hands Free Phone",
    "Bluetooth Streaming Audio", "Satellite Radio", "Smartphone Interface",
    "Navigation System", "Voice Recognition System",
    "Internet Access", "Real-Time Traffic Information", "Premium Radio",
    # comfort & convenience
    "Cruise Control", "Remote Keyless Entry", "Remote Engine Start",
    "Power Windows", "Power Outlet", "Rear Window Defroster",
    "Steering Wheel Controls", "Tilt Steering Wheel",
    "Tilt/Telescoping Steering Wheel",
    "Cup Holder Count", "Folding Rear Seat",
    "Power Driver's Seat", "Dual Power Front Seats",
    "Heated Mirrors", "Alarm System",
    # interior premium
    "Leather Seats", "Leather-Wrapped Steering Wheel",
    "Remote Control Liftgate/Trunk Release",
    "Power Folding Exterior Mirrors", "Rain Sensing Windshield Wipers",
    # exterior
    "Alloy Wheels", "Fog Lights", "Rear Spoiler",
    # drivetrain advanced
    "Dual-Clutch Automatic Transmission",
]


def split(src: Path, reviews_src: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(src, low_memory=False)

    # Build (make, model, year) → review_id FK from the reviews dataset
    fk_map = build_review_fk(reviews_src)
    df = df.merge(fk_map, on=["make", "model", "year"], how="left")

    # Place review_id as the second column (after car_id)
    cols = df.columns.tolist()
    cols.remove("review_id")
    cols.insert(1, "review_id")
    df = df[cols]

    stats    = df[["review_id"] + STATS_COLS].copy()
    features = df[["review_id"] + FEATURE_COLS].copy()

    # Cast feature columns to Int8 (they read back as float64 from CSV)
    feat_data_cols = [c for c in FEATURE_COLS if c != "car_id"]
    for col in feat_data_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").astype("Int8")

    return stats, features


def main():
    src          = PROCESSED / "all_cars_clean.csv"
    reviews_src  = PROCESSED / "all_reviews_clean.csv"

    print(f"Reading {src.name} …")
    stats, features = split(src, reviews_src)

    out_stats    = OUT_DIR / "car_stats.csv"
    out_features = OUT_DIR / "car_features.csv"

    stats.to_csv(out_stats, index=False)
    features.to_csv(out_features, index=False)

    print(f"  → car_stats.csv    {len(stats):,} rows × {len(stats.columns)} cols")
    print(f"  → car_features.csv {len(features):,} rows × {len(features.columns)} cols")

    # ── feature tier summary ──────────────────────────────────────────────────
    cats_path = SCHEMA_DIR / "feature_categories.json"
    with open(cats_path) as f:
        cats = json.load(f)

    lookup = cats["flat_lookup"]
    feat_data_cols = [c for c in FEATURE_COLS if c != "car_id"]

    print("\n── Feature tier summary ──")
    for tier in ("standard", "luxury"):
        members = [c for c in feat_data_cols if lookup.get(c, {}).get("tier") == tier]
        print(f"  {tier.upper():8s} ({len(members):2d} features): "
              + ", ".join(members[:5]) + (" …" if len(members) > 5 else ""))

    print("\n── Null counts in car_features.csv ──")
    nulls = features[feat_data_cols].isnull().sum()
    print(nulls[nulls > 0].to_string())


if __name__ == "__main__":
    main()
