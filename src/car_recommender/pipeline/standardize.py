"""
standardize.py
--------------
Loads raw CSVs, applies type conversions and value standardizations
defined in schema/schema.json, and writes cleaned outputs.

  Input:   data/raw/all_cars.csv
           data/raw/all_reviews.csv
  Output:  data/processed/all_cars_clean.csv
           data/processed/all_reviews_clean.csv
"""

import re
from pathlib import Path

import pandas as pd

from car_recommender.core.paths import PROJECT_ROOT

ROOT      = PROJECT_ROOT
RAW_DIR   = ROOT / "data" / "raw"
OUT_DIR   = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

NA_STRINGS = {"not available", "not available.", "n/a", "na", ""}

def is_na_string(val: str) -> bool:
    return val.strip().lower() in NA_STRINGS


def parse_numeric_unit(series: pd.Series, unit_word: str) -> pd.Series:
    """Strip a trailing unit word (e.g. 'inches', 'gallons', 'feet', 'seconds')
    and return a float Series.  Handles 'Not Available' → NaN."""
    pattern = re.compile(rf"\s*{re.escape(unit_word)}\s*$", re.IGNORECASE)

    def _convert(val):
        if pd.isna(val):
            return float("nan")
        s = str(val).strip()
        if is_na_string(s):
            return float("nan")
        s = pattern.sub("", s).strip()
        try:
            return float(s)
        except ValueError:
            return float("nan")

    return series.map(_convert)


def parse_zero_to_sixty(series: pd.Series) -> pd.Series:
    """'5.8 seconds' → 5.8;  'N/A' / 'Not Available' → NaN."""
    return parse_numeric_unit(series, "seconds")


def recode_ordinal(series: pd.Series, encoding: dict) -> pd.Series:
    """Map feature availability strings to ordinal ints via a case-insensitive
    lookup.  Unrecognised values become NaN."""
    lower_enc = {k.lower(): v for k, v in encoding.items()}

    def _map(val):
        if pd.isna(val):
            return pd.NA
        s = str(val).strip().lower()
        if s in NA_STRINGS:
            return lower_enc.get("not available", pd.NA)
        return lower_enc.get(s, pd.NA)

    return series.map(_map).astype("Int8")  # nullable integer


# ── car ID ───────────────────────────────────────────────────────────────────

def _alnum(text: str, n: int) -> str:
    """Return up to n uppercase alphanumeric characters from text."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())[:n]


def assign_car_ids(df: pd.DataFrame) -> pd.Series:
    """Build compact 10-character IDs: {MAKE3}{YY}{MODEL≤3}{SEQ2}

    MAKE3   – first 3 alphanumeric chars of brand  (e.g. Ford → FOR)
    YY      – last 2 digits of model year           (e.g. 2001 → 01)
    MODEL≤3 – first 1-3 alphanumeric chars of model (e.g. F-150 → F15, S5 → S5)
    SEQ2    – 2-digit counter (01-99) within each MAKE3+YY+MODEL≤3 group,
              ordered by the original row position

    Total length: 7-10 characters.
    Examples: FOR01F1501, MER21GLC03, AUD22S501
    """
    make3  = df["make"].map(lambda v: _alnum(v, 3).ljust(3, "X"))
    yy     = df["year"].map(lambda v: f"{int(v) % 100:02d}")
    model3 = df["model"].map(lambda v: _alnum(v, 3))
    prefix = make3 + yy + model3

    # Assign a per-group sequential counter (1-based, zero-padded to 2 digits)
    seq = (
        prefix.groupby(prefix).cumcount() + 1
    ).map(lambda n: f"{n:02d}")

    return (prefix + seq).astype("string")


def parse_expert_ranking(series: pd.Series) -> pd.Series:
    """'#1 in Best Full-Size Pickups for 2021' → 1;  else NaN."""
    def _extract(val):
        if pd.isna(val):
            return pd.NA
        m = re.match(r"#(\d+)", str(val).strip())
        return int(m.group(1)) if m else pd.NA

    return series.map(_extract).astype("Int16")


# ── bodytype + doors extractor ───────────────────────────────────────────────

# Rules are checked in order; first match wins.
# Longer / more specific phrases must appear before shorter ones.
_BODY_RULES: list[tuple[str, str]] = [
    # Multi-word phrases first
    (r"Gran\s+Coupe",        "Sedan"),        # BMW Gran Coupe is a 4-door sedan
    (r"Sport\s+Utility",     "SUV"),
    (r"Cargo\s+Van",         "Van"),
    (r"Passenger\s+Van",     "Minivan"),
    (r"Sport\s+Wagon",       "Wagon"),
    (r"Crew\s+Cab",          "Pickup"),
    # Convertible synonyms
    (r"Cabriolet",           "Convertible"),
    (r"Roadster",            "Convertible"),
    (r"Spyder",              "Convertible"),
    (r"Convertible",         "Convertible"),
    # Coupe synonyms
    (r"Hard\s*Top",          "Coupe"),
    (r"Fastback",            "Coupe"),
    (r"Targa",               "Coupe"),
    (r"\bCpe\b",             "Coupe"),   # abbreviated "Cpe"
    (r"Coupe",               "Coupe"),
    # Remaining single words
    (r"Hatchback",           "Hatchback"),
    (r"Sedan",               "Sedan"),
    (r"Wagon",               "Wagon"),
    (r"Minivan",             "Minivan"),
    (r"\bVan\b",             "Van"),
    (r"\bSUV\b",             "SUV"),
    (r"Pickup",              "Pickup"),
    (r"\bCab\b",             "Pickup"),
    (r"XtraCab",             "Pickup"),  # Toyota XtraCab
]

_DOOR_PAT = re.compile(r"\b(\d)D\b", re.IGNORECASE)


def _derive_body_from_trim(trim: str):
    """Return the canonical bodytype derived from a trim string, or None."""
    for pattern, body in _BODY_RULES:
        if re.search(pattern, trim, re.IGNORECASE):
            return body
    return None


# ── trim cleaner ─────────────────────────────────────────────────────────────

# Bed-size suffixes (pickup trucks): "5 ft", "6 3/4 ft", "8 ft", etc.
_BED_SIZE_PAT = re.compile(
    r"\s+\d+(?:\s+\d+/\d+)?\s*ft\b.*$", re.IGNORECASE
)

# Body-type tokens to strip from the end of the cleaned trim string.
# Multi-word phrases are listed before single words (order matters).
# NOTE: "Targa", "Cab", "Crew Cab" are intentionally excluded — they are
# cab/configuration names that carry useful trim-level information.
_STRIP_BODY_PAT = re.compile(
    r"\s*\b(?:"
    r"Gran\s+Coupe"
    r"|Sport\s+Utility"
    r"|Cargo\s+Van"
    r"|Passenger\s+Van"
    r"|Sport\s+Wagon"
    r"|Cabriolet|Roadster|Convertible|Spyder"
    r"|Hard\s*Top|Fastback|\bCpe\b|Coupe"
    r"|Hatchback|Minivan|Pickup|Sedan|Wagon|\bVan\b|\bSUV\b"
    r")\b\s*$",
    re.IGNORECASE,
)

# Door token: single digit followed by D (e.g. 2D, 4D) — NOT "85D" (Tesla)
_STRIP_DOOR_PAT = re.compile(r"\s*\b\d[Dd]\b", re.IGNORECASE)

# Stray trailing parenthetical year tags like "(2015.5)"
_YEAR_TAG_PAT = re.compile(r"\s*\(\d{4}(?:\.\d)?\)\s*$")


def clean_trim(trim_val):
    """Remove bodytype words, door count, bed size, and stray year tags
    from a trim string, leaving only the model-specific trim name.

    Examples:
      'Equinox RS Sport Utility 4D'       → 'Equinox RS'
      'Tacoma Double Cab SR Pickup 4D 5 ft' → 'Tacoma Double Cab SR'
      'Targa 4 Coupe 2D'                  → 'Targa 4'
      'Model S 85D Sedan 4D'              → 'Model S 85D'
      'Series 430i Gran Coupe 4D'         → 'Series 430i'
    """
    if pd.isna(trim_val):
        return pd.NA
    s = str(trim_val).strip()

    # 1. Remove bed size and any trailing content after it
    s = _BED_SIZE_PAT.sub("", s).strip()

    # 2. Remove year tags like "(2015.5)"
    s = _YEAR_TAG_PAT.sub("", s).strip()

    # 3. Remove all door-count tokens (handles duplicates like "4D ... 4D")
    s = _STRIP_DOOR_PAT.sub("", s).strip()

    # 4. Strip body-type words from the end, repeatedly until stable
    prev = None
    while prev != s:
        prev = s
        s = _STRIP_BODY_PAT.sub("", s).strip()

    return s if s else "Base"


def extract_body_and_doors(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich the DataFrame with two new columns:

    doors     – number of doors parsed from trim (e.g. '4D' → 4); Int8
    bodytype  – 'Default' rows are replaced with the bodytype derived from the
                trim string; explicitly set non-Default rows are kept as-is.

    Returns the same DataFrame with columns modified in-place.
    """
    # ── doors ────────────────────────────────────────────────────────────────
    df["doors"] = (
        df["trim"]
        .str.extract(_DOOR_PAT, expand=False)
        .astype("Int8")
    )

    # ── bodytype from trim ────────────────────────────────────────────────────
    derived = df["trim"].map(
        lambda v: _derive_body_from_trim(str(v)) if pd.notna(v) else None
    )

    # Only overwrite rows where bodytype is 'Default' (or missing)
    # Work on plain object dtype first, then cast to category at the end
    body_obj = df["bodytype"].astype(object)
    is_default = body_obj.str.lower().isin({"default", "nan", "none", "<na>"})
    body_obj[is_default] = derived[is_default]
    df["bodytype"] = body_obj.astype("category")
    return df


# ── all_cars.csv ─────────────────────────────────────────────────────────────

def standardize_cars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # ── identifiers / simple numerics (already correct dtype or just cast) ──
    df["make"]  = df["make"].astype("string")
    df["model"] = df["model"].astype("string")
    df["year"]  = df["year"].astype("Int16")
    df["bodytype"] = df["bodytype"].astype("category")
    df["trim"]  = df["trim"].astype("string")

    for col in ["price", "mpg_city", "mpg_hwy", "mpg_comb",
                "hp", "torque_lbft", "cargo_cuft", "curb_weight_lbs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Fuel Type – recode icon names to plain labels ──
    fuel_recode = {"ElectricLeafIcon": "Electric", "HybridLeafIcon": "Hybrid"}
    df["Fuel Type"] = (
        df["Fuel Type"]
        .replace(fuel_recode)
        .where(df["Fuel Type"].notna())
        .astype("category")
    )

    # ── 0-60 ──
    df["0 - 60"] = parse_zero_to_sixty(df["0 - 60"])

    # ── categorical fields with 'Not Available' → NaN ──
    for col in ["Drivetrain", "Transmission Type", "Recommended Fuel"]:
        df[col] = (
            df[col]
            .apply(lambda v: pd.NA if pd.notna(v) and is_na_string(str(v)) else v)
            .astype("category")
        )

    # ── measurement strings → float ──
    for col in [
        "Wheel Base", "Overall Length",
        "Front Head Room", "Front Leg Room", "Front Shoulder Room",
        "Width with mirrors",
    ]:
        df[col] = parse_numeric_unit(df[col], "inches")

    df["Fuel Capacity"]    = parse_numeric_unit(df["Fuel Capacity"],  "gallons")
    df["Turning Diameter"] = parse_numeric_unit(df["Turning Diameter"], "feet")

    # ── feature availability columns → ordinal int ──
    ORDINAL_ENCODING = {
        "Not Available": 0,
        "Not available": 0,
        "Not Required":  0,
        "Optional":      1,
        "Standard":      2,
    }
    feature_cols = [
        "Blind-Spot Alert", "Collision Warning System", "Child Seat Anchors",
        "Child Door Locks", "Traction Control", "Bluetooth Wireless Technology",
        "Cruise Control", "Remote Keyless Entry", "Remote Engine Start",
        "Smartphone Interface", "Internet Access", "Navigation System",
        "Voice Recognition System", "Real-Time Traffic Information",
        "Hands Free Phone", "Premium Radio", "Bluetooth Streaming Audio",
        "Satellite Radio", "Remote Control Liftgate/Trunk Release",
        "Heated Mirrors", "Leather Seats", "Folding Rear Seat",
        "Dual Power Front Seats", "Power Driver's Seat",
        "Leather-Wrapped Steering Wheel", "Power Outlet", "Power Windows",
        "Rear Window Defroster", "Steering Wheel Controls", "Tilt Steering Wheel",
        "Tilt/Telescoping Steering Wheel", "Cup Holder Count", "Alloy Wheels",
        "Fog Lights", "Power Folding Exterior Mirrors", "Rear Spoiler",
        "Rain Sensing Windshield Wipers", "Alarm System",
        "Dual-Clutch Automatic Transmission", "Hill Start Assist",
        "Stability Control",
    ]
    for col in feature_cols:
        if col in df.columns:
            df[col] = recode_ordinal(df[col], ORDINAL_ENCODING)

    # ── Engine stays as free-text string ──
    df["Engine"] = df["Engine"].astype("string")

    # ── bodytype fill + doors extraction (uses raw trim) ──────────────────────
    df = extract_body_and_doors(df)

    # ── strip bodytype / door tokens from trim ────────────────────────────────
    df["trim"] = df["trim"].map(clean_trim).astype("string")

    # ── car_id – generated last so all source columns are already clean ──
    df.insert(0, "car_id", assign_car_ids(df))

    return df


# ── all_reviews.csv ──────────────────────────────────────────────────────────

def standardize_reviews(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    df["make"]  = df["make"].astype("string")
    df["model"] = df["model"].astype("string")
    df["year"]  = df["year"].astype("Int16")

    float_cols = [
        "consumer_overall_rating", "consumer_recommend_pct",
        "star_5_pct", "star_4_pct", "star_3_pct", "star_2_pct", "star_1_pct",
        "rating_value", "rating_performance", "rating_quality",
        "rating_comfort", "rating_reliability", "rating_styling",
        "expert_rating",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["consumer_review_count"] = (
        pd.to_numeric(df["consumer_review_count"], errors="coerce")
        .astype("Int32")
    )

    df["expert_ranking"] = parse_expert_ranking(df["expert_ranking"])

    # pros / cons are entirely empty – keep as nullable string
    for col in ["pros", "cons"]:
        df[col] = df[col].astype("string")

    return df


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    cars_path    = RAW_DIR / "all_cars (1).csv"
    reviews_path = RAW_DIR / "all_reviews.csv"

    print("Loading and standardizing all_cars(1).csv …")
    cars = standardize_cars(cars_path)
    out_cars = OUT_DIR / "all_cars_clean.csv"
    cars.to_csv(out_cars, index=False)
    print(f"  → saved {out_cars}  ({len(cars):,} rows × {len(cars.columns)} cols)")

    print("Loading and standardizing all_reviews.csv …")
    reviews = standardize_reviews(reviews_path)
    out_reviews = OUT_DIR / "all_reviews_clean.csv"
    reviews.to_csv(out_reviews, index=False)
    print(f"  → saved {out_reviews}  ({len(reviews):,} rows × {len(reviews.columns)} cols)")

    # ── quick sanity report ──────────────────────────────────────────────────
    print("\n── all_cars_clean dtypes ──")
    print(cars.dtypes.to_string())
    print("\n── all_reviews_clean dtypes ──")
    print(reviews.dtypes.to_string())

    print("\n── null counts (all_cars) ──")
    null_summary = cars.isnull().sum()
    print(null_summary[null_summary > 0].to_string())

    print("\n── null counts (all_reviews) ──")
    null_summary_r = reviews.isnull().sum()
    print(null_summary_r[null_summary_r > 0].to_string())


if __name__ == "__main__":
    main()
