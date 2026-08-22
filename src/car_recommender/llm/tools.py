"""Tool schema and input validation for the advisor's get_car_recommendations tool.

Moved from app.py. The validation step is the security boundary between
whatever Claude decides to put in a tool call and the DataFrame — every
field is allowlisted or bounds-checked before it touches `filters.apply_filters`.
"""

import pandas as pd

from car_recommender.ml.filters import CATEGORICAL_FEATURES, NUMERIC_BOUNDS


def build_known_categorical(df: pd.DataFrame) -> dict[str, set[str]]:
    """Lower-cased allowlists of valid categorical values, built once from the catalog."""
    return {col: set(df[col].dropna().astype(str).str.lower().unique()) for col in CATEGORICAL_FEATURES}


def validate_tool_input(raw: dict, known_categorical: dict[str, set[str]]) -> dict:
    """Sanitize and bounds-check every field from Claude before it touches the DataFrame."""
    safe: dict = {}

    for key, df_col in [("make", "make"), ("bodytype", "bodytype"), ("Drivetrain", "Drivetrain")]:
        if key in raw:
            vals = raw[key] if isinstance(raw[key], list) else [raw[key]]
            allowed = known_categorical[df_col]
            safe[key] = [str(v)[:64] for v in vals if isinstance(v, str) and v.lower() in allowed]

    if "fuel_type" in raw:
        vals = raw["fuel_type"] if isinstance(raw["fuel_type"], list) else [raw["fuel_type"]]
        allowed = known_categorical["Fuel Type"]
        safe["fuel_type"] = [str(v)[:64] for v in vals if isinstance(v, str) and v.lower() in allowed]

    for key in ("is_ev", "luxury"):
        if key in raw:
            safe[key] = bool(raw[key])

    for key, (lo, hi) in NUMERIC_BOUNDS.items():
        if key in raw:
            try:
                safe[key] = max(lo, min(hi, float(raw[key])))
            except (TypeError, ValueError):
                pass

    if "top_n" in raw:
        try:
            safe["top_n"] = max(1, min(20, int(raw["top_n"])))
        except (TypeError, ValueError):
            safe["top_n"] = 5

    return safe


ADVISOR_TOOLS = [
    {
        "name": "get_car_recommendations",
        "description": (
            "Search the car catalog and return the top-N cars ranked by predicted buyer "
            "satisfaction. Call this whenever the user wants recommendations or mentions "
            "any filtering criteria such as budget, body type, fuel economy, or brand."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "make": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more car makes to include. Omit to search all makes.",
                },
                "bodytype": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more body types: Sedan, SUV, Pickup, Hatchback, Coupe, Convertible, Wagon, Minivan, Van.",
                },
                "min_year": {"type": "integer", "description": "Minimum model year."},
                "max_year": {"type": "integer", "description": "Maximum model year."},
                "max_price": {"type": "number", "description": "Maximum MSRP in USD (new car price — quality proxy only)."},
                "min_price": {"type": "number", "description": "Minimum MSRP in USD."},
                "min_mpg_comb": {"type": "number", "description": "Minimum combined MPG."},
                "min_hp": {"type": "number", "description": "Minimum horsepower."},
                "Drivetrain": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more drivetrains: FWD, AWD, RWD, 4WD, 2WD.",
                },
                "fuel_type": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more fuel types, e.g. ['Gasoline'], ['Hybrid'], ['Electric'].",
                },
                "is_ev": {"type": "boolean", "description": "Set true to show only battery-electric vehicles."},
                "luxury": {"type": "boolean", "description": "Set true to show only luxury/premium brand vehicles."},
                "top_n": {"type": "integer", "description": "How many recommendations to return (1-20, default 5)."},
            },
            "required": [],
        },
    }
]
