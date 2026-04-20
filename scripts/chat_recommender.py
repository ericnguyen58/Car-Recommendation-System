"""
chat_recommender.py
===================
Conversational car advisor for first-time used car buyers.
Uses Claude claude-opus-4-6 with tool use to translate natural language into
structured catalog queries, then formats results in plain language.

Requirements:
  pip install anthropic

Run:
  python scripts/chat_recommender.py

How it works:
  1. Claude listens to the buyer in plain language
  2. When recommendations are needed, Claude calls get_car_recommendations
     with structured filters derived from the conversation
  3. The tool runs against the pre-loaded catalog and returns ranked results
  4. Claude explains the results in buyer-friendly terms
  5. Conversation continues until the buyer is satisfied or exits
"""

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE       = Path(__file__).resolve().parents[1]
DATA_PATH  = BASE / 'data' / 'final' / 'model_ready.UVCRS'
MODELS_DIR = BASE / 'models'

CATEGORICAL_FEATURES = ['make', 'bodytype', 'Fuel Type', 'Drivetrain', 'Transmission Type']

NUMERIC_FILTERS = {
    'max_price':      ('price',      '<='),
    'min_price':      ('price',      '>='),
    'min_mpg_comb':   ('mpg_comb',   '>='),
    'max_mpg_comb':   ('mpg_comb',   '<='),
    'min_hp':         ('hp',         '>='),
    'max_hp':         ('hp',         '<='),
    'min_year':       ('year',       '>='),
    'max_year':       ('year',       '<='),
    'min_cargo_cuft': ('cargo_cuft', '>='),
}

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
print('Loading model artifacts...')
rf_model  = joblib.load(MODELS_DIR / 'rf_recommender.joblib')
le_dict   = joblib.load(MODELS_DIR / 'label_encoders.joblib')

with open(MODELS_DIR / 'feature_list.json') as f:
    feat_list = json.load(f)

df = pd.read_csv(DATA_PATH)

for col in CATEGORICAL_FEATURES:
    le    = le_dict[col]
    known = set(le.classes_)
    df[col + '_enc'] = df[col].astype(str).apply(
        lambda v, le=le, known=known: le.transform([v])[0] if v in known else -1
    )

df['predicted_rating'] = rf_model.predict(df[feat_list]).round(3)

VALID_OPTIONS = {
    col: sorted(df[col].dropna().unique().tolist())
    for col in CATEGORICAL_FEATURES
}

print(f'Ready. {len(df):,} cars loaded.\n')


# ---------------------------------------------------------------------------
# Filter and rank
# ---------------------------------------------------------------------------
def apply_filters(prefs: dict) -> pd.DataFrame:
    """Apply structured filters from the tool call to the catalog DataFrame."""
    mask = pd.Series(True, index=df.index)

    # Remap underscore tool key back to DataFrame column name with space
    if 'fuel_type' in prefs:
        prefs = dict(prefs)
        prefs['Fuel Type'] = prefs.pop('fuel_type')

    for key in CATEGORICAL_FEATURES:
        if key not in prefs:
            continue
        val  = prefs[key]
        vals = [v.lower() for v in (val if isinstance(val, list) else [val])]
        mask &= df[key].str.lower().isin(vals)

    if 'is_ev'  in prefs: mask &= df['is_ev']  == int(bool(prefs['is_ev']))
    if 'luxury' in prefs: mask &= df['luxury'] == int(bool(prefs['luxury']))

    for key, (col, op) in NUMERIC_FILTERS.items():
        if key not in prefs:
            continue
        val = float(prefs[key])
        mask &= (df[col] <= val) if op == '<=' else (df[col] >= val)

    return df[mask]


def get_recommendations(prefs: dict, top_n: int = 5) -> dict:
    """Filter catalog and return top-N cars ranked by predicted buyer satisfaction."""
    results = apply_filters(prefs)

    if results.empty:
        return {
            "count": 0,
            "message": "No cars matched those filters. Try relaxing one or more criteria.",
        }

    display_cols = [c for c in [
        'make', 'model', 'year', 'trim', 'bodytype', 'Fuel Type', 'Drivetrain',
        'price', 'mpg_comb', 'hp', 'rating_reliability', 'predicted_rating',
    ] if c in results.columns]

    top = (
        results[display_cols]
        .sort_values('predicted_rating', ascending=False)
        .drop_duplicates(subset=['make', 'model', 'year'])
        .head(top_n)
        .reset_index(drop=True)
    )
    top.index += 1

    records = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for _, row in top.iterrows()
    ]

    return {
        "count":         len(records),
        "total_matched": int(len(results)),
        "cars":          records,
        "note":          (
            "Prices shown are new-car MSRP — used market prices will be "
            "significantly lower (often 30-60% less depending on age and mileage)."
        ),
    }


# ---------------------------------------------------------------------------
# Claude API setup
# ---------------------------------------------------------------------------
MODEL_ID   = 'claude-opus-4-6'
MAX_TOKENS = 4096

SYSTEM = f"""You are a knowledgeable, friendly car buying advisor helping first-time buyers \
find a reliable used car. The catalog covers model years 2001-2024 across all major makes.

You have one tool: get_car_recommendations. Use it whenever the buyer asks for \
suggestions, wants to compare options, or mentions any preference that maps to a filter.

When presenting results:
- Highlight the top 2-3 picks and explain specifically why each fits the buyer's stated needs
- Always call out the reliability rating (out of 5.0) when available — this is the most \
important factor for used car buyers
- Mention fuel economy for budget-conscious buyers
- Remind buyers that listed prices are new-car MSRP, not used market value
- If a search returns zero results, diagnose which filter is likely too strict, \
suggest an adjustment, and call the tool again with relaxed criteria
- Keep explanations plain, concise, and free of jargon

Valid catalog values for reference:
  Body types  : {', '.join(VALID_OPTIONS['bodytype'])}
  Drivetrains : {', '.join(VALID_OPTIONS['Drivetrain'])}
  Fuel types  : {', '.join(VALID_OPTIONS['Fuel Type'][:12])} (and more)
"""

TOOLS = [
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
                    "description": (
                        "One or more car makes to include, e.g. ['Toyota', 'Honda']. "
                        "Omit to search all makes."
                    ),
                },
                "bodytype": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "One or more body types: Sedan, SUV, Pickup, Hatchback, Coupe, "
                        "Convertible, Wagon, Minivan, Van."
                    ),
                },
                "min_year": {
                    "type": "integer",
                    "description": (
                        "Minimum model year. 2012-2016 is usually the sweet spot for "
                        "affordable, modern used cars."
                    ),
                },
                "max_year": {
                    "type": "integer",
                    "description": "Maximum model year.",
                },
                "max_price": {
                    "type": "number",
                    "description": (
                        "Maximum MSRP in USD. This is the NEW car price — used market "
                        "price is much lower. Use as a class/quality filter, not a budget filter."
                    ),
                },
                "min_price": {
                    "type": "number",
                    "description": "Minimum MSRP in USD (new car price, see max_price note).",
                },
                "min_mpg_comb": {
                    "type": "number",
                    "description": "Minimum combined fuel economy in MPG.",
                },
                "min_hp": {
                    "type": "number",
                    "description": "Minimum horsepower.",
                },
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
                "is_ev": {
                    "type": "boolean",
                    "description": "Set true to show only battery-electric vehicles.",
                },
                "luxury": {
                    "type": "boolean",
                    "description": "Set true to show only luxury/premium brand vehicles.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "How many recommendations to return (1-20, default 5).",
                },
            },
            "required": [],
        },
    }
]


def execute_tool(name: str, tool_input: dict) -> str:
    if name != "get_car_recommendations":
        return json.dumps({"error": f"Unknown tool: {name}"})

    top_n  = min(int(tool_input.pop("top_n", 5)), 20)
    result = get_recommendations(tool_input, top_n=top_n)
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Conversational loop
# ---------------------------------------------------------------------------
def chat():
    try:
        import anthropic
    except ImportError:
        print("anthropic package required. Install it with: pip install anthropic")
        sys.exit(1)

    api_key  = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        env_file = BASE / '.env'
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip().startswith('ANTHROPIC_API_KEY'):
                    api_key = line.split('=', 1)[-1].strip().strip('"').strip("'")
                    break
    client   = anthropic.Anthropic(api_key=api_key, max_retries=5)
    messages = []

    print('Car Advisor — first-time buyer assistant')
    print('Tell me what you are looking for. Type "quit" to exit.\n')

    while True:
        try:
            user_input = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye.')
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print('Goodbye.')
            break

        messages.append({"role": "user", "content": user_input})

        # Inner loop handles tool calls until Claude produces a final text response
        while True:
            with client.messages.stream(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
            ) as stream:
                started = False
                for text in stream.text_stream:
                    if not started:
                        print('\nAdvisor: ', end='', flush=True)
                        started = True
                    print(text, end='', flush=True)

                if started:
                    print()

                response = stream.get_final_message()

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            # Execute all tool calls and feed results back
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, dict(block.input))
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })

            messages.append({"role": "user", "content": tool_results})

        print()


if __name__ == '__main__':
    chat()
