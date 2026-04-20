"""
recommend_cli.py
================
Interactive command-line interface for the car recommender.
Loads the model and dataset once at startup — no retraining needed.

Run:
  python scripts/recommend_cli.py

Usage inside the prompt:
  bodytype=SUV max_price=40000 min_year=2020 top=5
  make=Toyota min_mpg_comb=30
  bodytype=Sedan luxury=1 min_hp=300 max_price=60000
  clear        — reset all active filters
  show         — show currently active filters
  options      — show valid values for categorical fields
  quit / exit  — exit
"""

from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd

BASE       = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE / 'models'
DATA_PATH  = BASE / 'data' / 'final' / 'model_ready.UVCRS'

CATEGORICAL_FEATURES = ['make', 'bodytype', 'Fuel Type', 'Drivetrain', 'Transmission Type']

# Numeric filter keys → (column, operator)
NUMERIC_FILTERS = {
    'max_price':     ('price',      '<='),
    'min_price':     ('price',      '>='),
    'min_mpg_comb':  ('mpg_comb',   '>='),
    'max_mpg_comb':  ('mpg_comb',   '<='),
    'min_hp':        ('hp',         '>='),
    'max_hp':        ('hp',         '<='),
    'min_year':      ('year',       '>='),
    'max_year':      ('year',       '<='),
    'min_cargo_cuft':('cargo_cuft', '>='),
}

STRING_FILTERS = ['bodytype', 'make', 'Fuel Type', 'Drivetrain', 'Transmission Type']

# ---------------------------------------------------------------------------
# Load everything once
# ---------------------------------------------------------------------------
print('Loading model and data...')
rf_model  = joblib.load(MODELS_DIR / 'rf_recommender.joblib')
le_dict   = joblib.load(MODELS_DIR / 'label_encoders.joblib')

with open(MODELS_DIR / 'feature_list.json') as f:
    feat_list = json.load(f)

df = pd.read_csv(DATA_PATH)

# Pre-encode all categoricals once
for col in CATEGORICAL_FEATURES:
    le    = le_dict[col]
    known = set(le.classes_)
    df[col + '_enc'] = df[col].astype(str).apply(
        lambda v, le=le, known=known: le.transform([v])[0] if v in known else -1
    )

# Pre-compute predictions for every row once
df['predicted_consumer_rating'] = rf_model.predict(df[feat_list]).round(3)

# Build option lists for help
OPTIONS = {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES}

print(f'Ready. {len(df):,} cars loaded.\n')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def show_options():
    for col, vals in OPTIONS.items():
        print(f'  {col}:')
        line = ''
        for v in vals:
            entry = f'{v}, '
            if len(line) + len(entry) > 78:
                print(f'    {line.rstrip(", ")}')
                line = entry
            else:
                line += entry
        if line:
            print(f'    {line.rstrip(", ")}')
        print()


def parse_input(raw: str) -> tuple[dict, int]:
    """Parse 'key=value key=value top=N' into (preferences_dict, top_n)."""
    prefs  = {}
    top_n  = 5
    errors = []

    for token in raw.strip().split():
        if '=' not in token:
            errors.append(f'  Skipped "{token}" — no = found')
            continue
        key, _, val = token.partition('=')
        key = key.strip()
        val = val.strip()

        if key == 'top':
            try:
                top_n = int(val)
            except ValueError:
                errors.append(f'  top must be an integer, got "{val}"')
            continue

        if key in ('is_ev', 'luxury'):
            prefs[key] = val.lower() in ('1', 'true', 'yes')
            continue

        if key in NUMERIC_FILTERS:
            try:
                prefs[key] = float(val)
            except ValueError:
                errors.append(f'  {key} must be a number, got "{val}"')
            continue

        if key in STRING_FILTERS:
            prefs[key] = val
            continue

        errors.append(f'  Unknown filter "{key}"')

    for e in errors:
        print(e)

    return prefs, top_n


def apply_filters(prefs: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    for key in STRING_FILTERS:
        if key in prefs:
            mask &= df[key].str.lower() == prefs[key].lower()

    if 'is_ev'  in prefs: mask &= df['is_ev']  == int(prefs['is_ev'])
    if 'luxury' in prefs: mask &= df['luxury'] == int(prefs['luxury'])

    for key, (col, op) in NUMERIC_FILTERS.items():
        if key not in prefs:
            continue
        if op == '<=':
            mask &= df[col] <= prefs[key]
        else:
            mask &= df[col] >= prefs[key]

    return df[mask]


def display(results: pd.DataFrame, top_n: int):
    if results.empty:
        print('  No cars match those filters.')
        return

    cols = ['make', 'model', 'year', 'trim', 'price', 'mpg_comb',
            'hp', 'Drivetrain', 'predicted_consumer_rating']
    top = (
        results[cols]
        .sort_values('predicted_consumer_rating', ascending=False)
        .drop_duplicates(subset=['make', 'model', 'year', 'trim'])
        .head(top_n)
        .reset_index(drop=True)
    )
    top.index += 1
    print()
    print(top.to_string())
    print(f'\n  {len(results):,} cars matched — showing top {min(top_n, len(top))}')


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
HELP_TEXT = """
Filters (space-separated, key=value):
  bodytype=SUV          make=Toyota       Fuel Type=Electric
  Drivetrain=AWD        Transmission Type=Automatic
  max_price=40000       min_price=20000   min_mpg_comb=28
  min_hp=200            max_hp=400        min_year=2020   max_year=2023
  min_cargo_cuft=15     is_ev=1           luxury=0        top=10

Commands:
  show      — show active filters
  clear     — clear all filters
  options   — show valid values for categorical fields
  help      — show this message
  quit      — exit
"""

def run():
    print('=' * 60)
    print('  Car Recommender')
    print('  Type "help" for usage, "quit" to exit')
    print('=' * 60)

    active_filters: dict = {}
    active_top_n: int    = 5

    while True:
        try:
            raw = input('\n> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye.')
            sys.exit(0)

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ('quit', 'exit', 'q'):
            print('Bye.')
            sys.exit(0)

        if cmd == 'help':
            print(HELP_TEXT)
            continue

        if cmd == 'options':
            show_options()
            continue

        if cmd == 'show':
            if not active_filters:
                print('  No active filters.')
            else:
                for k, v in active_filters.items():
                    print(f'  {k} = {v}')
                print(f'  top = {active_top_n}')
            continue

        if cmd == 'clear':
            active_filters = {}
            active_top_n   = 5
            print('  Filters cleared.')
            continue

        # Parse new filters and merge with active ones
        new_prefs, new_top = parse_input(raw)

        if new_prefs or new_top != 5:
            active_filters.update(new_prefs)
            active_top_n = new_top

        results = apply_filters(active_filters)
        display(results, active_top_n)


if __name__ == '__main__':
    run()
