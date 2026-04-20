"""
fill_consumer_view.py
=====================
Fills remaining nulls in consumer_view.UVCRS as aggressively as possible
without fabricating data.

Strategies used (in priority order per column):

  ratings     → (make, model) median  → (make, bodytype) → (make) → global median
  mpg_city/hwy→ derive from mpg_comb × fuel-type ratio (computed from real data)
                → then (make, model, year, Fuel Type) median
  cargo_cuft  → (make, model, year) mean → (make, model) → (make, bodytype)
                → (bodytype) global mean  [Pickups excluded — structurally absent]
  hp / torque → (Fuel Type) global median as final fallback after existing cascade
  doors       → global mode (4.0) — all remaining nulls are phantom EPA rows
  bodytype    → (make) mode  — last resort before giving up
  Transmission→ global mode (Automatic)

What is intentionally left null:
  Engine       — free-text spec string, no meaningful imputation
  cargo_cuft   — Pickup trucks (structural absence, not missing data)
  review_id    — no source for unreviewed cars
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
CSV  = BASE / 'data' / 'final' / 'consumer_view.UVCRS'

df = pd.read_csv(CSV)
print(f'Loaded: {df.shape}')

RATING_COLS = [
    'rating_value', 'rating_performance', 'rating_quality',
    'rating_comfort', 'rating_reliability', 'rating_styling',
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cascade_median(col, group_levels):
    for grp in group_levels:
        mask = df[col].isna()
        if not mask.any():
            break
        filled = df.groupby(grp, dropna=False)[col].transform('median')
        df.loc[mask, col] = filled[mask]

def cascade_mean(col, group_levels, row_mask=None):
    for grp in group_levels:
        mask = df[col].isna()
        if row_mask is not None:
            mask = mask & row_mask
        if not mask.any():
            break
        src   = df[row_mask] if row_mask is not None else df
        filled = src.groupby(grp, dropna=False)[col].transform('mean')
        df.loc[mask, col] = filled.reindex(df.index)[mask]

def mode_fill(col, grp):
    mask = df[col].isna()
    if not mask.any():
        return
    def _mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else np.nan
    filled = df.groupby(grp, dropna=False)[col].transform(_mode)
    df.loc[mask, col] = filled[mask]

def null_report(label):
    cols = RATING_COLS + [
        'consumer_overall_rating', 'consumer_review_count',
        'mpg_city', 'mpg_hwy', 'mpg_comb', 'hp', 'torque_lbft',
        'cargo_cuft', 'doors', 'bodytype', 'Transmission Type', 'price',
    ]
    total_null = df[cols].isnull().sum().sum()
    print(f'\n[{label}]  total remaining nulls across consumer cols: {total_null:,}')

# ---------------------------------------------------------------------------
# 1. RATINGS  (3,406 rows with no review data)
#    Rationale: expert ratings are highly consistent within a model line.
#    Imputing from same make+model gives a defensible proxy for unreviewed trims.
# ---------------------------------------------------------------------------
null_report('before ratings fill')

for col in RATING_COLS:
    cascade_median(col, [
        ['make', 'model'],
        ['make', 'bodytype'],
        ['make'],
    ])
    # Final fallback: global median
    global_med = df[col].median()
    df[col] = df[col].fillna(global_med)

null_report('after ratings fill')

# ---------------------------------------------------------------------------
# 1b. CONSUMER OVERALL RATING  (model-year level, not trim level)
#     Same cascade as expert ratings.
#     consumer_review_count nulls → 0 (no reviews recorded = 0 reviews)
# ---------------------------------------------------------------------------
cascade_median('consumer_overall_rating', [
    ['make', 'model'],
    ['make', 'bodytype'],
    ['make'],
])
global_consumer_med = df['consumer_overall_rating'].median()
df['consumer_overall_rating'] = df['consumer_overall_rating'].fillna(global_consumer_med)

df['consumer_review_count'] = df['consumer_review_count'].fillna(0)

null_report('after consumer rating fill')

# ---------------------------------------------------------------------------
# 2. MPG CITY / HWY
#    Derive from mpg_comb using per-fuel-type city/hwy ratios computed from
#    observed data (avoids assuming a single ratio for all fuel types).
#    Remaining nulls (where mpg_comb is also null) → group median cascade.
# ---------------------------------------------------------------------------

# Compute ratios from rows where all three are observed
has_all = df[df['mpg_city'].notna() & df['mpg_hwy'].notna() & df['mpg_comb'].notna()]
ratios = has_all.groupby('Fuel Type').apply(lambda x: pd.Series({
    'city_ratio': (x['mpg_city'] / x['mpg_comb']).median(),
    'hwy_ratio':  (x['mpg_hwy']  / x['mpg_comb']).median(),
}))

# Apply ratios where mpg_comb is known
for fuel, row in ratios.iterrows():
    fuel_mask = df['Fuel Type'] == fuel

    city_null = df['mpg_city'].isna() & df['mpg_comb'].notna() & fuel_mask
    df.loc[city_null, 'mpg_city'] = (df.loc[city_null, 'mpg_comb'] * row['city_ratio']).round(1)

    hwy_null  = df['mpg_hwy'].isna() & df['mpg_comb'].notna() & fuel_mask
    df.loc[hwy_null,  'mpg_hwy']  = (df.loc[hwy_null,  'mpg_comb'] * row['hwy_ratio']).round(1)

# Remaining city/hwy (mpg_comb also null) → group median
cascade_median('mpg_city', [['make', 'model', 'year', 'Fuel Type'], ['make', 'model', 'Fuel Type']])
cascade_median('mpg_hwy',  [['make', 'model', 'year', 'Fuel Type'], ['make', 'model', 'Fuel Type']])

null_report('after mpg city/hwy fill')

# ---------------------------------------------------------------------------
# 3. CARGO_CUFT  (Pickups excluded — cargo is structurally absent for them)
#    More aggressive cascade including (bodytype) global mean as last resort.
# ---------------------------------------------------------------------------
non_pickup = df['bodytype'] != 'Pickup'

cascade_mean('cargo_cuft', [
    ['make', 'model', 'year'],
    ['make', 'model'],
    ['make', 'bodytype'],
    ['bodytype'],           # global mean per bodytype — last resort
], row_mask=non_pickup)

null_report('after cargo fill')

# ---------------------------------------------------------------------------
# 4. HP / TORQUE  — add Fuel Type global median as final fallback
#    (e.g. all remaining null EVs get the EV median; gas cars get gas median)
# ---------------------------------------------------------------------------
for col in ['hp', 'torque_lbft']:
    cascade_median(col, [
        ['make', 'model', 'year'],
        ['make', 'model'],
        ['make', 'bodytype', 'Fuel Type'],
        ['Fuel Type'],          # global median per fuel type
    ])

null_report('after hp/torque fill')

# ---------------------------------------------------------------------------
# 5. DOORS  — all 1,766 remaining nulls are phantom EPA entries (bodytype=NaN)
#    Global mode = 4.0 (sedan/SUV default)
# ---------------------------------------------------------------------------
df['doors'] = df['doors'].fillna(4.0)

# ---------------------------------------------------------------------------
# 6. BODYTYPE  — (make) mode as last resort
# ---------------------------------------------------------------------------
mode_fill('bodytype', ['make'])

# ---------------------------------------------------------------------------
# 7. TRANSMISSION TYPE  — global mode (Automatic) as final fallback
# ---------------------------------------------------------------------------
global_trans = df['Transmission Type'].mode()[0]
df['Transmission Type'] = df['Transmission Type'].fillna(global_trans)

# ---------------------------------------------------------------------------
# Final null report
# ---------------------------------------------------------------------------
null_report('FINAL')

print('\n=== Final null counts (consumer columns) ===')
check_cols = [
    'price', 'mpg_city', 'mpg_hwy', 'mpg_comb',
    'hp', 'torque_lbft', 'cargo_cuft',
    'bodytype', 'doors', 'Drivetrain', 'Transmission Type',
] + RATING_COLS
result = pd.DataFrame({
    'null':   df[check_cols].isnull().sum(),
    'null_%': (df[check_cols].isnull().mean() * 100).round(1),
})
print(result[result['null'] > 0].sort_values('null_%', ascending=False).to_string())

# Core completeness
core = ['price', 'mpg_comb', 'hp', 'torque_lbft', 'bodytype', 'Drivetrain', 'Transmission Type']
complete = df[core].notna().all(axis=1)
print(f'\nRows with ALL core features filled: {complete.sum():,} ({complete.mean():.1%})')

df.to_csv(CSV, index=False)
print(f'\nSaved → {CSV}')
