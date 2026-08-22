"""
build_consumer_view.py
======================
Produces data/final/consumer_view.csv — a clean, consumer-focused dataset.

Imputation order (consumer columns only):
  1. bodytype      — mode by (make, model)
  2. doors         — keyword fill → mode by (make, model, bodytype)
  3. mpg_comb      — derive from city+hwy → median by (make, model, year, Fuel Type)
                     → (make, model, Fuel Type)
  4. hp            — median by (make, model, year) → (make, model)
                     → (make, bodytype, Fuel Type)
  5. torque_lbft   — same cascade as hp
  6. Transmission  — mode by (make, bodytype) → (make)
  7. Drivetrain    — mode by (make, bodytype)
  8. price         — median by (make, bodytype, year) → (make, bodytype) → (make)
  9. cargo_cuft    — mean by (make, model, year) → (make, model) → (make, bodytype)
                     Pickups excluded (structurally absent)

Flags:
  hp_missing, torque_missing  — 1 if value was null in raw data (before imputation)

Joins:
  car_features  (NaN → 0 = Not Available)
  review_ratings (left join on review_id)
"""

import pandas as pd

from car_recommender.core.paths import PROJECT_ROOT

BASE    = PROJECT_ROOT
OUT     = BASE / 'data' / 'final' / 'consumer_view.csv'

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
cars     = pd.read_csv(BASE / 'data/final/cars/car_stats.csv')
features = pd.read_csv(BASE / 'data/final/cars/car_features.csv')
reviews  = pd.read_csv(BASE / 'data/final/reviews/review_ratings.csv')
summary  = pd.read_csv(BASE / 'data/final/reviews/review_summary.csv',
                       usecols=['review_id', 'consumer_overall_rating', 'consumer_review_count'])

print(f'Loaded  cars     : {cars.shape}')
print(f'Loaded  features : {features.shape}')
print(f'Loaded  reviews  : {reviews.shape}')
print(f'Loaded  summary  : {summary.shape}')

# ---------------------------------------------------------------------------
# Null flags BEFORE any imputation (preserves "data was actually missing" signal)
# ---------------------------------------------------------------------------
cars['hp_missing']     = cars['hp'].isna().astype('Int8')
cars['torque_missing'] = cars['torque_lbft'].isna().astype('Int8')

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
LUXURY_MAKES = {
    'Acura','Alfa-Romeo','Audi','Bentley','BMW','Cadillac','Ferrari','Genesis',
    'Infiniti','Jaguar','Lamborghini','Land-Rover','Lexus','Lincoln','Lucid',
    'Maserati','McLaren Automotive','Mercedes-Benz','Polestar','Porsche',
    'Rivian','Rolls-Royce','Tesla','Volvo',
}
cars['is_ev']  = cars['Fuel Type'].str.contains('Electric', case=False, na=False)
cars['luxury'] = cars['make'].isin(LUXURY_MAKES).astype('Int8')

# ---------------------------------------------------------------------------
# Helper — cascade group-fill
# ---------------------------------------------------------------------------
def cascade_fill(col, groups_list, agg='median'):
    for grp in groups_list:
        still_null = cars[col].isna()
        if not still_null.any():
            break
        filled = cars.groupby(grp, dropna=False)[col].transform(agg)
        cars.loc[still_null, col] = filled[still_null]

def mode_fill(col, grp):
    """Fill nulls with group mode (most frequent)."""
    still_null = cars[col].isna()
    if not still_null.any():
        return
    def _mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else x
    filled = cars.groupby(grp, dropna=False)[col].transform(_mode)
    cars.loc[still_null, col] = filled[still_null]

# ---------------------------------------------------------------------------
# 1. bodytype — mode by (make, model)
# ---------------------------------------------------------------------------
mode_fill('bodytype', ['make', 'model'])

# ---------------------------------------------------------------------------
# 2. doors — keyword fill → mode by (make, model, bodytype)
# ---------------------------------------------------------------------------
FOUR_DOORS  = ('Double Cab|Quad Cab|Super Cab|Crew Cab|Flex Titanium|Extended Cab|'
               'Suburban|Yukon XL|Sienna|Sequoia|Sedona|RLX|MDX|ZDX')
TWO_DOORS   = 'King Cab|XtraCab|Access Cab|Regular Cab|Club Cab|Clubman John Cooper'
THREE_DOORS = 'Accent Blue'

null_d = cars['doors'].isna()
cars.loc[null_d & cars['trim'].str.contains(FOUR_DOORS,  case=False, na=False), 'doors'] = 4
cars.loc[null_d & cars['trim'].str.contains(TWO_DOORS,   case=False, na=False), 'doors'] = 2
cars.loc[null_d & cars['trim'].str.contains(THREE_DOORS, case=False, na=False), 'doors'] = 3

mode_fill('doors', ['make', 'model', 'bodytype'])
mode_fill('doors', ['make', 'bodytype'])        # fallback

# ---------------------------------------------------------------------------
# 3. mpg_comb — derive from city+hwy, then group median
# ---------------------------------------------------------------------------
fill_mask = cars['mpg_comb'].isna() & cars['mpg_city'].notna() & cars['mpg_hwy'].notna()
cars.loc[fill_mask, 'mpg_comb'] = (
    (cars.loc[fill_mask, 'mpg_city'] + cars.loc[fill_mask, 'mpg_hwy']) / 2
).round(1)

cascade_fill('mpg_comb', [
    ['make', 'model', 'year', 'Fuel Type'],
    ['make', 'model', 'Fuel Type'],
    ['make', 'bodytype', 'Fuel Type'],
])

# ---------------------------------------------------------------------------
# 4. hp — cascade median
# ---------------------------------------------------------------------------
cascade_fill('hp', [
    ['make', 'model', 'year'],
    ['make', 'model'],
    ['make', 'bodytype', 'Fuel Type'],
])

# ---------------------------------------------------------------------------
# 5. torque_lbft — same cascade as hp
# ---------------------------------------------------------------------------
cascade_fill('torque_lbft', [
    ['make', 'model', 'year'],
    ['make', 'model'],
    ['make', 'bodytype', 'Fuel Type'],
])

# ---------------------------------------------------------------------------
# 6. Transmission Type — mode cascade
# ---------------------------------------------------------------------------
mode_fill('Transmission Type', ['make', 'bodytype'])
mode_fill('Transmission Type', ['make'])

# ---------------------------------------------------------------------------
# 7. Drivetrain — mode by (make, bodytype)
# ---------------------------------------------------------------------------
mode_fill('Drivetrain', ['make', 'bodytype'])

# ---------------------------------------------------------------------------
# 8. price — median cascade
# ---------------------------------------------------------------------------
cascade_fill('price', [
    ['make', 'bodytype', 'year'],
    ['make', 'bodytype'],
    ['make'],
])

# ---------------------------------------------------------------------------
# 9. cargo_cuft — mean cascade, Pickups excluded
# ---------------------------------------------------------------------------
non_pickup = cars['bodytype'] != 'Pickup'
for grp in [['make', 'model', 'year'], ['make', 'model'], ['make', 'bodytype']]:
    still_null = cars['cargo_cuft'].isna() & non_pickup
    if not still_null.any():
        break
    filled = cars.loc[non_pickup].groupby(grp, dropna=False)['cargo_cuft'].transform('mean')
    cars.loc[still_null, 'cargo_cuft'] = filled.reindex(cars.index)[still_null]

# ---------------------------------------------------------------------------
# Null summary after imputation
# ---------------------------------------------------------------------------
CONSUMER_COLS = [
    'price', 'mpg_comb', 'hp', 'torque_lbft',
    'bodytype', 'doors', 'cargo_cuft',
    'Fuel Type', 'Drivetrain', 'Transmission Type',
]
print('\n=== Null summary after imputation ===')
print(f'Total rows: {len(cars):,}')
for c in CONSUMER_COLS:
    n = cars[c].isna().sum()
    print(f'  {c:<22} {n:>5,} null  ({n/len(cars)*100:5.1f}%)')

# ---------------------------------------------------------------------------
# Join car_features (NaN → 0 = Not Available)
# ---------------------------------------------------------------------------
# Keep only the consumer-relevant feature columns (safety, comfort, connectivity)
CONSUMER_FEATURES = [
    # Safety
    'Child Seat Anchors', 'Child Door Locks', 'Traction Control',
    'Stability Control', 'Hill Start Assist', 'Blind-Spot Alert',
    'Collision Warning System',
    # Connectivity & infotainment
    'Bluetooth Wireless Technology', 'Hands Free Phone',
    'Bluetooth Streaming Audio', 'Satellite Radio', 'Smartphone Interface',
    'Navigation System', 'Voice Recognition System', 'Internet Access',
    'Real-Time Traffic Information', 'Premium Radio',
    # Convenience
    'Cruise Control', 'Remote Keyless Entry', 'Remote Engine Start',
    'Power Windows', 'Power Outlet', 'Rear Window Defroster',
    'Steering Wheel Controls', 'Tilt Steering Wheel',
]

features_clean = features[['car_id'] + CONSUMER_FEATURES].copy()
features_clean[CONSUMER_FEATURES] = features_clean[CONSUMER_FEATURES].fillna(0).astype('Int8')

# ---------------------------------------------------------------------------
# Build consumer_view
# ---------------------------------------------------------------------------
KEEP_STATS = [
    'car_id', 'review_id',
    'make', 'model', 'year', 'trim', 'bodytype', 'doors',
    'price', 'Fuel Type', 'Engine', 'mpg_city', 'mpg_hwy', 'mpg_comb',
    'hp', 'hp_missing', 'torque_lbft', 'torque_missing',
    'Drivetrain', 'Transmission Type',
    'cargo_cuft',
    'is_ev', 'luxury',
]

# Merge expert ratings + consumer summary into one review table
review_full = reviews.merge(summary, on='review_id', how='left')

consumer_view = (
    cars[KEEP_STATS]
    .merge(features_clean, on='car_id', how='left')
    .merge(
        cars[['car_id', 'review_id']].drop_duplicates()
            .merge(review_full, on='review_id', how='left')
            .drop(columns='review_id'),
        on='car_id', how='left'
    )
)

# ---------------------------------------------------------------------------
# Final null summary
# ---------------------------------------------------------------------------
print('\n=== Consumer view — final null summary ===')
print(f'Shape: {consumer_view.shape}')
remaining = consumer_view.isnull().sum()
remaining = remaining[remaining > 0].sort_values(ascending=False)
pct = (remaining / len(consumer_view) * 100).round(1)
print(pd.concat([remaining.rename('null'), pct.rename('null_%')], axis=1).to_string())

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
consumer_view.to_csv(OUT, index=False)
print(f'\nSaved → {OUT}')
print(f'Shape  : {consumer_view.shape}')
