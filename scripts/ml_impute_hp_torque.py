"""
ml_impute_hp_torque.py
======================
ML-based imputation for hp and torque_lbft in car_stats.UVCRS.
Runs after impute_hp_torque.py (manual patches) and before build_consumer_view.py.

Strategy:
  1. Train a RandomForestRegressor on rows with real hp/torque values.
     Features include brand origin group (American/European/Japanese/Korean/Other)
     which captures regional engine design patterns (American V8s, Japanese
     efficiency-tuned engines, European turbos, etc.).

  2. Predict hp/torque for all null rows.

  3. Snap to known real value:
       Level 1 — same make+model+year+Fuel Type  (exact trim siblings)
       Level 2 — same make+model+Fuel Type        (same model, any year)
     If a snap target exists, round prediction to nearest real value in the group.
     This handles "this trim is one of N known engine configs for this model year."
     Rows with no snap target keep the raw ML prediction.

  4. Save patched car_stats.UVCRS and a report of what was imputed.

Features used:
  make (encoded), model (encoded), year, bodytype (encoded),
  Fuel Type (encoded), Drivetrain (encoded), Transmission Type (encoded),
  brand_origin (encoded), doors, curb_weight_lbs, mpg_comb
  (curb_weight and mpg_comb filled with group medians for rows where null)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

BASE      = Path(__file__).resolve().parents[1]
STATS_CSV = BASE / 'data' / 'final' / 'cars' / 'car_stats.UVCRS'
MODELS_DIR = BASE / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Brand origin groups — captures regional engine design patterns
# ---------------------------------------------------------------------------
BRAND_ORIGIN = {
    'American': {
        'Chevrolet', 'Ford', 'Dodge', 'GMC', 'Ram', 'Chrysler', 'Buick',
        'Cadillac', 'Lincoln', 'Jeep', 'Tesla', 'Rivian', 'Lucid',
    },
    'European': {
        'BMW', 'Mercedes-Benz', 'Audi', 'Volkswagen', 'Porsche', 'Volvo',
        'Jaguar', 'Land-Rover', 'Alfa-Romeo', 'Fiat', 'Mini', 'Maserati',
        'Ferrari', 'Lamborghini', 'Bentley', 'Rolls-Royce',
        'McLaren Automotive', 'Polestar',
    },
    'Japanese': {
        'Toyota', 'Honda', 'Nissan', 'Mazda', 'Subaru', 'Mitsubishi',
        'Lexus', 'Acura', 'Infiniti',
    },
    'Korean': {'Hyundai', 'Kia', 'Genesis'},
}

def brand_origin(make: str) -> str:
    for region, makes in BRAND_ORIGIN.items():
        if make in makes:
            return region
    return 'Other'

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(STATS_CSV)
print(f'Loaded car_stats: {df.shape}')
print(f'  hp nulls:      {df["hp"].isna().sum():,}')
print(f'  torque nulls:  {df["torque_lbft"].isna().sum():,}')

df['brand_origin'] = df['make'].apply(brand_origin)

# ---------------------------------------------------------------------------
# Feature prep
# ---------------------------------------------------------------------------
CAT_COLS = ['make', 'model', 'bodytype', 'Fuel Type',
            'Drivetrain', 'Transmission Type', 'brand_origin']
NUM_COLS = ['year', 'doors', 'curb_weight_lbs', 'mpg_comb']

encoders = {}
df_enc = df.copy()

for col in CAT_COLS:
    le = LabelEncoder()
    df_enc[col + '_enc'] = le.fit_transform(df_enc[col].fillna('Unknown').astype(str))
    encoders[col] = le

# Fill auxiliary numerics with group median for training stability
for col in ['curb_weight_lbs', 'mpg_comb', 'doors']:
    med = df_enc.groupby(['make', 'model', 'year'])[col].transform('median')
    df_enc[col] = df_enc[col].fillna(med).fillna(df_enc[col].median())

FEATURES = [c + '_enc' for c in CAT_COLS] + NUM_COLS


# ---------------------------------------------------------------------------
# Snap helper — round prediction to nearest real value in the group
# ---------------------------------------------------------------------------
def snap_to_known(pred_val: float, real_vals: list) -> float:
    """Return the real value closest to pred_val."""
    return min(real_vals, key=lambda v: abs(v - pred_val))


def build_snap_lookup(col: str, level: str) -> dict:
    """
    Build {group_key: [real_values]} from rows where col is not null.
    level='year'  → key = (make, model, year, Fuel Type)
    level='model' → key = (make, model, Fuel Type)
    """
    real = df[df[col].notna()]
    if level == 'year':
        grp = real.groupby(['make', 'model', 'year', 'Fuel Type'])[col].apply(list)
    else:
        grp = real.groupby(['make', 'model', 'Fuel Type'])[col].apply(list)
    return grp.to_dict()


# ---------------------------------------------------------------------------
# Train + predict for each target
# ---------------------------------------------------------------------------
def impute_column(col: str):
    has_val  = df_enc[col].notna()
    is_null  = df_enc[col].isna()
    n_null   = is_null.sum()

    if n_null == 0:
        print(f'  {col}: no nulls — skipping')
        return df[col].copy()

    X_train = df_enc.loc[has_val, FEATURES]
    y_train = df_enc.loc[has_val, col]
    X_pred  = df_enc.loc[is_null,  FEATURES]

    print(f'\n  Training RF for {col}  (train={len(X_train):,}, predict={n_null:,})…')
    rf = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # In-sample RMSE (sanity check)
    train_rmse = mean_squared_error(y_train, rf.predict(X_train)) ** 0.5
    print(f'  In-sample RMSE: {train_rmse:.2f}')

    raw_preds = rf.predict(X_pred)

    # Build snap lookups
    snap_year  = build_snap_lookup(col, 'year')
    snap_model = build_snap_lookup(col, 'model')

    snapped = 0
    final_preds = []
    for i, (idx, pred) in enumerate(zip(X_pred.index, raw_preds)):
        row  = df.loc[idx]
        key_year  = (row['make'], row['model'], row['year'],  row['Fuel Type'])
        key_model = (row['make'], row['model'],               row['Fuel Type'])

        if key_year in snap_year:
            snapped += 1
            final_preds.append(snap_to_known(pred, snap_year[key_year]))
        elif key_model in snap_model:
            snapped += 1
            final_preds.append(snap_to_known(pred, snap_model[key_model]))
        else:
            # Round to nearest 5 hp / 5 lb-ft (realistic spec granularity)
            final_preds.append(round(pred / 5) * 5)

    print(f'  Snapped to known trim value: {snapped:,} / {n_null:,} ({snapped/n_null:.1%})')
    print(f'  Pure ML prediction:          {n_null - snapped:,} / {n_null:,}')

    result = df[col].copy()
    result.loc[is_null] = final_preds
    return result


# ---------------------------------------------------------------------------
# Run imputation
# ---------------------------------------------------------------------------
print('\n=== HP imputation ===')
df['hp'] = impute_column('hp')

print('\n=== Torque imputation ===')
df['torque_lbft'] = impute_column('torque_lbft')

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
print(f'\n=== Post-imputation null check ===')
print(f'  hp nulls:      {df["hp"].isna().sum()}')
print(f'  torque nulls:  {df["torque_lbft"].isna().sum()}')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df.drop(columns=['brand_origin']).to_csv(STATS_CSV, index=False)
print(f'\nSaved → {STATS_CSV}')
