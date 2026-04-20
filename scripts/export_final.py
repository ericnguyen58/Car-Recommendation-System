"""
export_final.py
===============
Exports a model-ready dataset from consumer_view.UVCRS with zero nulls
in all model-relevant columns.

Steps:
  1. Load consumer_view.UVCRS (output of fill_consumer_view.py)
  2. Fill Pickup cargo_cuft = 0  (structurally absent, not missing)
  3. Drop rows that still have nulls in any model column (excl. Engine, review_id)
     → ~170 phantom EPA rows that have no make/model/bodytype
  4. Save to data/final/model_ready.UVCRS
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SRC  = BASE / 'data' / 'final' / 'consumer_view.UVCRS'
OUT  = BASE / 'data' / 'final' / 'model_ready.UVCRS'

df = pd.read_csv(SRC)
print(f'Loaded: {df.shape}')

# ---------------------------------------------------------------------------
# 1. Pickup cargo_cuft = 0  (no cargo box measurement — fill with 0, not null)
# ---------------------------------------------------------------------------
pickup_null = (df['bodytype'] == 'Pickup') & df['cargo_cuft'].isna()
df.loc[pickup_null, 'cargo_cuft'] = 0.0
print(f'  Pickup cargo_cuft filled: {pickup_null.sum()} rows → 0.0')

# ---------------------------------------------------------------------------
# 2. Define model columns (exclude Engine free-text, exclude review_id)
# ---------------------------------------------------------------------------
EXCLUDE = {'Engine', 'review_id'}
MODEL_COLS = [c for c in df.columns if c not in EXCLUDE]

null_counts = df[MODEL_COLS].isnull().sum()
print(f'\nNull counts before drop (model cols):')
print(null_counts[null_counts > 0].sort_values(ascending=False).to_string())

# ---------------------------------------------------------------------------
# 3. Drop rows with any remaining null in model columns
# ---------------------------------------------------------------------------
before = len(df)
df_clean = df[df[MODEL_COLS].notna().all(axis=1)].copy()
dropped  = before - len(df_clean)
print(f'\nDropped {dropped:,} rows ({dropped/before:.1%}) with unresolvable nulls')
print(f'Final shape: {df_clean.shape}')

# Sanity check
remaining = df_clean[MODEL_COLS].isnull().sum().sum()
assert remaining == 0, f'Still {remaining} nulls in model cols!'
print(f'Null check passed: 0 nulls in all {len(MODEL_COLS)} model columns')

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
df_clean.to_csv(OUT, index=False)
print(f'\nSaved → {OUT}')
print(f'Shape  : {df_clean.shape}')
print(f'Kept   : {len(df_clean):,} / {before:,} rows ({len(df_clean)/before:.1%})')
