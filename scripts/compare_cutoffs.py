"""
compare_cutoffs.py
==================
Trains and evaluates RF recommender for multiple TRAIN_MAX_YEAR cutoffs.
Each model is evaluated on its OWN honest holdout (years > cutoff),
not a hardcoded split — fixing the evaluate_model.py leakage issue.

Usage:
  python scripts/compare_cutoffs.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

BASE      = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / 'data' / 'final' / 'model_ready.UVCRS'

CUTOFFS = [2019, 2022]

RATING_COLS = [
    'rating_value', 'rating_performance', 'rating_quality',
    'rating_comfort', 'rating_reliability', 'rating_styling',
]
NUMERIC_FEATURES  = ['year', 'price', 'mpg_city', 'mpg_hwy', 'mpg_comb',
                     'hp', 'torque_lbft', 'cargo_cuft', 'doors'] + RATING_COLS
BINARY_FEATURES   = ['is_ev', 'luxury']
ORDINAL_FEATURES  = [
    'Child Seat Anchors', 'Child Door Locks', 'Traction Control',
    'Stability Control', 'Hill Start Assist', 'Blind-Spot Alert',
    'Collision Warning System', 'Bluetooth Wireless Technology',
    'Hands Free Phone', 'Bluetooth Streaming Audio', 'Satellite Radio',
    'Smartphone Interface', 'Navigation System', 'Voice Recognition System',
    'Internet Access', 'Real-Time Traffic Information', 'Premium Radio',
    'Cruise Control', 'Remote Keyless Entry', 'Remote Engine Start',
    'Power Windows', 'Power Outlet', 'Rear Window Defroster',
    'Steering Wheel Controls', 'Tilt Steering Wheel',
]
CATEGORICAL_FEATURES = ['make', 'bodytype', 'Fuel Type', 'Drivetrain', 'Transmission Type']

# ---------------------------------------------------------------------------
def bucket(x):
    if x < 3.5: return 'Low  (<3.5)'
    if x < 4.2: return 'Mid  (3.5-4.2)'
    if x < 4.6: return 'High (4.2-4.6)'
    return             'Top  (>=4.6)'

def print_metrics(true, pred, label=''):
    rmse     = mean_squared_error(true, pred) ** 0.5
    mae      = mean_absolute_error(true, pred)
    r2       = r2_score(true, pred)
    spearman = spearmanr(true, pred).statistic
    pearson  = np.corrcoef(true, pred)[0, 1]
    if label:
        print(f'\n--- {label} ---')
    print(f'  n        : {len(true):,}')
    print(f'  RMSE     : {rmse:.4f}')
    print(f'  MAE      : {mae:.4f}')
    print(f'  R²       : {r2:.4f}')
    print(f'  Spearman : {spearman:.4f}')
    print(f'  Pearson  : {pearson:.4f}')
    return rmse, mae, r2, spearman

# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f'Loaded: {df.shape}')

# Encode categoricals once
label_encoders = {}
df_enc = df.copy()
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df_enc[col + '_enc'] = le.fit_transform(df_enc[col].astype(str))
    label_encoders[col] = le

CAT_ENC_FEATURES = [c + '_enc' for c in CATEGORICAL_FEATURES]
ALL_FEATURES     = NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES + CAT_ENC_FEATURES

has_consumer = df_enc['consumer_review_count'] > 0
reviewed     = df_enc[has_consumer].copy()

# ---------------------------------------------------------------------------
summaries = {}

for CUTOFF in CUTOFFS:
    sep = '=' * 70
    print(f'\n{sep}')
    print(f'  TRAIN_MAX_YEAR = {CUTOFF}  |  train ≤{CUTOFF}, test >{CUTOFF}')
    print(sep)

    train_df = reviewed[reviewed['year'] <= CUTOFF]
    test_df  = reviewed[reviewed['year'] >  CUTOFF]

    print(f'\nSplit:')
    print(f'  Train : {len(train_df):,} rows  ({train_df["year"].min()}–{train_df["year"].max()})')
    print(f'  Test  : {len(test_df):,}  rows  ({test_df["year"].min()}–{test_df["year"].max()})')

    X_train, y_train = train_df[ALL_FEATURES], train_df['consumer_overall_rating']
    X_test,  y_test  = test_df[ALL_FEATURES],  test_df['consumer_overall_rating']

    print(f'\nTraining RF (n_estimators=200)…')
    rf = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=4, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    y_pred_test  = rf.predict(X_test)

    # 1. Overall metrics
    print_metrics(y_train, y_pred_train, f'Train-era (≤{CUTOFF}) — in-sample')
    rmse_test, mae_test, r2_test, sp_test = print_metrics(
        y_test, y_pred_test, f'Test-era  (>{CUTOFF}) — honest holdout'
    )
    summaries[CUTOFF] = dict(rmse=rmse_test, mae=mae_test, r2=r2_test, spearman=sp_test)

    # 2. Per-bodytype RMSE (test only)
    test_results = test_df[['bodytype', 'year', 'consumer_overall_rating']].copy()
    test_results['predicted'] = y_pred_test

    print(f'\n--- Per-bodytype RMSE (test set) ---')
    bt = (
        test_results.groupby('bodytype')
        .apply(lambda g: pd.Series({
            'n':             len(g),
            'RMSE':          round((mean_squared_error(g['consumer_overall_rating'], g['predicted']) ** 0.5), 4),
            'Spearman':      round(spearmanr(g['consumer_overall_rating'], g['predicted']).statistic, 4)
                             if len(g) > 5 else np.nan,
            'mean_actual':   round(g['consumer_overall_rating'].mean(), 3),
            'mean_pred':     round(g['predicted'].mean(), 3),
        }), include_groups=False)
        .sort_values('RMSE')
    )
    print(bt.to_string())

    # 3. Per-make RMSE (test only)
    make_m = (
        test_results.groupby(test_df['make'])
        .apply(lambda g: pd.Series({
            'n':        len(g),
            'RMSE':     round((mean_squared_error(g['consumer_overall_rating'], g['predicted']) ** 0.5), 4),
            'Spearman': round(spearmanr(g['consumer_overall_rating'], g['predicted']).statistic, 4)
                        if len(g) > 5 else np.nan,
        }), include_groups=False)
        .sort_values('RMSE')
    )
    print(f'\n--- Per-make RMSE — best 10 (test set) ---')
    print(make_m.head(10).to_string())
    print(f'\n--- Per-make RMSE — worst 10 (test set) ---')
    print(make_m.tail(10).to_string())

    # 4. Bucket accuracy (test only)
    test_results['actual_bucket']    = test_results['consumer_overall_rating'].apply(bucket)
    test_results['predicted_bucket'] = test_results['predicted'].apply(bucket)
    correct = (test_results['actual_bucket'] == test_results['predicted_bucket']).mean()

    print(f'\n--- Rating Bucket Accuracy (test set) ---')
    print(f'  Exact match : {correct:.1%}')
    conf = pd.crosstab(
        test_results['actual_bucket'],
        test_results['predicted_bucket'],
        normalize='index'
    ).round(3)
    print(conf.to_string())

    # Low bucket detail
    low_mask = test_results['actual_bucket'] == 'Low  (<3.5)'
    print(f'\n  Low-bucket rows in test: {low_mask.sum()}')
    if low_mask.sum() > 0:
        low_recall = (test_results.loc[low_mask, 'predicted_bucket'] == 'Low  (<3.5)').mean()
        print(f'  Low-bucket recall      : {low_recall:.1%}')

    # 5. Distribution comparison (test only)
    print(f'\n--- Prediction vs Actual Distribution (test set) ---')
    comp = pd.DataFrame({
        'Actual':    y_test.describe().round(3),
        'Predicted': pd.Series(y_pred_test).describe().round(3),
    })
    print(comp.to_string())
    bias = (y_pred_test - y_test.values).mean()
    print(f'  Mean bias: {bias:+.4f}  ({"over" if bias > 0 else "under"}-predicts)')

# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('  SUMMARY — Test-era honest holdout comparison')
print('=' * 70)
print(f'{"Cutoff":<10} {"RMSE":>8} {"MAE":>8} {"R²":>8} {"Spearman":>10}')
print('-' * 50)
for c, s in summaries.items():
    print(f'≤{c:<9} {s["rmse"]:>8.4f} {s["mae"]:>8.4f} {s["r2"]:>8.4f} {s["spearman"]:>10.4f}')
