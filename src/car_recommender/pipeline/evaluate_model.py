"""
evaluate_model.py
=================
Loads the saved rf_recommender.joblib and evaluates it against model_ready.csv.
Does NOT retrain — reads the model exactly as saved.

Outputs:
  - Overall metrics (RMSE, MAE, R², Spearman ρ) on all reviewed cars
  - Year-era breakdown (pre-2015 vs 2015+) — descriptive, not a train/test split
  - Per-bodytype RMSE
  - Per-make RMSE (top/bottom 10)
  - Rating bucket accuracy (how often predicted bucket matches actual)
  - Prediction vs actual distribution comparison

Note: The model was trained with a two-stage split (see train_recommender.py).
      This script evaluates on all reviewed rows as a holistic health check.
      Primary train/test metrics are in model_meta.json.

Run:
  python -m car_recommender.pipeline.evaluate_model
"""

import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score

from car_recommender.core.paths import PROJECT_ROOT

BASE       = PROJECT_ROOT
DATA_PATH  = BASE / 'data' / 'final' / 'model_ready.csv'
MODELS_DIR = BASE / 'models'

ERA_SPLIT_YEAR     = 2015   # descriptive era split for health-check reporting only
CATEGORICAL_FEATURES = ['make', 'bodytype', 'Fuel Type', 'Drivetrain', 'Transmission Type']

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
print('Loading model artifacts…')
rf        = joblib.load(MODELS_DIR / 'rf_recommender.joblib')
le_dict   = joblib.load(MODELS_DIR / 'label_encoders.joblib')

with open(MODELS_DIR / 'feature_list.json') as f:
    feat_list = json.load(f)

with open(MODELS_DIR / 'model_meta.json') as f:
    meta = json.load(f)

print(f'  Model target  : {meta["target"]}')
print(f'  Trained on    : years {meta["train_years"]}')
print(f'  Split         : {meta["split"]}')
print(f'  n_estimators  : {meta["n_estimators"]}')
print(f'  Training rows : {meta["n_train"]:,}')
print(f'  Primary test  : Spearman={meta["spearman"]}  RMSE={meta["rmse"]}  R²={meta["r2"]}')
print(f'  Diagnostic    : Spearman={meta["diagnostic_spearman"]}  RMSE={meta["diagnostic_rmse"]}  R²={meta["diagnostic_r2"]}')

# ---------------------------------------------------------------------------
# Load and encode data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

for col in CATEGORICAL_FEATURES:
    le    = le_dict[col]
    known = set(le.classes_)
    df[col + '_enc'] = df[col].astype(str).apply(
        lambda v, le=le, known=known: le.transform([v])[0] if v in known else -1
    )

# Restrict to rows with real consumer reviews
reviewed = df[df['consumer_review_count'] > 0].copy()
reviewed['predicted'] = rf.predict(reviewed[feat_list]).round(2)
y_true = reviewed['consumer_overall_rating']
y_pred = reviewed['predicted']

print(f'\nEvaluating on {len(reviewed):,} rows with real consumer reviews.')

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def metrics(true, pred, label=''):
    rmse     = mean_squared_error(true, pred) ** 0.5
    mae      = mean_absolute_error(true, pred)
    r2       = r2_score(true, pred)
    spearman = spearmanr(true, pred).statistic
    pearson  = pearsonr(true, pred)[0]
    if label:
        print(f'\n=== {label} ===')
    print(f'  RMSE              : {rmse:.4f}')
    print(f'  MAE               : {mae:.4f}')
    print(f'  R²                : {r2:.4f}')
    print(f'  Spearman ρ (rank) : {spearman:.4f}')
    print(f'  Pearson r         : {pearson:.4f}')
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'spearman': spearman}

# ---------------------------------------------------------------------------
# 1. Overall metrics
# ---------------------------------------------------------------------------
metrics(y_true, y_pred, 'Overall (all reviewed cars)')

# ---------------------------------------------------------------------------
# 2. Year-era breakdown (descriptive health check, not a train/test split)
# ---------------------------------------------------------------------------
older_era  = reviewed[reviewed['year'] <  ERA_SPLIT_YEAR]
newer_era  = reviewed[reviewed['year'] >= ERA_SPLIT_YEAR]

metrics(older_era['consumer_overall_rating'], older_era['predicted'],
        f'Older cars (<{ERA_SPLIT_YEAR}, n={len(older_era):,})')
metrics(newer_era['consumer_overall_rating'], newer_era['predicted'],
        f'Newer cars (>={ERA_SPLIT_YEAR}, n={len(newer_era):,})')

# ---------------------------------------------------------------------------
# 3. Per-bodytype RMSE
# ---------------------------------------------------------------------------
print('\n=== Per-bodytype RMSE ===')
bt = (
    reviewed.groupby('bodytype')
    .apply(lambda g: pd.Series({
        'n':         len(g),
        'RMSE':      round((mean_squared_error(g['consumer_overall_rating'], g['predicted']) ** 0.5), 4),
        'Spearman':  round(spearmanr(g['consumer_overall_rating'], g['predicted']).statistic, 4),
        'mean_actual':    round(g['consumer_overall_rating'].mean(), 3),
        'mean_predicted': round(g['predicted'].mean(), 3),
    }), include_groups=False)
    .sort_values('RMSE')
)
print(bt.to_string())

# ---------------------------------------------------------------------------
# 4. Per-make RMSE (best and worst 10)
# ---------------------------------------------------------------------------
make_metrics = (
    reviewed.groupby('make')
    .apply(lambda g: pd.Series({
        'n':        len(g),
        'RMSE':     round((mean_squared_error(g['consumer_overall_rating'], g['predicted']) ** 0.5), 4),
        'Spearman': round(spearmanr(g['consumer_overall_rating'], g['predicted']).statistic, 4)
                    if len(g) > 5 else np.nan,
    }), include_groups=False)
    .sort_values('RMSE')
)

print('\n=== Per-make RMSE — best 10 ===')
print(make_metrics.head(10).to_string())
print('\n=== Per-make RMSE — worst 10 ===')
print(make_metrics.tail(10).to_string())

# ---------------------------------------------------------------------------
# 5. Rating bucket accuracy
#    Bucket: Low (<3.5), Mid (3.5–4.2), High (4.2–4.6), Top (≥4.6)
# ---------------------------------------------------------------------------
def bucket(x):
    if x < 3.5:  return 'Low  (<3.5)'
    if x < 4.2:  return 'Mid  (3.5–4.2)'
    if x < 4.6:  return 'High (4.2–4.6)'
    return             'Top  (≥4.6)'

reviewed['actual_bucket']    = y_true.apply(bucket)
reviewed['predicted_bucket'] = y_pred.apply(bucket)

correct  = (reviewed['actual_bucket'] == reviewed['predicted_bucket']).mean()
adj_correct = (
    (reviewed['actual_bucket'] == reviewed['predicted_bucket']) |
    (reviewed['actual_bucket'].shift() == reviewed['predicted_bucket'])  # off-by-one bucket
).mean()

print('\n=== Rating Bucket Accuracy ===')
print(f'  Exact bucket match : {correct:.1%}')
print()
print('  Confusion (rows=actual, cols=predicted):')
conf = pd.crosstab(
    reviewed['actual_bucket'],
    reviewed['predicted_bucket'],
    normalize='index'
).round(3)
print(conf.to_string())

# Precision, Recall, F1 per bucket
BUCKET_ORDER = ['Low  (<3.5)', 'Mid  (3.5–4.2)', 'High (4.2–4.6)', 'Top  (≥4.6)']
print('\n=== Precision / Recall / F1 per Rating Bucket ===')
print(classification_report(
    reviewed['actual_bucket'],
    reviewed['predicted_bucket'],
    labels=BUCKET_ORDER,
    zero_division=0,
))

# ---------------------------------------------------------------------------
# 6. Prediction distribution vs actual
# ---------------------------------------------------------------------------
print('\n=== Prediction vs Actual Distribution ===')
comp = pd.DataFrame({
    'Actual':    y_true.describe().round(3),
    'Predicted': y_pred.describe().round(3),
})
print(comp.to_string())

bias = (y_pred - y_true).mean()
print(f'\n  Mean bias (predicted − actual): {bias:+.4f}')
print(f'  {"Model over-predicts on average" if bias > 0 else "Model under-predicts on average"}')
