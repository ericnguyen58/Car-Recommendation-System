"""
train_recommender.py
====================
Trains a Random Forest quality-score ranker on model_ready.csv.

Target:
  consumer_overall_rating — aggregated real buyer rating (2.5–5.0 scale).
  Only rows where consumer_review_count > 0 are used for training.

Train / test split strategy:
  Two-stage split:
  1. Group-aware diagnostic holdout — 20% of make+model groups held out
     entirely (never seen in training). Evaluated after training as a
     lower-bound diagnostic: "how well do specs alone predict quality for
     a brand-new unseen model?"
  2. Stratified random 80/20 on the remaining groups — primary train/test.
     Same make+models appear in both; tests ranking of known catalog cars,
     which is the actual deployment scenario (fixed used-car catalog).
  Primary metric is the stratified random test. Group-aware is diagnostic only.

Evaluation metrics reported:
  RMSE, R², Spearman rank correlation (most relevant for a ranker),
  per-bodytype RMSE breakdown (exposes imbalance), cargo_cuft ablation.

Features (46 total):
  Numeric  : year, price, mpg_city, mpg_hwy, mpg_comb, hp, torque_lbft,
             cargo_cuft, doors
  Expert   : rating_value, rating_performance, rating_quality, rating_comfort,
             rating_reliability, rating_styling
  Binary   : is_ev, luxury
  Ordinal  : 25 feature columns (0/1/2 — safety, connectivity, convenience)
  Encoded  : make, bodytype, Fuel Type, Drivetrain, Transmission Type

  hp_missing / torque_missing excluded — data quality flags, not car quality.
"""

import json

import joblib
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

from car_recommender.core.paths import PROJECT_ROOT

BASE         = PROJECT_ROOT
DATA_PATH    = BASE / 'data' / 'final' / 'model_ready.csv'
MODELS_DIR   = BASE / 'models'
MODELS_DIR.mkdir(exist_ok=True)

TEST_SIZE    = 0.20   # used for both stages of the two-stage split
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f'Loaded: {df.shape}')

RATING_COLS = [
    'rating_value', 'rating_performance', 'rating_quality',
    'rating_comfort', 'rating_reliability', 'rating_styling',
]

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    'year', 'price', 'mpg_city', 'mpg_hwy', 'mpg_comb',
    'hp', 'torque_lbft', 'cargo_cuft', 'doors',
] + RATING_COLS

BINARY_FEATURES = ['is_ev', 'luxury']

ORDINAL_FEATURES = [
    'Child Seat Anchors', 'Child Door Locks', 'Traction Control',
    'Stability Control', 'Hill Start Assist', 'Blind-Spot Alert',
    'Collision Warning System',
    'Bluetooth Wireless Technology', 'Hands Free Phone',
    'Bluetooth Streaming Audio', 'Satellite Radio', 'Smartphone Interface',
    'Navigation System', 'Voice Recognition System', 'Internet Access',
    'Real-Time Traffic Information', 'Premium Radio',
    'Cruise Control', 'Remote Keyless Entry', 'Remote Engine Start',
    'Power Windows', 'Power Outlet', 'Rear Window Defroster',
    'Steering Wheel Controls', 'Tilt Steering Wheel',
]

CATEGORICAL_FEATURES = ['make', 'bodytype', 'Fuel Type', 'Drivetrain', 'Transmission Type']

# ---------------------------------------------------------------------------
# Encode categoricals
# ---------------------------------------------------------------------------
label_encoders = {}
df_enc = df.copy()

for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df_enc[col + '_enc'] = le.fit_transform(df_enc[col].astype(str))
    label_encoders[col] = le

CAT_ENC_FEATURES = [c + '_enc' for c in CATEGORICAL_FEATURES]
ALL_FEATURES     = NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES + CAT_ENC_FEATURES

# ---------------------------------------------------------------------------
# Two-stage split
# ---------------------------------------------------------------------------
has_consumer  = df_enc['consumer_review_count'] > 0
reviewed      = df_enc[has_consumer].copy()
groups_series = reviewed['make'].astype(str) + '||' + reviewed['model'].astype(str)
n_groups      = groups_series.nunique()

# Stage 1 — group-aware diagnostic holdout (20% of make+model groups, never in training)
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
remaining_idx, diagnostic_idx = next(gss.split(reviewed, groups=groups_series))

remaining_df  = reviewed.iloc[remaining_idx].copy()
diagnostic_df = reviewed.iloc[diagnostic_idx].copy()

# Stage 2 — stratified random 80/20 on remaining rows (primary train/test)
remaining_df['year_bucket'] = pd.cut(
    remaining_df['year'],
    bins=[2000, 2005, 2010, 2015, 2020, 2025],
    labels=['2001-2005', '2006-2010', '2011-2015', '2016-2020', '2021-2025'],
)
train_df, primary_test_df = train_test_split(
    remaining_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=remaining_df['year_bucket'],
)

train_models = set(zip(train_df['make'],        train_df['model']))
diag_models  = set(zip(diagnostic_df['make'],   diagnostic_df['model']))
diag_overlap = train_models & diag_models
diag_n_groups = diagnostic_df.groupby(['make', 'model']).ngroups

print(f'\nTwo-stage split ({n_groups:,} total make+model groups):')
print(f'  Train            : {len(train_df):,} rows  ({len(train_models):,} groups)')
print(f'  Primary test     : {len(primary_test_df):,} rows  (same groups, stratified random)')
print(f'  Diagnostic test  : {len(diagnostic_df):,} rows  ({diag_n_groups} groups, never seen in training)')
print(f'  Diagnostic overlap: {len(diag_overlap)}  ← must be 0')

X_train, y_train = train_df[ALL_FEATURES],        train_df['consumer_overall_rating']
X_test,  y_test  = primary_test_df[ALL_FEATURES], primary_test_df['consumer_overall_rating']

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print('\nTraining RandomForestRegressor (n_estimators=200)…')
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Evaluation — main metrics
# ---------------------------------------------------------------------------
y_pred   = rf.predict(X_test)
rmse     = mean_squared_error(y_test, y_pred) ** 0.5
r2       = r2_score(y_test, y_pred)
spearman = spearmanr(y_test, y_pred).statistic

print(f'\n=== PRIMARY — Stratified random holdout (n={len(y_test):,}, known models) ===')
print(f'  RMSE              : {rmse:.4f}')
print(f'  R²                : {r2:.4f}')
print(f'  Spearman ρ (rank) : {spearman:.4f}')
print('  Note: Spearman is the most meaningful metric for a ranker —')
print('        it measures whether relative order is correct, not absolute error.')

# ---------------------------------------------------------------------------
# Evaluation — per bodytype RMSE (exposes class imbalance)
# ---------------------------------------------------------------------------
test_results = primary_test_df[['bodytype', 'consumer_overall_rating']].copy()
test_results['predicted'] = y_pred

print('\n=== Per-bodytype RMSE (test set) ===')
bt_rmse = (
    test_results.groupby('bodytype')
    .apply(lambda g: pd.Series({
        'n':    len(g),
        'RMSE': (mean_squared_error(g['consumer_overall_rating'], g['predicted']) ** 0.5),
    }), include_groups=False)
    .round(4)
    .sort_values('RMSE')
)
print(bt_rmse.to_string())
print('  Higher RMSE for rare body types reflects training set imbalance.')

# ---------------------------------------------------------------------------
# Evaluation — diagnostic (group-aware holdout, unseen make+models)
# ---------------------------------------------------------------------------
y_diag_true = diagnostic_df['consumer_overall_rating']
y_diag_pred = rf.predict(diagnostic_df[ALL_FEATURES])
diag_rmse     = mean_squared_error(y_diag_true, y_diag_pred) ** 0.5
diag_r2       = r2_score(y_diag_true, y_diag_pred)
diag_spearman = spearmanr(y_diag_true, y_diag_pred).statistic

print(f'\n=== DIAGNOSTIC — Group-aware holdout (n={len(y_diag_true):,}, unseen models) ===')
print(f'  RMSE              : {diag_rmse:.4f}')
print(f'  R²                : {diag_r2:.4f}')
print(f'  Spearman ρ (rank) : {diag_spearman:.4f}')
print('  Interpretation    : lower bound — spec-based ranking of never-seen models')

# ---------------------------------------------------------------------------
# Evaluation — cargo_cuft ablation
# ---------------------------------------------------------------------------
print('\n=== cargo_cuft ablation ===')
no_cargo_features = [f for f in ALL_FEATURES if f != 'cargo_cuft']
rf_no_cargo = RandomForestRegressor(
    n_estimators=200, min_samples_leaf=4, n_jobs=-1, random_state=42
)
rf_no_cargo.fit(X_train[no_cargo_features], y_train)
y_pred_nc    = rf_no_cargo.predict(X_test[no_cargo_features])
rmse_nc      = mean_squared_error(y_test, y_pred_nc) ** 0.5
spearman_nc  = spearmanr(y_test, y_pred_nc).statistic
print(f'  With cargo_cuft   : RMSE={rmse:.4f}  Spearman={spearman:.4f}')
print(f'  Without cargo_cuft: RMSE={rmse_nc:.4f}  Spearman={spearman_nc:.4f}')
delta = spearman - spearman_nc
print(f'  Δ Spearman = {delta:+.4f} — cargo_cuft {"adds real signal" if delta > 0.01 else "contribution is marginal" if delta > 0 else "is not contributing positively"}')

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
importance = pd.Series(rf.feature_importances_, index=ALL_FEATURES)
print('\n=== Top 20 Feature Importances ===')
print(importance.sort_values(ascending=False).head(20).round(4).to_string())

# ---------------------------------------------------------------------------
# Save model artifacts
# ---------------------------------------------------------------------------
joblib.dump(rf,             MODELS_DIR / 'rf_recommender.joblib')
joblib.dump(label_encoders, MODELS_DIR / 'label_encoders.joblib')

with open(MODELS_DIR / 'feature_list.json', 'w') as f:
    json.dump(ALL_FEATURES, f, indent=2)

meta = {
    'rmse':                    round(rmse, 4),
    'r2':                      round(r2, 4),
    'spearman':                round(float(spearman), 4),
    'n_train':                 len(X_train),
    'n_test':                  len(X_test),
    'train_years':             f'{train_df["year"].min()}-{train_df["year"].max()} (all eras)',
    'test_years':              f'{primary_test_df["year"].min()}-{primary_test_df["year"].max()} (all eras)',
    'n_train_groups':                len(train_models),
    'n_primary_test_groups':         primary_test_df.groupby(['make', 'model']).ngroups,
    'n_diagnostic_test_groups':      diag_n_groups,
    'diagnostic_rmse':               round(diag_rmse, 4),
    'diagnostic_r2':                 round(diag_r2, 4),
    'diagnostic_spearman':           round(float(diag_spearman), 4),
    'n_total_model_ready':           len(df),
    'n_with_consumer_reviews':       int(has_consumer.sum()),
    'target':                        'consumer_overall_rating (real buyer ratings)',
    'split':                         'stratified random 80/20 (primary) + group-aware diagnostic holdout',
    'n_estimators':            200,
    'features':                ALL_FEATURES,
}
with open(MODELS_DIR / 'model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f'\nSaved artifacts to {MODELS_DIR}/')


# ---------------------------------------------------------------------------
# recommend() — content-based filter + RF ranking
# ---------------------------------------------------------------------------

def recommend(preferences: dict, top_n: int = 5,
              unique_models: bool = True,
              data_path=DATA_PATH) -> pd.DataFrame:
    """
    Return top_n car recommendations ranked by predicted consumer rating.

    Parameters
    ----------
    preferences : dict
        bodytype, make, Fuel Type, Drivetrain, Transmission Type  : str
        is_ev, luxury                                             : bool
        max_price, min_price                                      : float
        min_mpg_comb, max_mpg_comb                               : float
        min_hp, max_hp                                            : float
        min_year, max_year                                        : int
        min_cargo_cuft                                            : float
    top_n : int
        Number of results to return.
    unique_models : bool
        If True (default), returns at most one result per make+model
        (best-rated year). Set False to see all trims.
    """
    rf_model  = joblib.load(MODELS_DIR / 'rf_recommender.joblib')
    le_dict   = joblib.load(MODELS_DIR / 'label_encoders.joblib')

    with open(MODELS_DIR / 'feature_list.json') as f:
        feat_list = json.load(f)

    df_rec = pd.read_csv(data_path)

    # Encode categoricals
    for col in CATEGORICAL_FEATURES:
        le    = le_dict[col]
        known = set(le.classes_)
        df_rec[col + '_enc'] = df_rec[col].astype(str).apply(
            lambda v, le=le, known=known: le.transform([v])[0] if v in known else -1
        )

    # Apply filters
    mask = pd.Series(True, index=df_rec.index)

    for key in ['bodytype', 'make', 'Fuel Type', 'Drivetrain', 'Transmission Type']:
        if key in preferences:
            mask &= df_rec[key].str.lower() == str(preferences[key]).lower()

    if 'is_ev'  in preferences: mask &= df_rec['is_ev']  == int(bool(preferences['is_ev']))
    if 'luxury' in preferences: mask &= df_rec['luxury'] == int(bool(preferences['luxury']))

    if 'max_price'     in preferences: mask &= df_rec['price']    <= preferences['max_price']
    if 'min_price'     in preferences: mask &= df_rec['price']    >= preferences['min_price']
    if 'min_mpg_comb'  in preferences: mask &= df_rec['mpg_comb'] >= preferences['min_mpg_comb']
    if 'max_mpg_comb'  in preferences: mask &= df_rec['mpg_comb'] <= preferences['max_mpg_comb']
    if 'min_hp'        in preferences: mask &= df_rec['hp']       >= preferences['min_hp']
    if 'max_hp'        in preferences: mask &= df_rec['hp']       <= preferences['max_hp']
    if 'min_year'      in preferences: mask &= df_rec['year']     >= preferences['min_year']
    if 'max_year'      in preferences: mask &= df_rec['year']     <= preferences['max_year']
    if 'min_cargo_cuft' in preferences: mask &= df_rec['cargo_cuft'] >= preferences['min_cargo_cuft']

    filtered = df_rec[mask].copy()
    print(f'Cars matching filters: {len(filtered):,}')

    if filtered.empty:
        print('No cars match the given preferences.')
        return pd.DataFrame()

    filtered['predicted_consumer_rating'] = rf_model.predict(filtered[feat_list]).round(3)

    # Include actual consumer rating where available for transparency
    IDENTITY = [
        'car_id', 'make', 'model', 'year', 'trim', 'bodytype',
        'price', 'mpg_comb', 'hp', 'torque_lbft', 'Drivetrain',
        'Transmission Type', 'Fuel Type',
        'consumer_overall_rating', 'consumer_review_count',
        'predicted_consumer_rating',
    ]

    result = filtered[IDENTITY].sort_values('predicted_consumer_rating', ascending=False)

    if unique_models:
        # Best-rated year per make+model, then best trim per make+model+year
        result = result.drop_duplicates(subset=['make', 'model'])
    else:
        result = result.drop_duplicates(subset=['make', 'model', 'year', 'trim'])

    return result.head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('\n=== Demo: Top 5 Sedans under $45k (2020+) ===')
    results = recommend(
        preferences={
            'bodytype':     'Sedan',
            'max_price':    45000,
            'min_year':     2020,
        },
        top_n=5,
    )
    if not results.empty:
        print(results[['make', 'model', 'year', 'trim', 'price', 'mpg_comb',
                       'hp', 'consumer_overall_rating', 'predicted_consumer_rating']].to_string(index=False))
    else:
        print('No results returned.')
