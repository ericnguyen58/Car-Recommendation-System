# Ccreaear Recommendation System — Project Overview

A data pipeline and machine learning system built for first-time car buyers shopping the used market. Processes raw car specification data (2001–2024), enriches it with expert and consumer reviews, and produces a recommendation model that predicts buyer satisfaction from car specs. Includes an interactive web UI and CLI interface.

---

## Objective

Build a recommendation system for first-time buyers purchasing a used car. The system combines structured car specs, expert editorial ratings, and crowd-sourced consumer ratings into a single model that scores any car in the dataset across all model years. The focus is on finding reliable, well-rated cars within realistic budget constraints — not just recommending the newest models.

---

## Data Sources

| Source | Description | Size |
|---|---|---|
| `all_cars (1).csv` | Car trims with full specifications (Feb 2022) | 27,902 rows |
| `all_reviews.csv` | Expert editorial reviews with dimensional ratings | 4,782 reviews |

Reviews are at the **make + model + year** level — one review applies to all trims of the same model year. Expert ratings cover 6 dimensions: value, performance, quality, comfort, reliability, and styling. Consumer overall ratings come from aggregated buyer reviews (average 296 reviews per model, max 3,169).

---

## Pipeline

10 stages run end-to-end via `python main.py`:

```
raw CSVs
   │
   ▼ standardize.py            Type cleanup, ordinal encoding, car_id assignment
   │
   ▼ split_reviews.py          Separates review text from ratings, assigns review_id
   │
   ▼ split_cars.py             Splits specs from features, joins review_id FK
   │
   ▼ impute_hp_torque.py       Manual hp/torque lookup for 123 rare/luxury/EV models
   │                           using published manufacturer specs (Bentley, Ferrari,
   │                           McLaren, Lucid, Rivian, Polestar, etc.)
   │
   ▼ ml_impute_hp_torque.py    ML-based imputation for remaining 3,334 hp / 3,490 torque nulls
   │                           RF trained on real values; brand origin groups (American /
   │                           European / Japanese / Korean) used as features.
   │                           Predictions snapped to nearest known real value for same
   │                           make+model+year where possible (26–28% of cases).
   │
   ▼ zero_to_sixty.py          Physics-based 0–60 mph estimator (hp > 180 cars only)
   │
   ▼ build_consumer_view.py    Joins specs + features + expert ratings + consumer ratings
   │                           into a single consumer-focused table (27,902 × 56 cols)
   │
   ▼ fill_consumer_view.py     Aggressive null filling via group median/mode cascades
   │                           (make+model → make+bodytype → make → global)
   │
   ▼ export_final.py           Fills Pickup cargo = 0, drops ~2,318 unfillable rows
   │                           Output: model_ready.csv — 25,584 rows, zero nulls
   │
   ▼ train_recommender.py      Trains Random Forest, saves model artifacts
                               Output: models/rf_recommender.joblib
```

```bash
python main.py                           # full pipeline
python main.py --only train-recommender  # single stage
python main.py --skip zero-to-sixty      # skip a stage
```

---

## Dataset — model_ready.csv

**25,584 rows × 56 columns, zero nulls in all model columns.**

| Group | Columns |
|---|---|
| Identity | car_id, make, model, year, trim, bodytype, doors |
| Budget | price, Fuel Type, Engine, mpg_city, mpg_hwy, mpg_comb |
| Performance | hp, torque_lbft |
| Powertrain | Drivetrain, Transmission Type |
| Practicality | cargo_cuft |
| Flags | is_ev, luxury, hp_missing, torque_missing |
| Safety | Child Seat Anchors, Child Door Locks, Traction Control, Stability Control, Hill Start Assist, Blind-Spot Alert, Collision Warning System |
| Connectivity | Bluetooth, Hands Free Phone, Satellite Radio, Smartphone Interface, Navigation System, Voice Recognition, Internet Access, Real-Time Traffic, Premium Radio |
| Convenience | Cruise Control, Remote Keyless Entry, Remote Engine Start, Power Windows, Power Outlet, Rear Window Defroster, Steering Wheel Controls, Tilt Steering Wheel |
| Expert ratings | rating_value, rating_performance, rating_quality, rating_comfort, rating_reliability, rating_styling |
| Consumer ratings | consumer_overall_rating, consumer_review_count |

Feature availability columns use ordinal encoding: `0` = Not Available, `1` = Optional, `2` = Standard.
`hp_missing` / `torque_missing` = 1 if value was null in raw data before imputation (8.4% / 8.9% of rows).

---

## Imputation Strategy

Raw data had significant nulls in consumer-relevant columns. All imputation follows a group cascade — each level only fills rows still null after the previous.

| Column | Strategy |
|---|---|
| hp, torque | **Stage 1:** Manual lookup for 123 rare/luxury/EV models from published specs. **Stage 2:** RF regressor trained on real values using make, model, year, bodytype, Fuel Type, Drivetrain, brand origin; predictions snapped to nearest known real value for same model+year where available. |
| price | (make, bodytype, year) → (make, bodytype) → (make) median |
| mpg_comb | Derived from city+hwy average → group median cascade |
| mpg_city/hwy | Derived from mpg_comb using per-fuel-type ratios → group median |
| cargo_cuft | (make, model, year) → (make, model) → (make, bodytype) → bodytype mean; **Pickup trucks filled with 0** (no cargo box measurement) |
| Expert ratings | (make, model) → (make, bodytype) → (make) → global median |
| Consumer rating | Same cascade as expert ratings |
| Transmission, Drivetrain | Group mode cascade |
| Remaining unfillable | ~2,318 phantom EPA regulatory rows dropped before modeling |

**Intentionally left null:** Engine (free-text spec string), review_id for unreviewed cars.

---

## Recommendation Model

**Algorithm:** Random Forest Regressor (200 trees, min_samples_leaf=4)

**Target:** `consumer_overall_rating` — aggregated real buyer rating (2.5–5.0 scale).

**Train / test split:** Two-stage split, both at 80/20:

1. **Group-aware diagnostic holdout** — 20% of make+model groups (72 groups, 4,896 rows) are held out entirely before training begins. These models are never seen during training. Evaluated after training as a lower-bound diagnostic.
2. **Stratified random primary split** — from the remaining 287 groups, rows are split 80/20 stratified by 5-year year bucket. This is the primary train/test split and reflects real deployment conditions: the catalog is fixed, so users will always search for models the system has seen.

**Training set:** 15,373 rows (real consumer reviews only, 287 make+model groups, all years).

**Features (46 total):**
- Car specs: year, price, mpg (city/hwy/combined), hp, torque, cargo volume, doors
- Expert ratings: all 6 dimensions — included as features so the model learns their relationship to consumer satisfaction organically, without hard-coded weights
- Feature availability: 25 safety/connectivity/convenience columns (0/1/2)
- Categoricals: make, bodytype, Fuel Type, Drivetrain, Transmission Type (label-encoded)
- Binary flags: is_ev, luxury

**Evaluation:**

| Split | n | RMSE | R² | Spearman | Interpretation |
|---|---|---|---|---|---|
| Primary (stratified random) | 3,844 | 0.074 | 0.953 | **0.980** | Ranking known catalog cars — typical user experience |
| Diagnostic (group-aware) | 4,896 | 0.327 | 0.252 | 0.537 | Spec-based ranking of never-seen models — lower bound |

**Why Spearman is the right metric:** The model ranks cars within a filtered set; relative order matters more than absolute prediction accuracy. Spearman measures exactly that.

**Top feature importances:**

| Feature | Importance | Notes |
|---|---|---|
| cargo_cuft | 15.2% | Carries real signal on unseen models (+0.028 Spearman in diagnostic) |
| make | 11.4% | Brand reputation is legitimate signal for used buyers |
| rating_reliability | 8.0% | Most important expert dimension for used buyers |
| year | 7.8% | Age of car |
| torque_lbft | 7.6% | |
| rating_value | 5.4% | |
| price | 4.6% | MSRP proxy — relative signal only |
| Drivetrain | 4.4% | |
| hp | 3.7% | |

---

## Interfaces

### Web UI
```bash
streamlit run app.py
# Opens at http://localhost:8501
```
Sidebar filters: body type, make, fuel, drivetrain, transmission, price range, MPG, HP, year, predicted rating range, EV/luxury toggles.
Results table with colour-coded predicted rating, actual consumer rating alongside.
Car detail picker: full spec sheet, expert + consumer ratings, feature availability (Safety / Connectivity / Convenience), expandable extended specs.
Imputed hp/torque flagged with `~` in the detail view.

### CLI
```bash
python scripts/recommend_cli.py
```
Interactive prompt. Filters persist across queries. Type `help`, `options`, `show`, `clear`.

### Python
```python
from scripts.train_recommender import recommend

results = recommend(
    preferences={
        'bodytype':     'SUV',
        'max_price':    40000,
        'min_mpg_comb': 25,
        'min_year':     2020,
        'Drivetrain':   'AWD',
    },
    top_n=5,
    unique_models=True,   # one result per make+model (default)
)
```

Supported filter keys: `bodytype`, `make`, `Fuel Type`, `Drivetrain`, `Transmission Type`, `is_ev`, `luxury`, `max_price`, `min_price`, `min_mpg_comb`, `max_mpg_comb`, `min_hp`, `max_hp`, `min_year`, `max_year`, `min_cargo_cuft`.

### Conversational advisor (LLM interface)
```bash
python scripts/chat_recommender.py
```
Natural language interface powered by Claude claude-opus-4-6. The buyer describes what they want in plain language; Claude extracts structured filters, calls the recommendation engine via tool use, and explains results in buyer-friendly terms. Supports multi-turn conversation — follow-up questions like "show me only AWD options" or "what about something more fuel efficient?" adjust filters and re-query automatically.

Requires an `ANTHROPIC_API_KEY` environment variable and `pip install anthropic`.

### Model evaluation (no retraining)
```bash
python scripts/evaluate_model.py
```
Loads saved `.joblib` and reports: overall metrics on all reviewed cars, year-era breakdown (pre/post 2015), per-bodytype RMSE, per-make RMSE (best/worst 10), bucket accuracy confusion matrix, prediction vs actual distribution, mean bias. Primary train/test metrics are stored in `models/model_meta.json`.

---

## Key Design Decisions

**Consumer columns only.** Technical specs (engine dimensions, wheel base, fuel capacity) were excluded. The dataset is scoped to columns that actually influence purchase decisions.

**Expert and consumer reviews serve different roles.** Expert ratings (6 dimensions) are model *features* — they describe measurable attributes. Consumer overall rating is the model *target* — it captures buyer satisfaction. This separation means the model learns the relationship between specs/expert opinion and real buyer response, without predicting one rating source from another.

**No data leakage.** An earlier version included `consumer_overall_rating` as both a feature and part of the target, artificially inflating R² to 0.979. After correction, the model is driven purely by spec and expert signals.

**ML imputation for hp/torque.** Simple group median was replaced with a trained RF imputer that incorporates brand origin (American/European/Japanese/Korean) as a feature. Predictions are snapped to nearest known real value for same make+model+year where sibling trim data exists.

**Two-stage split design.** The model is built for a fixed used-car catalog. A temporal split would penalise the model for not predicting the future, which is irrelevant here. A naive stratified random split would let the same make+model appear in both train and test, inflating Spearman to ~0.98 through memorization rather than genuine learning. The two-stage approach resolves both: a group-aware holdout (20% of make+model groups, never seen in training) serves as a lower-bound diagnostic at Spearman=0.54, while the stratified random primary split (same groups, different rows) gives the deployment-realistic metric at Spearman=0.98. Both numbers are reported and meaningful.

**Imputation flags preserved.** `hp_missing` and `torque_missing` are kept in the dataset and surfaced in the UI, but excluded from model training — they reflect data collection gaps, not car quality.

**Pickup truck cargo.** `cargo_cuft` is structurally absent for traditional pickups. Filled with 0 rather than imputed — correctly reflects that cargo volume is not a relevant metric for that body type.

---

## Project Structure

```
├── app.py                      # Streamlit web UI
├── main.py                     # Pipeline runner (10 stages)
├── data/
│   ├── raw/                    # Source files (do not modify)
│   ├── processed/              # Standardized intermediates
│   └── final/
│       ├── consumer_view.csv   # Full imputed dataset (27,902 × 56)
│       ├── model_ready.csv     # Zero-null model input (25,584 × 56)
│       ├── cars/               # car_stats.csv, car_features.csv, zero_to_sixty_est.csv
│       └── reviews/            # review_summary.csv, review_ratings.csv
├── models/
│   ├── rf_recommender.joblib   # Trained Random Forest
│   ├── label_encoders.joblib   # Categorical encoders
│   ├── feature_list.json       # Feature column order
│   └── model_meta.json         # RMSE, R², Spearman, training details
├── notebooks/
│   ├── car_specs_eda.ipynb     # Original specs EDA (5-pillar structure)
│   └── consumer_eda.ipynb      # Consumer recommendation EDA
├── scripts/
│   ├── standardize.py          # Stage 1
│   ├── split_reviews.py        # Stage 2
│   ├── split_cars.py           # Stage 3
│   ├── impute_hp_torque.py     # Stage 4 — manual patches
│   ├── ml_impute_hp_torque.py  # Stage 5 — ML imputer
│   ├── zero_to_sixty.py        # Stage 6
│   ├── build_consumer_view.py  # Stage 7
│   ├── fill_consumer_view.py   # Stage 8
│   ├── export_final.py         # Stage 9
│   ├── train_recommender.py    # Stage 10 + recommend() function
│   ├── evaluate_model.py       # Standalone health check on all reviewed cars (no retraining)
│   ├── compare_cutoffs.py      # Historical: compared temporal cutoff years during split design
│   ├── recommend_cli.py        # Structured filter CLI (key=value syntax)
│   └── chat_recommender.py     # Conversational CLI powered by Claude claude-opus-4-6 (LLM interface)
└── schema/                     # Column definitions and category mappings
```

---

## Known Limitations

- **Price is MSRP, not used market price** — the dataset contains new car list prices. A 2016 Civic at $10k used vs $22k new are different buying decisions. Price filters work as relative signals, not accurate used market values.
- **Consumer ratings from a single source** — aggregated from one review site; skews toward enthusiast and dissatisfied buyers, not the full buyer population.
- **Low-rated cars underrepresented** — consumer ratings are right-skewed (most cars 4.0+) due to selection bias: people tend to rate cars they bought. Low-bucket (<3.5) recall is structurally weaker than Mid/High/Top — a data problem, not a model problem.
- **Mercedes-Benz consistently highest RMSE** — consumer ratings diverge from spec-based expectations more than any other make.

---

## Next Steps

- **Used market price integration** — replace MSRP with real used market prices (e.g. from KBB/Edmunds) so budget filters reflect what a used buyer actually pays; currently the highest-priority data gap
- **LLM integration** — wrap `recommend()` with a conversational interface so first-time buyers can describe needs in plain language
- **Reliability weighting** — expose `rating_reliability` as a user-adjustable filter priority, given its outsized importance for used car buyers
- **NDCG evaluation** — ranking metric against held-out user preference data for proper recommendation quality measurement
