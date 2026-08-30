# Car Recommender — Cloud-Native Car Recommendation Platform

![CI](https://github.com/ericnguyen58/Car-Recommendation-System/actions/workflows/ci.yml/badge.svg)

A data pipeline and machine learning platform built for first-time car buyers shopping the used market. Processes raw car specification data (2001–2024), enriches it with expert and consumer reviews, and serves a recommendation model through a FastAPI backend, a Streamlit UI, and an LLM-powered conversational advisor. Deploys to **AWS** (S3, ECR, App Runner, Secrets Manager) with a GitHub Actions CI/CD pipeline.

---

## Architecture

```
                         ┌────────────────────┐
                         │   Streamlit UI      │  frontend/streamlit_app.py
                         │  (thin HTTP client) │  — no model loading, no DataFrame
                         └──────────┬──────────┘
                                    │ HTTPS + X-API-Key
                                    ▼
                         ┌────────────────────┐
                         │   FastAPI backend   │  src/car_recommender/api/
                         │  GET  /cars/filters │
                         │  GET  /cars/{id}    │
                         │  POST /recommendations
                         │  POST /advisor/chat │──────► Anthropic API (Claude)
                         └──────────┬──────────┘
                                    │
                     MODEL_STORE_BACKEND=local | s3
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                                ▼
         data/ + models/ (local disk)      s3://<bucket>/data, /models
                                                       │
                                            Secrets Manager (ANTHROPIC_API_KEY,
                                            API_PASSWORD) — DEPLOY_ENV=aws

  ML pipeline (offline, produces the artifacts the API serves):
  src/car_recommender/pipeline/  →  python -m car_recommender.pipeline.runner
```

Locally, everything defaults to the filesystem (`MODEL_STORE_BACKEND=local`, `DEPLOY_ENV=local`) — no AWS account needed to develop or run the tests. In production the same code reads model/data artifacts from S3 and secrets from Secrets Manager, running on AWS App Runner behind an image pushed to ECR (see [AWS Deployment](#aws-deployment)).

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

10 stages run end-to-end via `python -m car_recommender.pipeline.runner`, plus one optional stage:

```
raw CSVs
   │
   ▼ standardize            Type cleanup, ordinal encoding, car_id assignment
   │
   ▼ split-reviews          Separates review text from ratings, assigns review_id
   │
   ▼ split-cars             Splits specs from features, joins review_id FK
   │
   ▼ impute-hp-torque       Manual hp/torque lookup for 123 rare/luxury/EV models
   │                        using published manufacturer specs (Bentley, Ferrari,
   │                        McLaren, Lucid, Rivian, Polestar, etc.)
   │
   ▼ ml-impute-hp-torque    ML-based imputation for remaining 3,334 hp / 3,490 torque nulls
   │                        RF trained on real values; brand origin groups (American /
   │                        European / Japanese / Korean) used as features.
   │                        Predictions snapped to nearest known real value for same
   │                        make+model+year where possible (26–28% of cases).
   │
   ▼ zero-to-sixty          Physics-based 0–60 mph estimator (hp > 180 cars only)
   │
   ▼ build-consumer-view    Joins specs + features + expert ratings + consumer ratings
   │                        into a single consumer-focused table (27,902 × 56 cols)
   │
   ▼ fill-consumer-view     Aggressive null filling via group median/mode cascades
   │                        (make+model → make+bodytype → make → global)
   │
   ▼ export-final           Fills Pickup cargo = 0, joins used_price.csv if present,
   │                        drops ~2,318 unfillable rows
   │                        Output: model_ready.csv — 25,584 rows, zero nulls
   │
   ▼ train-recommender      Trains Random Forest, saves model artifacts
                            Output: models/rf_recommender.joblib

  scrape-used-price (optional, off by default — see Used-Market Price below)
```

```bash
python -m car_recommender.pipeline.runner                            # full pipeline
python -m car_recommender.pipeline.runner --only train-recommender   # single stage
python -m car_recommender.pipeline.runner --skip zero-to-sixty       # skip a stage
python -m car_recommender.pipeline.runner --only scrape-used-price   # optional, see below
```

---

## Dataset — model_ready.csv

**25,584 rows × 56 columns, zero nulls in all model columns** (plus `used_price_est` / `used_price_missing` when the optional scrape stage has been run).

| Group | Columns |
|---|---|
| Identity | car_id, make, model, year, trim, bodytype, doors |
| Budget | price, Fuel Type, Engine, mpg_city, mpg_hwy, mpg_comb |
| Performance | hp, torque_lbft |
| Powertrain | Drivetrain, Transmission Type |
| Practicality | cargo_cuft |
| Flags | is_ev, luxury, hp_missing, torque_missing, used_price_missing |
| Safety | Child Seat Anchors, Child Door Locks, Traction Control, Stability Control, Hill Start Assist, Blind-Spot Alert, Collision Warning System |
| Connectivity | Bluetooth, Hands Free Phone, Satellite Radio, Smartphone Interface, Navigation System, Voice Recognition, Internet Access, Real-Time Traffic, Premium Radio |
| Convenience | Cruise Control, Remote Keyless Entry, Remote Engine Start, Power Windows, Power Outlet, Rear Window Defroster, Steering Wheel Controls, Tilt Steering Wheel |
| Expert ratings | rating_value, rating_performance, rating_quality, rating_comfort, rating_reliability, rating_styling |
| Consumer ratings | consumer_overall_rating, consumer_review_count |
| Used-market price (optional) | used_price_est, used_price_missing |

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

**Intentionally left null:** Engine (free-text spec string), review_id for unreviewed cars, used_price_est (unless the optional scrape stage has been run).

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

### FastAPI backend

```bash
uv run uvicorn car_recommender.api.main:app --reload
# http://localhost:8000/docs for interactive OpenAPI docs
```

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /cars/filters` | Distinct makes/bodytypes/fuel types/etc. + numeric bounds, for building filter UIs |
| `GET /cars/{car_id}` | Full spec sheet: ratings, feature availability, extended specs |
| `POST /recommendations` | Filtered, ranked, deduplicated-by-model search |
| `POST /advisor/chat` | Conversational advisor — Claude + the `get_car_recommendations` tool, run server-side |

All routes except `/health` require `X-API-Key: <API_PASSWORD>` when `API_PASSWORD` is set (open in local dev when it's unset). `/advisor/chat` is additionally rate-limited (30 requests/hour per client by default).

### Streamlit UI (thin client)

```bash
uv run streamlit run frontend/streamlit_app.py
# http://localhost:8501 — talks to the API above, no local model/data loading
```

Sidebar filters call `POST /recommendations`; the Advisor tab calls `POST /advisor/chat` and holds the (opaque, server-round-tripped) conversation history in `st.session_state`. Set `API_BASE_URL` (defaults to `http://localhost:8000`) and `API_KEY` (if the backend has `API_PASSWORD` set) via env var, `.env`, or `st.secrets`.

### Conversational advisor (LLM)

`POST /advisor/chat`, implemented in `src/car_recommender/llm/advisor.py`. The buyer describes what they want in plain language; Claude extracts structured filters, calls `get_car_recommendations` via tool use, and explains results in buyer-friendly terms.

The model is a **config value, not hardcoded** — `ADVISOR_MODEL` (default: `claude-haiku-4-5`, the cheapest current model, for development). Upgrading the advisor to a stronger model is a one-line env var change, no code change:

```bash
export ADVISOR_MODEL=claude-opus-5   # or claude-sonnet-5, etc.
```

Requires `ANTHROPIC_API_KEY` (env var, `.env`, or — in production — AWS Secrets Manager; see below).

### Python

```python
from car_recommender.ml.filters import apply_filters, get_recommendations
from car_recommender.ml.model_store import ModelStore
from car_recommender.core.config import get_settings

store = ModelStore.load(get_settings())
results = get_recommendations(
    {"bodytype": "SUV", "max_price": 40000, "min_mpg_comb": 25, "min_year": 2020, "Drivetrain": "AWD"},
    store.df,
    top_n=5,
)
```

### Model evaluation (no retraining)

```bash
python -m car_recommender.pipeline.evaluate_model
```
Loads saved `.joblib` and reports: overall metrics on all reviewed cars, year-era breakdown (pre/post 2015), per-bodytype RMSE, per-make RMSE (best/worst 10), bucket accuracy confusion matrix, prediction vs actual distribution, mean bias. Primary train/test metrics are stored in `models/model_meta.json`.

---

## Used-Market Price (optional)

`price` in the dataset is new-car MSRP, not what a used buyer actually pays. `src/car_recommender/scraping/kbb.py` + `pipeline/used_price.py` scrape Kelley Blue Book for used-market price estimates, joined into `model_ready.csv` as `used_price_est` / `used_price_missing` when present.

This stage is **off by default** and network/browser-dependent — it currently covers a small hand-picked sample of makes/models (see `CARS`/`YEARS` in `scraping/kbb.py`), not the full catalog. Expanding coverage is a deliberate follow-up given how slow, fragile, and rate-limit-sensitive live scraping is — run it yourself and iterate:

```bash
python -m car_recommender.pipeline.runner --only scrape-used-price
python -m car_recommender.pipeline.runner --only export-final   # re-join the results
```

---

## AWS Deployment

Real AWS usage, not just a mention: **S3** for model/data artifacts, **ECR** for the API's container image, **App Runner** to run it, **Secrets Manager** for `ANTHROPIC_API_KEY`/`API_PASSWORD`, and **GitHub Actions OIDC** (no stored AWS keys) to deploy on every push to `main`.

```
src/car_recommender/ml/model_store.py    MODEL_STORE_BACKEND=s3 → downloads artifacts from S3
src/car_recommender/core/secrets.py      DEPLOY_ENV=aws → fetches secrets from Secrets Manager
infra/main.tf                            Terraform: S3 bucket, ECR repo, App Runner service,
                                          Secrets Manager containers, GitHub OIDC role
.github/workflows/deploy.yml             Builds/pushes the image, redeploys App Runner
```

**Setup** (apply with your own AWS account — this isn't run from a coding session):

```bash
cd infra
terraform init
terraform apply

# populate the two secrets (never stored in .tf files or state)
aws secretsmanager put-secret-value --secret-id car-recommender/anthropic-api-key --secret-string "sk-ant-..."
aws secretsmanager put-secret-value --secret-id car-recommender/api-password      --secret-string "<a password>"

# upload model/data artifacts once
aws s3 cp models/rf_recommender.joblib s3://$(terraform output -raw artifacts_bucket)/models/rf_recommender.joblib
aws s3 cp models/label_encoders.joblib s3://$(terraform output -raw artifacts_bucket)/models/label_encoders.joblib
aws s3 cp models/feature_list.json     s3://$(terraform output -raw artifacts_bucket)/models/feature_list.json
aws s3 cp data/final/model_ready.csv   s3://$(terraform output -raw artifacts_bucket)/data/model_ready.csv
```

Then set these as GitHub repo **variables** (Settings → Secrets and variables → Actions → Variables) to arm `deploy.yml` — until `AWS_ROLE_ARN` is set, that workflow no-ops rather than failing, so `ci.yml`'s badge is unaffected either way:

| Variable | Value |
|---|---|
| `AWS_ROLE_ARN` | `terraform output -raw github_actions_role_arn` |
| `AWS_ECR_REPOSITORY_URL` | `terraform output -raw ecr_repository_url` |
| `AWS_APPRUNNER_SERVICE_ARN` | `terraform output -raw apprunner_service_arn` |
| `AWS_REGION` | your region (default `us-east-1`) |

---

## Project Structure

```
├── src/car_recommender/
│   ├── core/          # Settings, logging, AWS Secrets Manager loader, sanitization, shared paths
│   ├── schemas/        # Pydantic request/response models
│   ├── ml/              # ModelStore (local/S3), filter + ranking logic
│   ├── llm/             # Advisor system prompt + tool-use loop, tool schema/validation
│   ├── api/              # FastAPI app + routers (cars, recommendations, advisor)
│   ├── pipeline/          # The 10-stage ETL/training pipeline + runner + evaluate_model + shap_explain
│   └── scraping/           # Selenium driver + KBB scraper + used-price orchestration
├── frontend/
│   └── streamlit_app.py    # Thin HTTP client UI
├── tests/                    # Fixture-based — no data/, no AWS, no ANTHROPIC_API_KEY required
├── infra/
│   └── main.tf                # Terraform: S3, ECR, App Runner, Secrets Manager, GitHub OIDC role
├── .github/workflows/
│   ├── ci.yml                   # Lint + test + docker build — drives the badge above
│   └── deploy.yml                # OIDC deploy to AWS, gated on AWS_ROLE_ARN being set
├── Dockerfile
├── data/                            # raw/processed/final CSVs (not tracked in git)
├── models/                            # Trained model artifacts (tracked in git)
├── notebooks/                          # EDA notebooks
├── reports/shap/                        # SHAP visualizations
└── schema/                                # Column/category definitions
```

---

## Key Design Decisions

**Consumer columns only.** Technical specs (engine dimensions, wheel base, fuel capacity) were excluded. The dataset is scoped to columns that actually influence purchase decisions.

**Expert and consumer reviews serve different roles.** Expert ratings (6 dimensions) are model *features* — they describe measurable attributes. Consumer overall rating is the model *target* — it captures buyer satisfaction. This separation means the model learns the relationship between specs/expert opinion and real buyer response, without predicting one rating source from another.

**No data leakage.** An earlier version included `consumer_overall_rating` as both a feature and part of the target, artificially inflating R² to 0.979. After correction, the model is driven purely by spec and expert signals.

**ML imputation for hp/torque.** Simple group median was replaced with a trained RF imputer that incorporates brand origin (American/European/Japanese/Korean) as a feature. Predictions are snapped to nearest known real value for same make+model+year where sibling trim data exists.

**Two-stage split design.** The model is built for a fixed used-car catalog. A temporal split would penalise the model for not predicting the future, which is irrelevant here. A naive stratified random split would let the same make+model appear in both train and test, inflating Spearman to ~0.98 through memorization rather than genuine learning. The two-stage approach resolves both: a group-aware holdout (20% of make+model groups, never seen in training) serves as a lower-bound diagnostic at Spearman=0.54, while the stratified random primary split (same groups, different rows) gives the deployment-realistic metric at Spearman=0.98. Both numbers are reported and meaningful.

**Imputation flags preserved.** `hp_missing` and `torque_missing` are kept in the dataset and surfaced in the API/UI, but excluded from model training — they reflect data collection gaps, not car quality.

**Pickup truck cargo.** `cargo_cuft` is structurally absent for traditional pickups. Filled with 0 rather than imputed — correctly reflects that cargo volume is not a relevant metric for that body type.

**FastAPI as the single source of truth.** v1 loaded the model and ran filter logic independently inside the Streamlit app, the CLI, and the chat script — three copies of the same logic, three places for it to drift. v2 moves inference, filtering, and the LLM advisor into one FastAPI backend; the Streamlit UI is a thin HTTP client with no model-loading code, and the standalone CLIs were retired.

**Tests never touch real data.** `data/` is gitignored and not present in CI. Tests build a small synthetic catalog + a deterministic fake model (`tests/conftest.py`) and mock the Anthropic client and AWS (`moto`) — the CI badge reflects code correctness, not whether a real dataset happens to be checked out.

---

## Known Limitations

- **Price is MSRP, not used market price** — the dataset contains new car list prices; `used_price_est` (optional, see [Used-Market Price](#used-market-price-optional)) currently covers a small sample, not the full catalog.
- **Consumer ratings from a single source** — aggregated from one review site; skews toward enthusiast and dissatisfied buyers, not the full buyer population.
- **Low-rated cars underrepresented** — consumer ratings are right-skewed (most cars 4.0+) due to selection bias: people tend to rate cars they bought. Low-bucket (<3.5) recall is structurally weaker than Mid/High/Top — a data problem, not a model problem.
- **Mercedes-Benz consistently highest RMSE** — consumer ratings diverge from spec-based expectations more than any other make.
- **Rate limiter is single-process, in-memory** — fine for one App Runner instance; a multi-instance deployment would want a shared store (e.g. Redis) instead.

---

## Next Steps

- **Full-catalog used-price scraping** — expand `scraping/kbb.py`'s coverage beyond the current sample
- **Reliability weighting** — expose `rating_reliability` as a user-adjustable filter priority, given its outsized importance for used car buyers
- **NDCG evaluation** — ranking metric against held-out user preference data for proper recommendation quality measurement
- **Shared rate-limit store** — move off in-memory rate limiting for multi-instance deployments
