"""car_recommender.pipeline.runner — Car Specs pipeline runner

Pipeline stages (in dependency order):

  1. standardize         raw -> data/processed/
                         Type cleanup, ordinal encoding, car_id assignment.

  2. split-reviews       processed -> data/final/reviews/
                         Splits review text and ratings; assigns review_id.

  3. split-cars          processed -> data/final/cars/
                         Joins review_id FK; outputs car_stats.csv + car_features.csv.

  4. impute-hp-torque    patches data/final/cars/car_stats.csv
                         Manual hp/torque lookup for rare/luxury/EV models.

  5. ml-impute-hp-torque patches data/final/cars/car_stats.csv
                         RF-based imputation for remaining hp/torque nulls.
                         Brand origin groups (American/European/Japanese/Korean)
                         used as features. Snaps predictions to nearest known
                         real value for same make+model+year where possible.

  6. zero-to-sixty       -> data/final/cars/zero_to_sixty_est.csv
                         Physics-based 0-60 estimator for cars with hp > 180.

  7. build-consumer-view -> data/final/consumer_view.csv
                         Imputes all consumer columns, joins features + reviews.

  8. fill-consumer-view  patches consumer_view.csv in-place
                         Aggressive null fill (ratings, mpg, hp, torque, doors, Transmission).

  9. export-final        -> data/final/model_ready.csv       <- zero-null model input
                         Fills Pickup cargo_cuft=0, drops phantom rows.

  10. train-recommender  -> models/rf_recommender.joblib + artifacts
                         Trains RF quality ranker; exposes recommend() function.

Optional, off by default (network + browser dependent — see scraping/used_price.py):

  scrape-used-price      -> data/final/cars/used_price.csv
                         Used-market price estimates via KBB. Run explicitly with
                         --only scrape-used-price; export-final joins it when present.

Usage:
  python -m car_recommender.pipeline.runner                    # run all default stages
  python -m car_recommender.pipeline.runner --skip stage-name   # skip one stage
  python -m car_recommender.pipeline.runner --only stage-name   # run a single stage
  car-recommender-pipeline                                      # console script (see pyproject.toml)
"""

import argparse
import subprocess
import sys

STAGES = [
    ("standardize",          "car_recommender.pipeline.standardize"),
    ("split-reviews",        "car_recommender.pipeline.split_reviews"),
    ("split-cars",           "car_recommender.pipeline.split_cars"),
    ("impute-hp-torque",     "car_recommender.pipeline.impute_hp_torque"),
    ("ml-impute-hp-torque",  "car_recommender.pipeline.ml_impute_hp_torque"),
    ("zero-to-sixty",        "car_recommender.pipeline.zero_to_sixty"),
    ("build-consumer-view",  "car_recommender.pipeline.build_consumer_view"),
    ("fill-consumer-view",   "car_recommender.pipeline.fill_consumer_view"),
    ("export-final",         "car_recommender.pipeline.export_final"),
    ("train-recommender",    "car_recommender.pipeline.train_recommender"),
]

# Optional stages: never run as part of a full pipeline pass, only via --only.
OPTIONAL_STAGES = [
    ("scrape-used-price",    "car_recommender.pipeline.used_price"),
]


def run_stage(name: str, module: str) -> bool:
    print(f"\n{'─' * 60}")
    print(f"  Stage : {name}")
    print(f"  Module: {module}")
    print(f"{'─' * 60}")
    result = subprocess.run([sys.executable, "-m", module], check=False)  # returncode handled explicitly below
    if result.returncode != 0:
        print(f"\n[ERROR] Stage '{name}' failed (exit {result.returncode}). Stopping.")
        return False
    return True


def main():
    all_stages = STAGES + OPTIONAL_STAGES
    stage_names = [name for name, _ in all_stages]

    parser = argparse.ArgumentParser(
        description="Run the car-recommender data pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages:\n" + "\n".join(f"  {n}" for n in stage_names),
    )
    parser.add_argument(
        "--only",
        metavar="STAGE",
        choices=stage_names,
        help="Run only a single named stage (required for optional stages like scrape-used-price).",
    )
    parser.add_argument(
        "--skip",
        metavar="STAGE",
        choices=[n for n, _ in STAGES],
        help="Skip one default stage and run the rest.",
    )
    args = parser.parse_args()

    if args.only:
        stages_to_run = [(n, m) for n, m in all_stages if n == args.only]
    elif args.skip:
        stages_to_run = [(n, m) for n, m in STAGES if n != args.skip]
    else:
        stages_to_run = STAGES

    print(f"Running {len(stages_to_run)} stage(s)…")
    for name, module in stages_to_run:
        if not run_stage(name, module):
            sys.exit(1)

    print("\nAll stages completed successfully.")


if __name__ == "__main__":
    main()
