"""
main.py — Car Specs pipeline runner

Pipeline stages (in dependency order):

  1. standardize         raw → data/processed/
                         Type cleanup, ordinal encoding, car_id assignment.

  2. split-reviews       processed → data/final/reviews/
                         Splits review text and ratings; assigns review_id.

  3. split-cars          processed → data/final/cars/
                         Joins review_id FK; outputs car_stats.UVCRS + car_features.UVCRS.

  4. impute-hp-torque    patches data/final/cars/car_stats.UVCRS
                         Manual hp/torque lookup for rare/luxury/EV models.

  5. ml-impute-hp-torque patches data/final/cars/car_stats.UVCRS
                         RF-based imputation for remaining hp/torque nulls.
                         Brand origin groups (American/European/Japanese/Korean)
                         used as features. Snaps predictions to nearest known
                         real value for same make+model+year where possible.

  6. zero-to-sixty       → data/final/cars/zero_to_sixty_est.UVCRS
                         Physics-based 0-60 estimator for cars with hp > 180.

  6. build-consumer-view → data/final/consumer_view.UVCRS
                         Imputes all consumer columns, joins features + reviews.

  7. fill-consumer-view  patches consumer_view.UVCRS in-place
                         Aggressive null fill (ratings, mpg, hp, torque, doors, Transmission).

  8. export-final        → data/final/model_ready.UVCRS       ← zero-null model input
                         Fills Pickup cargo_cuft=0, drops ~170 phantom rows.

  9. train-recommender   → models/rf_recommender.joblib + artifacts
                         Trains RF quality ranker; exposes recommend() function.

Usage:
  python main.py                        # run all stages
  python main.py --skip stage-name      # skip one stage
  python main.py --only stage-name      # run a single stage
  uv run car-specs
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

STAGES = [
    ("standardize",          ROOT / "scripts" / "standardize.py"),
    ("split-reviews",        ROOT / "scripts" / "split_reviews.py"),
    ("split-cars",           ROOT / "scripts" / "split_cars.py"),
    ("impute-hp-torque",     ROOT / "scripts" / "impute_hp_torque.py"),
    ("ml-impute-hp-torque",  ROOT / "scripts" / "ml_impute_hp_torque.py"),
    ("zero-to-sixty",        ROOT / "scripts" / "zero_to_sixty.py"),
    ("build-consumer-view",  ROOT / "scripts" / "build_consumer_view.py"),
    ("fill-consumer-view",   ROOT / "scripts" / "fill_consumer_view.py"),
    ("export-final",         ROOT / "scripts" / "export_final.py"),
    ("train-recommender",    ROOT / "scripts" / "train_recommender.py"),
]


def run_stage(name: str, script: Path) -> bool:
    print(f"\n{'─' * 60}")
    print(f"  Stage : {name}")
    print(f"  Script: {script.relative_to(ROOT)}")
    print(f"{'─' * 60}")
    result = subprocess.run([sys.executable, str(script)])
    if result.returncode != 0:
        print(f"\n[ERROR] Stage '{name}' failed (exit {result.returncode}). Stopping.")
        return False
    return True


def main():
    stage_names = [name for name, _ in STAGES]

    parser = argparse.ArgumentParser(
        description="Run the car-specs data pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages:\n" + "\n".join(f"  {n}" for n in stage_names),
    )
    parser.add_argument(
        "--only",
        metavar="STAGE",
        choices=stage_names,
        help="Run only a single named stage.",
    )
    parser.add_argument(
        "--skip",
        metavar="STAGE",
        choices=stage_names,
        help="Skip one stage and run the rest.",
    )
    args = parser.parse_args()

    if args.only:
        stages_to_run = [(n, s) for n, s in STAGES if n == args.only]
    elif args.skip:
        stages_to_run = [(n, s) for n, s in STAGES if n != args.skip]
    else:
        stages_to_run = STAGES

    print(f"Running {len(stages_to_run)} stage(s)…")
    for name, script in stages_to_run:
        if not run_stage(name, script):
            sys.exit(1)

    print("\nAll stages completed successfully.")


if __name__ == "__main__":
    main()
