"""
main.py — Car Specs pipeline runner

Stages (in dependency order):
  1. standardize     scripts/standardize.py       → data/processed/
  2. split-reviews   scripts/split_reviews.py     → data/final/reviews/
  3. split-cars      scripts/split_cars.py        → data/final/cars/
  4. zero-to-sixty   scripts/zero_to_sixty.py     → data/final/cars/zero_to_sixty_est.csv
  5. zero-to-sixty-ml scripts/zero_to_sixty_ml.py → data/final/cars/zero_to_sixty_ml.csv  (slow)

Usage:
  python main.py                # run all stages
  python main.py --skip-ml      # skip the XGBoost stage (stages 1–4)
  python main.py --only split-cars
  uv run car-specs --skip-ml
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

STAGES = [
    ("standardize",      ROOT / "scripts" / "standardize.py"),
    ("split-reviews",    ROOT / "scripts" / "split_reviews.py"),
    ("split-cars",       ROOT / "scripts" / "split_cars.py"),
    ("zero-to-sixty",    ROOT / "scripts" / "zero_to_sixty.py"),
    ("zero-to-sixty-ml", ROOT / "scripts" / "zero_to_sixty_ml.py"),
]


def run_stage(name: str, script: Path) -> bool:
    print(f"\n{'─' * 60}")
    print(f"  Stage: {name}")
    print(f"  Script: {script.relative_to(ROOT)}")
    print(f"{'─' * 60}")
    result = subprocess.run([sys.executable, str(script)])
    if result.returncode != 0:
        print(f"\n[ERROR] Stage '{name}' failed (exit {result.returncode}). Stopping.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the car-specs data pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {name}" for name, _ in STAGES),
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip the zero-to-sixty-ml stage (XGBoost, slow).",
    )
    parser.add_argument(
        "--only",
        metavar="STAGE",
        choices=[name for name, _ in STAGES],
        help="Run only a single named stage.",
    )
    args = parser.parse_args()

    if args.only:
        stages_to_run = [(n, s) for n, s in STAGES if n == args.only]
    elif args.skip_ml:
        stages_to_run = [(n, s) for n, s in STAGES if n != "zero-to-sixty-ml"]
    else:
        stages_to_run = STAGES

    print(f"Running {len(stages_to_run)} stage(s)…")
    for name, script in stages_to_run:
        if not run_stage(name, script):
            sys.exit(1)

    print("\nAll stages completed successfully.")


if __name__ == "__main__":
    main()
