"""Single source of truth for the repository's on-disk layout.

Every module that used to compute its own project root via
``Path(__file__).resolve().parents[N]`` imports ``PROJECT_ROOT`` from here
instead — one constant to fix if the package ever moves, rather than N
copies of fragile relative-parent arithmetic scattered across the pipeline.
"""

from pathlib import Path

# This file lives at src/car_recommender/core/paths.py, so counting parent
# directories: [0]=core, [1]=car_recommender, [2]=src, [3]=repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_FINAL = DATA_DIR / "final"

MODELS_DIR = PROJECT_ROOT / "models"
SCHEMA_DIR = PROJECT_ROOT / "schema"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
