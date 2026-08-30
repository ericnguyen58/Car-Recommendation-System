"""Audit logging — no API keys, no secrets, no full conversation content.

Moved from app.py's basicConfig call so both the API and any other entry
point can share one logging setup.
"""

import logging

from car_recommender.core.config import Settings

_LOG = logging.getLogger("car_recommender")


def setup_logging(settings: Settings) -> logging.Logger:
    settings.LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=settings.LOGS_DIR / "advisor.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return _LOG


def get_logger() -> logging.Logger:
    return _LOG
