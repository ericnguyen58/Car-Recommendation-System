"""Application settings — single source of truth for every configurable
value the API, the pipeline, and the LLM advisor need.

Values load from process environment variables first, then a `.env` file
(local dev convenience), then the defaults below. Nothing here requires an
AWS account: MODEL_STORE_BACKEND and DEPLOY_ENV both default to the local
filesystem / plain-env-var path, and only switch over to S3 / Secrets
Manager when explicitly set.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from car_recommender.core.paths import DATA_FINAL
from car_recommender.core.paths import LOGS_DIR as _LOGS_DIR
from car_recommender.core.paths import MODELS_DIR as _MODELS_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Data / model artifact location ---
    # "local" reads DATA_PATH/MODELS_DIR off disk, as this project always has.
    # "s3" streams the same files from s3://AWS_S3_BUCKET/... instead — see ml/model_store.py.
    MODEL_STORE_BACKEND: str = "local"
    DATA_PATH: Path = DATA_FINAL / "model_ready.csv"
    MODELS_DIR: Path = _MODELS_DIR
    AWS_S3_BUCKET: str = ""
    AWS_REGION: str = "us-east-1"

    # --- Secrets / deployment ---
    # "local": ANTHROPIC_API_KEY / API_PASSWORD come from the environment or .env, as today.
    # "aws": whichever of those two is still empty after env loading is fetched from Secrets Manager.
    DEPLOY_ENV: str = "local"
    ANTHROPIC_API_KEY: str = ""
    API_PASSWORD: str = ""
    ANTHROPIC_API_KEY_SECRET_NAME: str = "car-recommender/anthropic-api-key"
    API_PASSWORD_SECRET_NAME: str = "car-recommender/api-password"

    # --- LLM advisor ---
    # Cheapest current model by default for development; swap by setting this
    # one env var (ADVISOR_MODEL) — no code change needed to "upgrade" the advisor.
    ADVISOR_MODEL: str = "claude-haiku-4-5"
    MAX_TURNS: int = 20
    MAX_TOOL_LOOPS: int = 5
    MAX_INPUT_LEN: int = 1000

    # --- API hardening ---
    RATE_LIMIT: int = 30
    RATE_WINDOW_SECONDS: int = 3600
    CORS_ORIGINS: list[str] = ["*"]

    LOGS_DIR: Path = _LOGS_DIR

    def model_post_init(self, __context, /) -> None:
        if self.DEPLOY_ENV == "aws":
            from car_recommender.core.secrets import get_secret

            if not self.ANTHROPIC_API_KEY:
                self.ANTHROPIC_API_KEY = get_secret(self.ANTHROPIC_API_KEY_SECRET_NAME, self.AWS_REGION)
            if not self.API_PASSWORD:
                self.API_PASSWORD = get_secret(self.API_PASSWORD_SECRET_NAME, self.AWS_REGION)


@lru_cache
def get_settings() -> Settings:
    return Settings()
