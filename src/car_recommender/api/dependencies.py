"""Shared FastAPI dependencies: the model store, API-key auth, and a simple
in-memory per-client rate limiter (mirrors app.py's original per-session gate,
now scoped per client instead of per Streamlit session)."""

import hmac
import time
from collections import defaultdict

from fastapi import Depends, Header, HTTPException, Request, status

from car_recommender.core.config import Settings, get_settings
from car_recommender.ml.model_store import ModelStore

# client identifier -> request timestamps (seconds). Single-process only —
# fine for one instance; a multi-instance deployment would want a shared
# store (e.g. Redis) instead. Documented as a known limitation in the README.
_request_log: dict[str, list[float]] = defaultdict(list)


def get_model_store(request: Request) -> ModelStore:
    return request.app.state.model_store


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.API_PASSWORD:
        return  # no password configured = open (local dev), matches the original app.py gate
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.API_PASSWORD):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key.")


def enforce_rate_limit(request: Request, settings: Settings = Depends(get_settings)) -> None:
    client_id = request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")
    now = time.time()
    timestamps = [t for t in _request_log[client_id] if now - t < settings.RATE_WINDOW_SECONDS]
    if len(timestamps) >= settings.RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait before trying again.",
        )
    timestamps.append(now)
    _request_log[client_id] = timestamps
