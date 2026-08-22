"""FastAPI application factory — the single backend for model inference,
catalog data, and the LLM advisor. Run with:

    uvicorn car_recommender.api.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from car_recommender.api.routers import advisor, cars, recommendations
from car_recommender.core.config import get_settings
from car_recommender.core.logging import setup_logging
from car_recommender.ml.model_store import ModelStore


def create_app(model_store: ModelStore | None = None) -> FastAPI:
    """Build the app. `model_store` lets tests (or any embedder) inject an
    already-built store instead of loading real artifacts from disk/S3 at
    startup — production always calls create_app() with no argument."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings)
        app.state.settings = settings
        app.state.model_store = model_store or ModelStore.load(settings)
        yield

    app = FastAPI(title="Car Recommender API", version="2.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(cars.router)
    app.include_router(recommendations.router)
    app.include_router(advisor.router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
