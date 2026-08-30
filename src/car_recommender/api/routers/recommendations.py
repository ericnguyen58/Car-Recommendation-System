"""POST /recommendations — ranked, deduplicated-by-model car search."""

import pandas as pd
from fastapi import APIRouter, Depends

from car_recommender.api.dependencies import get_model_store, require_api_key
from car_recommender.ml.filters import apply_filters
from car_recommender.ml.model_store import ModelStore
from car_recommender.schemas.recommendation import CarSummary, RecommendationRequest, RecommendationResponse

router = APIRouter(prefix="/recommendations", tags=["recommendations"], dependencies=[Depends(require_api_key)])

# Maps the API's clean request field names to ml.filters' internal prefs keys
# (which mirror the raw DataFrame column names/casing — see ml/filters.py).
_CATEGORICAL_MAP = {
    "make": "make",
    "bodytype": "bodytype",
    "fuel_type": "fuel_type",  # apply_filters remaps this to 'Fuel Type' itself
    "drivetrain": "Drivetrain",
    "transmission_type": "Transmission Type",
}
_NUMERIC_FIELDS = [
    "min_price", "max_price", "min_mpg_comb", "max_mpg_comb",
    "min_hp", "max_hp", "min_year", "max_year", "min_cargo_cuft",
]


def _to_prefs(req: RecommendationRequest) -> dict:
    prefs: dict = {}
    for api_key, internal_key in _CATEGORICAL_MAP.items():
        val = getattr(req, api_key)
        if val:
            prefs[internal_key] = val
    if req.is_ev is not None:
        prefs["is_ev"] = req.is_ev
    if req.luxury is not None:
        prefs["luxury"] = req.luxury
    for field in _NUMERIC_FIELDS:
        val = getattr(req, field)
        if val is not None:
            prefs[field] = val
    return prefs


@router.post("", response_model=RecommendationResponse)
def get_recommendations(
    req: RecommendationRequest, store: ModelStore = Depends(get_model_store)
) -> RecommendationResponse:
    results = apply_filters(_to_prefs(req), store.df)

    if req.min_predicted_rating is not None:
        results = results[results["predicted_consumer_rating"] >= req.min_predicted_rating]
    if req.max_predicted_rating is not None:
        results = results[results["predicted_consumer_rating"] <= req.max_predicted_rating]

    top = (
        results.sort_values("predicted_consumer_rating", ascending=False)
        .drop_duplicates(subset=["make", "model"])
        .head(req.top_n)
    )

    cars = [
        CarSummary(
            car_id=int(row["car_id"]),
            make=row["make"],
            model=row["model"],
            year=int(row["year"]),
            trim=row.get("trim") if pd.notna(row.get("trim")) else None,
            bodytype=row["bodytype"],
            fuel_type=row["Fuel Type"],
            drivetrain=row["Drivetrain"],
            price=float(row["price"]),
            mpg_comb=float(row["mpg_comb"]),
            hp=float(row["hp"]),
            rating_reliability=(
                float(row["rating_reliability"]) if pd.notna(row.get("rating_reliability")) else None
            ),
            consumer_overall_rating=(
                float(row["consumer_overall_rating"]) if pd.notna(row.get("consumer_overall_rating")) else None
            ),
            predicted_consumer_rating=float(row["predicted_consumer_rating"]),
        )
        for _, row in top.iterrows()
    ]
    return RecommendationResponse(total_matched=len(results), cars=cars)
