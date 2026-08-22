"""GET /cars/filters, GET /cars/{car_id} — catalog browsing endpoints."""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from car_recommender.api.dependencies import get_model_store, require_api_key
from car_recommender.ml.model_store import ModelStore
from car_recommender.schemas.car import CarDetail, ExpertRatings, FeatureAvailability, FilterOptions

router = APIRouter(prefix="/cars", tags=["cars"], dependencies=[Depends(require_api_key)])

FEATURE_COLS = {
    "safety": [
        "Child Seat Anchors", "Child Door Locks", "Traction Control",
        "Stability Control", "Hill Start Assist", "Blind-Spot Alert",
        "Collision Warning System",
    ],
    "connectivity": [
        "Bluetooth Wireless Technology", "Hands Free Phone",
        "Bluetooth Streaming Audio", "Satellite Radio", "Smartphone Interface",
        "Navigation System", "Voice Recognition System", "Internet Access",
        "Real-Time Traffic Information", "Premium Radio",
    ],
    "convenience": [
        "Cruise Control", "Remote Keyless Entry", "Remote Engine Start",
        "Power Windows", "Power Outlet", "Rear Window Defroster",
        "Steering Wheel Controls", "Tilt Steering Wheel",
    ],
}


@router.get("/filters", response_model=FilterOptions)
def get_filter_options(store: ModelStore = Depends(get_model_store)) -> FilterOptions:
    df = store.df
    return FilterOptions(
        bodytypes=sorted(df["bodytype"].dropna().unique()),
        makes=sorted(df["make"].dropna().unique()),
        fuel_types=sorted(df["Fuel Type"].dropna().unique()),
        drivetrains=sorted(df["Drivetrain"].dropna().unique()),
        transmissions=sorted(df["Transmission Type"].dropna().unique()),
        price_min=float(df["price"].min()),
        price_max=float(df["price"].max()),
        mpg_max=float(df["mpg_comb"].max()),
        hp_max=float(df["hp"].max()),
        year_min=int(df["year"].min()),
        year_max=int(df["year"].max()),
        predicted_rating_min=float(df["predicted_consumer_rating"].min()),
        predicted_rating_max=float(df["predicted_consumer_rating"].max()),
    )


@router.get("/{car_id}", response_model=CarDetail)
def get_car_detail(car_id: int, store: ModelStore = Depends(get_model_store)) -> CarDetail:
    matches = store.df[store.df["car_id"] == car_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"No car with car_id={car_id}.")
    row = matches.iloc[0]

    def feature_group(cols: list[str]) -> dict[str, int]:
        return {c: (int(row[c]) if pd.notna(row.get(c)) else 0) for c in cols}

    return CarDetail(
        car_id=int(row["car_id"]),
        make=row["make"],
        model=row["model"],
        year=int(row["year"]),
        trim=row.get("trim") if pd.notna(row.get("trim")) else None,
        bodytype=row["bodytype"],
        price=float(row["price"]),
        mpg_city=float(row["mpg_city"]),
        mpg_hwy=float(row["mpg_hwy"]),
        mpg_comb=float(row["mpg_comb"]),
        hp=float(row["hp"]),
        hp_imputed=bool(row["hp_missing"]),
        torque_lbft=float(row["torque_lbft"]),
        torque_imputed=bool(row["torque_missing"]),
        cargo_cuft=float(row["cargo_cuft"]) if pd.notna(row.get("cargo_cuft")) else None,
        drivetrain=row["Drivetrain"],
        transmission_type=row["Transmission Type"],
        fuel_type=row["Fuel Type"],
        doors=int(row["doors"]),
        is_ev=bool(row["is_ev"]),
        luxury=bool(row["luxury"]),
        engine=row.get("Engine") if pd.notna(row.get("Engine")) else None,
        expert_ratings=ExpertRatings(
            value=float(row["rating_value"]),
            performance=float(row["rating_performance"]),
            quality=float(row["rating_quality"]),
            comfort=float(row["rating_comfort"]),
            reliability=float(row["rating_reliability"]),
            styling=float(row["rating_styling"]),
        ),
        consumer_overall_rating=(
            float(row["consumer_overall_rating"]) if pd.notna(row.get("consumer_overall_rating")) else None
        ),
        consumer_review_count=(
            int(row["consumer_review_count"]) if pd.notna(row.get("consumer_review_count")) else 0
        ),
        predicted_consumer_rating=float(row["predicted_consumer_rating"]),
        features=FeatureAvailability(
            safety=feature_group(FEATURE_COLS["safety"]),
            connectivity=feature_group(FEATURE_COLS["connectivity"]),
            convenience=feature_group(FEATURE_COLS["convenience"]),
        ),
    )
