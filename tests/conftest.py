"""Shared pytest fixtures — a tiny synthetic catalog + a fake model, so tests
never need the real (gitignored) data/model files, the network, or AWS."""

import pandas as pd
import pytest

from car_recommender.api.main import create_app
from car_recommender.ml.model_store import ModelStore

FEATURE_AVAILABILITY_COLS = [
    "Child Seat Anchors", "Child Door Locks", "Traction Control", "Stability Control",
    "Hill Start Assist", "Blind-Spot Alert", "Collision Warning System",
    "Bluetooth Wireless Technology", "Hands Free Phone", "Bluetooth Streaming Audio",
    "Satellite Radio", "Smartphone Interface", "Navigation System",
    "Voice Recognition System", "Internet Access", "Real-Time Traffic Information",
    "Premium Radio", "Cruise Control", "Remote Keyless Entry", "Remote Engine Start",
    "Power Windows", "Power Outlet", "Rear Window Defroster",
    "Steering Wheel Controls", "Tilt Steering Wheel",
]

_CARS = [
    {
        "car_id": 1, "make": "Toyota", "model": "Camry", "year": 2020, "trim": "LE",
        "bodytype": "Sedan", "price": 25000, "mpg_city": 28, "mpg_hwy": 39, "mpg_comb": 32,
        "hp": 203, "torque_lbft": 184, "hp_missing": 0, "torque_missing": 0, "cargo_cuft": 15.1,
        "Drivetrain": "FWD", "Transmission Type": "Automatic", "Fuel Type": "Gasoline",
        "Engine": "2.5L I4", "doors": 4, "is_ev": 0, "luxury": 0,
        "rating_value": 4.5, "rating_performance": 4.0, "rating_quality": 4.2,
        "rating_comfort": 4.3, "rating_reliability": 4.8, "rating_styling": 4.1,
        "consumer_overall_rating": 4.6, "consumer_review_count": 120,
    },
    {
        "car_id": 2, "make": "Honda", "model": "Civic", "year": 2021, "trim": "Sport",
        "bodytype": "Sedan", "price": 23000, "mpg_city": 30, "mpg_hwy": 38, "mpg_comb": 33,
        "hp": 158, "torque_lbft": 138, "hp_missing": 0, "torque_missing": 0, "cargo_cuft": 14.8,
        "Drivetrain": "FWD", "Transmission Type": "Manual", "Fuel Type": "Gasoline",
        "Engine": "2.0L I4", "doors": 4, "is_ev": 0, "luxury": 0,
        "rating_value": 4.4, "rating_performance": 3.9, "rating_quality": 4.1,
        "rating_comfort": 4.0, "rating_reliability": 4.7, "rating_styling": 4.3,
        "consumer_overall_rating": 4.5, "consumer_review_count": 95,
    },
    {
        "car_id": 3, "make": "Tesla", "model": "Model 3", "year": 2022, "trim": "Long Range",
        "bodytype": "Sedan", "price": 47000, "mpg_city": 132, "mpg_hwy": 126, "mpg_comb": 129,
        "hp": 346, "torque_lbft": 389, "hp_missing": 0, "torque_missing": 0, "cargo_cuft": 15.0,
        "Drivetrain": "AWD", "Transmission Type": "Automatic", "Fuel Type": "Electric",
        "Engine": None, "doors": 4, "is_ev": 1, "luxury": 1,
        "rating_value": 4.2, "rating_performance": 4.9, "rating_quality": 4.0,
        "rating_comfort": 4.4, "rating_reliability": 3.9, "rating_styling": 4.6,
        "consumer_overall_rating": 4.3, "consumer_review_count": 210,
    },
    {
        "car_id": 4, "make": "Ford", "model": "F-150", "year": 2019, "trim": "XLT",
        "bodytype": "Pickup", "price": 38000, "mpg_city": 19, "mpg_hwy": 24, "mpg_comb": 21,
        "hp": 290, "torque_lbft": 265, "hp_missing": 0, "torque_missing": 0, "cargo_cuft": 0.0,
        "Drivetrain": "4WD", "Transmission Type": "Automatic", "Fuel Type": "Gasoline",
        "Engine": "3.3L V6", "doors": 4, "is_ev": 0, "luxury": 0,
        "rating_value": 4.0, "rating_performance": 4.1, "rating_quality": 3.9,
        "rating_comfort": 3.8, "rating_reliability": 4.2, "rating_styling": 4.0,
        "consumer_overall_rating": 4.1, "consumer_review_count": 305,
    },
]


class FakeLabelEncoder:
    def __init__(self, classes: list[str]):
        self.classes_ = classes

    def transform(self, values):
        return [self.classes_.index(v) for v in values]


class FakeModel:
    """Deterministic stand-in for the trained RandomForestRegressor: derives a
    rating from hp alone, so ranking order in tests is stable and predictable."""

    def predict(self, X):
        return (3.0 + X["hp"] / 1000).to_numpy()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    df = pd.DataFrame(_CARS)
    for col in FEATURE_AVAILABILITY_COLS:
        df[col] = 2  # "Standard" on every fixture row
    return df


@pytest.fixture
def fake_encoders(sample_df) -> dict:
    from car_recommender.ml.filters import CATEGORICAL_FEATURES

    return {col: FakeLabelEncoder(sorted(sample_df[col].dropna().unique())) for col in CATEGORICAL_FEATURES}


@pytest.fixture
def model_store(sample_df, fake_encoders) -> ModelStore:
    return ModelStore(df=sample_df, model=FakeModel(), encoders=fake_encoders, feature_list=["hp"])


@pytest.fixture
def client(model_store):
    from fastapi.testclient import TestClient

    app = create_app(model_store=model_store)
    with TestClient(app) as test_client:
        yield test_client
