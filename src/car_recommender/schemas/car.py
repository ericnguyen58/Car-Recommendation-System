from pydantic import BaseModel


class ExpertRatings(BaseModel):
    value: float
    performance: float
    quality: float
    comfort: float
    reliability: float
    styling: float


class FeatureAvailability(BaseModel):
    """0 = Not Available, 1 = Optional, 2 = Standard, keyed by feature name."""

    safety: dict[str, int]
    connectivity: dict[str, int]
    convenience: dict[str, int]


class CarDetail(BaseModel):
    car_id: int
    make: str
    model: str
    year: int
    trim: str | None = None
    bodytype: str
    price: float
    mpg_city: float
    mpg_hwy: float
    mpg_comb: float
    hp: float
    hp_imputed: bool
    torque_lbft: float
    torque_imputed: bool
    cargo_cuft: float | None = None
    drivetrain: str
    transmission_type: str
    fuel_type: str
    doors: int
    is_ev: bool
    luxury: bool
    engine: str | None = None
    expert_ratings: ExpertRatings
    consumer_overall_rating: float | None = None
    consumer_review_count: int
    predicted_consumer_rating: float
    features: FeatureAvailability


class FilterOptions(BaseModel):
    bodytypes: list[str]
    makes: list[str]
    fuel_types: list[str]
    drivetrains: list[str]
    transmissions: list[str]
    price_min: float
    price_max: float
    mpg_max: float
    hp_max: float
    year_min: int
    year_max: int
    predicted_rating_min: float
    predicted_rating_max: float
