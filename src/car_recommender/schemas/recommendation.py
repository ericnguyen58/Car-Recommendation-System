from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    make: list[str] | None = None
    bodytype: list[str] | None = None
    fuel_type: list[str] | None = None
    drivetrain: list[str] | None = None
    transmission_type: list[str] | None = None
    is_ev: bool | None = None
    luxury: bool | None = None
    min_price: float | None = None
    max_price: float | None = None
    min_mpg_comb: float | None = None
    max_mpg_comb: float | None = None
    min_hp: float | None = None
    max_hp: float | None = None
    min_year: int | None = None
    max_year: int | None = None
    min_cargo_cuft: float | None = None
    min_predicted_rating: float | None = None
    max_predicted_rating: float | None = None
    top_n: int = Field(default=10, ge=1, le=50)


class CarSummary(BaseModel):
    car_id: int
    make: str
    model: str
    year: int
    trim: str | None = None
    bodytype: str
    fuel_type: str
    drivetrain: str
    price: float
    mpg_comb: float
    hp: float
    rating_reliability: float | None = None
    consumer_overall_rating: float | None = None
    predicted_consumer_rating: float


class RecommendationResponse(BaseModel):
    total_matched: int
    cars: list[CarSummary]
