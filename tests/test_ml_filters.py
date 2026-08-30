from car_recommender.ml.filters import apply_filters, get_recommendations


def test_apply_filters_categorical(sample_df):
    result = apply_filters({"bodytype": "Sedan"}, sample_df)
    assert set(result["make"]) == {"Toyota", "Honda", "Tesla"}


def test_apply_filters_fuel_type_remap(sample_df):
    result = apply_filters({"fuel_type": ["electric"]}, sample_df)
    assert list(result["make"]) == ["Tesla"]


def test_apply_filters_numeric_range(sample_df):
    result = apply_filters({"max_price": 26000}, sample_df)
    assert set(result["make"]) == {"Toyota", "Honda"}


def test_apply_filters_is_ev(sample_df):
    result = apply_filters({"is_ev": True}, sample_df)
    assert list(result["make"]) == ["Tesla"]


def test_apply_filters_invalid_numeric_is_ignored(sample_df):
    result = apply_filters({"max_price": "not-a-number"}, sample_df)
    assert len(result) == len(sample_df)


def test_get_recommendations_empty_result(sample_df):
    out = get_recommendations({"make": "Nonexistent"}, sample_df)
    assert out == {"count": 0, "message": "No cars matched those filters. Try relaxing one or more criteria."}


def test_get_recommendations_ranks_and_dedupes(model_store):
    # model_store.df carries predicted_consumer_rating (ModelStore computes it on construction);
    # FakeModel scores purely by hp, so Tesla (hp=346) should rank first.
    out = get_recommendations({}, model_store.df, top_n=10)
    assert out["total_matched"] == len(model_store.df)
    assert out["count"] == len(model_store.df)  # each fixture row is already a distinct make+model
    makes_in_order = [c["make"] for c in out["cars"]]
    assert makes_in_order[0] == "Tesla"
