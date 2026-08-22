"""Verifies ModelStore's S3 backend without any real AWS account — moto
mocks the AWS API entirely, so this proves the S3 code path works before
the user ever touches a real bucket."""

import io
import json

import boto3
import joblib
from moto import mock_aws

from car_recommender.core.config import Settings
from car_recommender.ml.model_store import ModelStore

BUCKET = "test-car-recommender-artifacts"
REGION = "us-east-1"


class FakeModel:
    """Module-level (not nested in the test function) so joblib can pickle it."""

    def predict(self, X):
        return (3.0 + X["hp"] / 1000).to_numpy()


@mock_aws
def test_model_store_loads_from_s3(sample_df, fake_encoders):
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)

    model_bytes = io.BytesIO()
    joblib.dump(FakeModel(), model_bytes)
    s3.put_object(Bucket=BUCKET, Key="models/rf_recommender.joblib", Body=model_bytes.getvalue())

    encoders_bytes = io.BytesIO()
    joblib.dump(fake_encoders, encoders_bytes)
    s3.put_object(Bucket=BUCKET, Key="models/label_encoders.joblib", Body=encoders_bytes.getvalue())

    s3.put_object(Bucket=BUCKET, Key="models/feature_list.json", Body=json.dumps(["hp"]).encode("utf-8"))

    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    s3.put_object(Bucket=BUCKET, Key="data/model_ready.csv", Body=csv_bytes)

    settings = Settings(MODEL_STORE_BACKEND="s3", AWS_S3_BUCKET=BUCKET, AWS_REGION=REGION)
    store = ModelStore.load(settings)

    assert len(store.df) == len(sample_df)
    assert "predicted_consumer_rating" in store.df.columns
    assert set(store.df["make"]) == set(sample_df["make"])


@mock_aws
def test_model_store_s3_requires_bucket():
    settings = Settings(MODEL_STORE_BACKEND="s3", AWS_S3_BUCKET="", AWS_REGION=REGION)
    try:
        ModelStore.load(settings)
        assert False, "expected RuntimeError for missing AWS_S3_BUCKET"
    except RuntimeError as e:
        assert "AWS_S3_BUCKET" in str(e)
