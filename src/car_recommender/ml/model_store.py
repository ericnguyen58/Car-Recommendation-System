"""Loads the trained model, encoders, and catalog DataFrame once at startup;
computes predicted_consumer_rating; exposes both to the API layer.

Two backends, selected by Settings.MODEL_STORE_BACKEND:
- "local" (default): reads Settings.DATA_PATH / Settings.MODELS_DIR off disk,
  what this project has always done — no AWS account required.
- "s3": downloads the same four artifacts from s3://AWS_S3_BUCKET/... first,
  then loads them the same way.

Either way, callers only ever touch a constructed ModelStore's `.df` — they
never need to know which backend loaded it. Tests build one directly from an
in-memory DataFrame + a fake model (see tests/conftest.py) via the regular
constructor, without touching disk or AWS at all.
"""

import json
import tempfile
from pathlib import Path

import joblib
import pandas as pd

from car_recommender.core.config import Settings
from car_recommender.ml.filters import CATEGORICAL_FEATURES

S3_ARTIFACT_KEYS = {
    "models/rf_recommender.joblib": "rf_recommender.joblib",
    "models/label_encoders.joblib": "label_encoders.joblib",
    "models/feature_list.json": "feature_list.json",
    "data/model_ready.csv": "model_ready.csv",
}


class ModelStore:
    def __init__(self, df: pd.DataFrame, model, encoders: dict, feature_list: list[str]):
        self.model = model
        self.encoders = encoders
        self.feature_list = feature_list
        self.df = self._with_predictions(df)

    def _with_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_FEATURES:
            le = self.encoders[col]
            known = set(le.classes_)
            df[col + "_enc"] = df[col].astype(str).apply(
                lambda v, le=le, known=known: le.transform([v])[0] if v in known else -1
            )
        df["predicted_consumer_rating"] = self.model.predict(df[self.feature_list]).round(2)
        return df

    @classmethod
    def load(cls, settings: Settings) -> "ModelStore":
        if settings.MODEL_STORE_BACKEND == "s3":
            data_path, models_dir = _download_from_s3(settings)
        else:
            data_path, models_dir = settings.DATA_PATH, settings.MODELS_DIR

        model = joblib.load(models_dir / "rf_recommender.joblib")
        encoders = joblib.load(models_dir / "label_encoders.joblib")
        feature_list = json.loads((models_dir / "feature_list.json").read_text())
        df = pd.read_csv(data_path)
        return cls(df=df, model=model, encoders=encoders, feature_list=feature_list)


def _download_from_s3(settings: Settings) -> tuple[Path, Path]:
    import boto3

    if not settings.AWS_S3_BUCKET:
        raise RuntimeError("MODEL_STORE_BACKEND=s3 requires AWS_S3_BUCKET to be set.")

    s3 = boto3.client("s3", region_name=settings.AWS_REGION)
    tmp_dir = Path(tempfile.mkdtemp(prefix="car-recommender-"))
    models_dir, data_dir = tmp_dir / "models", tmp_dir / "data"
    models_dir.mkdir()
    data_dir.mkdir()

    dest_dir = {
        "rf_recommender.joblib": models_dir,
        "label_encoders.joblib": models_dir,
        "feature_list.json": models_dir,
        "model_ready.csv": data_dir,
    }
    for s3_key, filename in S3_ARTIFACT_KEYS.items():
        dest = dest_dir[filename] / filename
        s3.download_file(settings.AWS_S3_BUCKET, s3_key, str(dest))

    return data_dir / "model_ready.csv", models_dir
