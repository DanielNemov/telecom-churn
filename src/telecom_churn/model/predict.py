from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

from telecom_churn.config import MODEL_PATH, TARGET_COL
from telecom_churn.etl.load import load_processed
from telecom_churn.logging_config import setup_logging

logger = logging.getLogger(__name__)


def load_model(path: Path | str = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    logger.info("Loaded model '%s' from %s", payload["model_name"], path)
    return payload


def predict(df: pd.DataFrame, payload: dict | None = None) -> pd.DataFrame:
    if payload is None:
        payload = load_model()
    model = payload["model"]
    result = df.copy()
    result["prediction_label"] = model.predict(df)
    result["prediction_score"] = model.predict_proba(df)[:, 1]
    return result


def main() -> None:
    setup_logging()
    df = load_processed()
    sample = df.drop(columns=[TARGET_COL]).head(5)
    result = predict(sample)
    print(result[["prediction_label", "prediction_score"]].to_string())


if __name__ == "__main__":
    main()
