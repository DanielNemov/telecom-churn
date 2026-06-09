import os
import pickle
import pandas as pd

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "best_model.pkl"
)


def load_model(path: str = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    print(f"Loaded model '{payload['model_name']}' from {path}")
    return payload


def predict(df: pd.DataFrame, payload: dict = None) -> pd.DataFrame:
    if payload is None:
        payload = load_model()
    model = payload["model"]
    proba = model.predict_proba(df)[:, 1]
    labels = model.predict(df)
    result = df.copy()
    result["prediction_label"] = labels
    result["prediction_score"] = proba
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "etl"))
    from load import load_processed

    df = load_processed()
    sample = df.drop(columns=["Churn"]).head(5)
    result = predict(sample)
    print(result[["prediction_label", "prediction_score"]].to_string())
