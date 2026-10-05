import pickle

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from telecom_churn.etl.transform import transform


@pytest.fixture
def raw_df() -> pd.DataFrame:
    data = {
        "customerID": ["1", "2", "3", "4", "5"],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No", "No"],
        "tenure": [1, 34, 2, 45, 2],
        "PhoneService": ["No", "Yes", "Yes", "No", "Yes"],
        "MultipleLines": [
            "No phone service",
            "No",
            "No",
            "No phone service",
            "No",
        ],
        "InternetService": ["DSL", "DSL", "DSL", "DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "Yes", "Yes", "No"],
        "OnlineBackup": ["Yes", "No", "Yes", "No", "No"],
        "DeviceProtection": ["No", "Yes", "No", "Yes", "No"],
        "TechSupport": ["No", "No", "No", "Yes", "No"],
        "StreamingTV": ["No", "No", "No", "No", "No"],
        "StreamingMovies": ["No", "No", "No", "No", "No"],
        "Contract": [
            "Month-to-month",
            "One year",
            "Month-to-month",
            "One year",
            "Month-to-month",
        ],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Electronic check",
        ],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70],
        "TotalCharges": ["29.85", "1889.5", "108.15", "1840.75", "151.65"],
        "Churn": ["No", "No", "Yes", "No", "Yes"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def processed_df(raw_df) -> pd.DataFrame:
    return transform(raw_df)


@pytest.fixture
def sample_features(processed_df) -> pd.DataFrame:
    return processed_df.drop(columns=["Churn"])


@pytest.fixture
def trained_sklearn_model(processed_df, tmp_path):
    X = processed_df.drop(columns=["Churn"])
    y = processed_df["Churn"]
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    model_path = tmp_path / "test_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model, str(model_path)
