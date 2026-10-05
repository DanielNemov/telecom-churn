from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

BINARY_YES_NO = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

MULTI_SERVICE_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["Churn"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["gender"] = (df["gender"].astype(str).str.strip().str.lower() == "male").astype(int)

    for col in BINARY_YES_NO:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip().str.lower() == "yes").astype(int)

    for col in MULTI_SERVICE_COLS:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))

    existing_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    if existing_numeric:
        scaler = StandardScaler()
        df[existing_numeric] = scaler.fit_transform(df[existing_numeric])

    logger.info("Transformed dataset shape: %s", df.shape)
    logger.info("Churn distribution:\n%s", df["Churn"].value_counts())
    return df
