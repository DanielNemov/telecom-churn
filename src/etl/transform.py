import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


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

    df["Churn"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)

    df["gender"] = (df["gender"].str.strip().str.lower() == "male").astype(int)

    for col in BINARY_YES_NO:
        if col in df.columns:
            df[col] = (df[col].str.strip().str.lower() == "yes").astype(int)

    le = LabelEncoder()
    for col in MULTI_SERVICE_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    scaler = StandardScaler()
    existing_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    df[existing_numeric] = scaler.fit_transform(df[existing_numeric])

    print(f"Transformed dataset shape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
    return df


if __name__ == "__main__":
    from extract import extract

    raw = extract()
    processed = transform(raw)
    print(processed.head())
