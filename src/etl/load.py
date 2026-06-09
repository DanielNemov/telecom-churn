import os
import pandas as pd

PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed"
)
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "telco_churn_processed.csv")


def load(df: pd.DataFrame, dest: str = PROCESSED_FILE) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"Saved {len(df)} rows to {dest}")
    return dest


def load_processed(path: str = PROCESSED_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded processed data: {df.shape}")
    return df


if __name__ == "__main__":
    from extract import extract
    from transform import transform

    raw = extract()
    processed = transform(raw)
    load(processed)
