import os
import urllib.request
import pandas as pd

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
RAW_FILE = os.path.join(RAW_DIR, "telco_churn.csv")


def download_data(url: str = RAW_DATA_URL, dest: str = RAW_FILE) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading dataset from {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")
    else:
        print(f"Raw file already exists: {dest}")
    return dest


def extract(path: str = RAW_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        download_data(dest=path)
    df = pd.read_csv(path)
    print(f"Extracted {len(df)} rows, {df.shape[1]} columns from {path}")
    return df


if __name__ == "__main__":
    df = extract()
    print(df.head())
