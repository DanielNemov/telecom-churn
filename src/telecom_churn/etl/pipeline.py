from pathlib import Path

from telecom_churn.etl.extract import extract
from telecom_churn.etl.load import load
from telecom_churn.etl.transform import transform
from telecom_churn.logging_config import setup_logging


def run_etl() -> Path:
    setup_logging()
    raw = extract()
    processed = transform(raw)
    dest = load(processed)
    return dest


if __name__ == "__main__":
    path = run_etl()
    print(f"ETL complete. Processed file: {path}")
