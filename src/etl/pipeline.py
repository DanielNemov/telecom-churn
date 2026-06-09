from extract import extract
from transform import transform
from load import load


def run_etl() -> str:
    raw = extract()
    processed = transform(raw)
    dest = load(processed)
    return dest


if __name__ == "__main__":
    path = run_etl()
    print(f"ETL complete. Processed file: {path}")
