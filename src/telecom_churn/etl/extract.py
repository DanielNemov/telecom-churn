from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pandas as pd

from telecom_churn.config import RAW_DATA_URL, RAW_FILE, ensure_dirs

logger = logging.getLogger(__name__)


def download_data(url: str = RAW_DATA_URL, dest: Path | str = RAW_FILE) -> Path:
    dest_path = Path(dest)
    ensure_dirs()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        logger.info("Downloading dataset from %s", url)
        urllib.request.urlretrieve(url, dest_path)
        logger.info("Saved to %s", dest_path)
    else:
        logger.info("Raw file already exists: %s", dest_path)
    return dest_path


def extract(path: Path | str = RAW_FILE) -> pd.DataFrame:
    dest_path = Path(path)
    if not dest_path.exists():
        download_data(dest=dest_path)
    df = pd.read_csv(dest_path)
    logger.info("Extracted %s rows, %s columns from %s", len(df), df.shape[1], dest_path)
    return df


if __name__ == "__main__":
    from telecom_churn.logging_config import setup_logging

    setup_logging()
    print(extract().head())
