from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from telecom_churn.config import PROCESSED_FILE, ensure_dirs

logger = logging.getLogger(__name__)


def load(df: pd.DataFrame, dest: Path | str = PROCESSED_FILE) -> Path:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_dirs()
    df.to_csv(dest_path, index=False)
    logger.info("Saved %s rows to %s", len(df), dest_path)
    return dest_path


def load_processed(path: Path | str = PROCESSED_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded processed data: %s", df.shape)
    return df
