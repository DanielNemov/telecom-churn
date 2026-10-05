from __future__ import annotations

import logging
import time

import mlflow
import pandas as pd
import psutil
from sklearn.model_selection import train_test_split

from telecom_churn.config import (
    DRIFT_REPORT_PATH,
    MLFLOW_MONITOR_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    ensure_dirs,
)
from telecom_churn.etl.load import load_processed
from telecom_churn.logging_config import setup_logging

logger = logging.getLogger(__name__)


def monitor_data_drift(df: pd.DataFrame) -> None:
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.report import Report

    ensure_dirs()
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=train_df, current_data=test_df)
    report.save_html(str(DRIFT_REPORT_PATH))
    logger.info("Drift report saved to %s", DRIFT_REPORT_PATH)


def monitor_infrastructure() -> dict:
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    infra_metrics = {
        "cpu_percent": cpu_percent,
        "memory_used_gb": round(mem.used / (1024**3), 2),
        "memory_total_gb": round(mem.total / (1024**3), 2),
        "memory_percent": mem.percent,
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_percent": disk.percent,
    }
    logger.info("Infrastructure metrics: %s", infra_metrics)
    return infra_metrics


def run_monitoring() -> None:
    setup_logging()
    ensure_dirs()
    df = load_processed()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_MONITOR_EXPERIMENT)

    with mlflow.start_run(run_name="monitoring"):
        start = time.time()
        monitor_data_drift(df)
        drift_time = round(time.time() - start, 2)
        infra = monitor_infrastructure()
        mlflow.log_metrics(infra)
        mlflow.log_metric("drift_report_time_sec", drift_time)
        if DRIFT_REPORT_PATH.exists():
            mlflow.log_artifact(str(DRIFT_REPORT_PATH))
        logger.info("Monitoring run complete in %ss", drift_time)
        logger.info("Reports dir: %s", REPORTS_DIR)


if __name__ == "__main__":
    run_monitoring()
