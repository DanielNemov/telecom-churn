import os
import sys
import time
import psutil
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "etl"))
from load import load_processed

REPORTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "reports"
)
DRIFT_REPORT_PATH = os.path.join(REPORTS_DIR, "drift_report.html")
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    os.path.join(os.path.dirname(__file__), "..", "..", "mlruns"),
)


def monitor_data_drift(df: pd.DataFrame) -> None:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset

    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=train_df, current_data=test_df)
    report.save_html(DRIFT_REPORT_PATH)
    print(f"Drift report saved to {DRIFT_REPORT_PATH}")


def monitor_infrastructure() -> dict:
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    infra_metrics = {
        "cpu_percent": cpu_percent,
        "memory_used_gb": round(mem.used / (1024 ** 3), 2),
        "memory_total_gb": round(mem.total / (1024 ** 3), 2),
        "memory_percent": mem.percent,
        "disk_used_gb": round(disk.used / (1024 ** 3), 2),
        "disk_total_gb": round(disk.total / (1024 ** 3), 2),
        "disk_percent": disk.percent,
    }

    print("Infrastructure metrics:")
    for k, v in infra_metrics.items():
        print(f"  {k}: {v}")

    return infra_metrics


def run_monitoring() -> None:
    df = load_processed()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("telecom-churn-monitoring")

    with mlflow.start_run(run_name="monitoring"):
        start = time.time()
        monitor_data_drift(df)
        drift_time = round(time.time() - start, 2)

        infra = monitor_infrastructure()

        mlflow.log_metrics(infra)
        mlflow.log_metric("drift_report_time_sec", drift_time)
        mlflow.log_artifact(DRIFT_REPORT_PATH)

        print(f"Monitoring run complete in {drift_time}s")


if __name__ == "__main__":
    run_monitoring()
