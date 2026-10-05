import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGES_DIR = REPORTS_DIR / "images"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

RAW_FILE = RAW_DIR / "telco_churn.csv"
PROCESSED_FILE = PROCESSED_DIR / "telco_churn_processed.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.html"

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", str(MLRUNS_DIR))
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "telecom-churn")
MLFLOW_MONITOR_EXPERIMENT = os.environ.get(
    "MLFLOW_MONITOR_EXPERIMENT", "telecom-churn-monitoring"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_COL = "Churn"


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR, MLRUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)
