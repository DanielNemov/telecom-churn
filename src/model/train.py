import os
import sys
import pickle
import time
import mlflow
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "etl"))

from extract import extract  # noqa: E402
from transform import transform  # noqa: E402
from load import load, load_processed, PROCESSED_FILE  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    os.path.join(os.path.dirname(__file__), "..", "..", "mlruns"),
)

CANDIDATES = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME", random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "XGBoost": XGBClassifier(
        random_state=42, eval_metric="logloss", verbosity=0
    ),
}

PARAM_GRIDS = {
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
    },
    "ExtraTrees": {"n_estimators": [100, 200], "max_depth": [None, 10]},
    "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
    "DecisionTree": {"max_depth": [5, 10, 20, None]},
    "KNN": {"n_neighbors": [3, 5, 7, 11]},
    "LightGBM": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 6],
    },
}


def compare_models(X_train, y_train) -> tuple:
    print("Comparing models (5-fold CV by AUC)...")
    results = {}
    for name, model in CANDIDATES.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=1
        )
        results[name] = scores.mean()
        print(f"  {name}: AUC={scores.mean():.4f} (+/- {scores.std():.4f})")
    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name} (AUC={results[best_name]:.4f})")
    return best_name, CANDIDATES[best_name], results[best_name]


def tune_model(name, model, X_train, y_train):
    print(f"Tuning {name} with RandomizedSearchCV...")
    param_grid = PARAM_GRIDS.get(name, {})
    if not param_grid:
        model.fit(X_train, y_train)
        return model
    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=10,
        cv=5,
        scoring="roc_auc",
        random_state=42,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
    }


def run_etl_if_needed() -> None:
    if not os.path.exists(PROCESSED_FILE):
        raw = extract()
        processed = transform(raw)
        load(processed)


def train() -> dict:
    run_etl_if_needed()
    df = load_processed()

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("telecom-churn")

    with mlflow.start_run(run_name="automl-sklearn"):
        start = time.time()

        best_name, best_model, _ = compare_models(X_train, y_train)
        tuned = tune_model(best_name, best_model, X_train, y_train)
        metrics = evaluate(tuned, X_test, y_test)
        elapsed = round(time.time() - start, 1)

        mlflow.log_param("best_model", best_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("training_time_sec", elapsed)
        mlflow.log_metrics(metrics)

        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": tuned, "model_name": best_name}, f)
        mlflow.log_artifact(MODEL_PATH)

        print(f"\nTraining complete in {elapsed}s. Metrics on test set:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    train()
