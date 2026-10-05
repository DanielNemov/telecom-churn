from __future__ import annotations

import logging
import pickle
import time

import mlflow
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from telecom_churn.config import (
    CV_FOLDS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_FILE,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    ensure_dirs,
)
from telecom_churn.etl.load import load_processed
from telecom_churn.etl.pipeline import run_etl
from telecom_churn.logging_config import setup_logging

logger = logging.getLogger(__name__)

CANDIDATES = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME", random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    "XGBoost": XGBClassifier(
        random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0
    ),
}

PARAM_GRIDS = {
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
    "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    "GradientBoosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
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
    logger.info("Comparing models (%s-fold CV by AUC)...", CV_FOLDS)
    results = {}
    for name, model in CANDIDATES.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc", n_jobs=1
        )
        results[name] = scores.mean()
        logger.info("  %s: AUC=%.4f (+/- %.4f)", name, scores.mean(), scores.std())
    best_name = max(results, key=results.get)
    logger.info("Best model: %s (AUC=%.4f)", best_name, results[best_name])
    return best_name, CANDIDATES[best_name], results[best_name]


def tune_model(name, model, X_train, y_train):
    logger.info("Tuning %s with RandomizedSearchCV...", name)
    param_grid = PARAM_GRIDS.get(name, {})
    if not param_grid:
        model.fit(X_train, y_train)
        return model
    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=min(10, sum(len(v) for v in param_grid.values())),
        cv=CV_FOLDS,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info("Best params: %s", search.best_params_)
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


def train() -> dict:
    setup_logging()
    ensure_dirs()
    if not PROCESSED_FILE.exists():
        run_etl()
    df = load_processed()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="automl-sklearn"):
        start = time.time()
        best_name, best_model, _ = compare_models(X_train, y_train)
        tuned = tune_model(best_name, best_model, X_train, y_train)
        metrics = evaluate(tuned, X_test, y_test)
        elapsed = round(time.time() - start, 1)

        mlflow.log_param("best_model", best_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("training_time_sec", elapsed)
        mlflow.log_metrics(metrics)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": tuned, "model_name": best_name}, f)
        mlflow.log_artifact(str(MODEL_PATH))

        logger.info("Training complete in %ss. Metrics on test set:", elapsed)
        for key, value in metrics.items():
            logger.info("  %s: %.4f", key, value)

    return metrics


if __name__ == "__main__":
    train()
