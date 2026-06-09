import os
import sys
import pickle
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "etl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "model"))


@pytest.fixture
def sample_features(processed_df) -> pd.DataFrame:
    return processed_df.drop(columns=["Churn"])


@pytest.fixture
def sample_labels(processed_df) -> pd.Series:
    return processed_df["Churn"]


@pytest.fixture
def trained_sklearn_model(processed_df, tmp_path):
    X = processed_df.drop(columns=["Churn"])
    y = processed_df["Churn"]
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    model_path = tmp_path / "test_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model, str(model_path)


class TestModelInterface:
    def test_sklearn_model_predict_shape(self, trained_sklearn_model, sample_features):
        model, _ = trained_sklearn_model
        preds = model.predict(sample_features)
        assert len(preds) == len(sample_features)

    def test_sklearn_model_predict_binary(self, trained_sklearn_model, sample_features):
        model, _ = trained_sklearn_model
        preds = model.predict(sample_features)
        assert set(preds).issubset({0, 1})

    def test_sklearn_model_predict_proba(self, trained_sklearn_model, sample_features):
        model, _ = trained_sklearn_model
        proba = model.predict_proba(sample_features)
        assert proba.shape == (len(sample_features), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_model_serialization(self, trained_sklearn_model, sample_features, tmp_path):
        model, model_path = trained_sklearn_model
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)
        original_preds = model.predict(sample_features)
        loaded_preds = loaded.predict(sample_features)
        assert list(original_preds) == list(loaded_preds)

    def test_model_feature_count(self, trained_sklearn_model, sample_features):
        model, _ = trained_sklearn_model
        assert model.n_features_in_ == sample_features.shape[1]


class TestDataIntegrity:
    def test_processed_df_has_churn_column(self, processed_df):
        assert "Churn" in processed_df.columns

    def test_processed_df_no_nan(self, processed_df):
        assert processed_df.isnull().sum().sum() == 0

    def test_features_all_numeric(self, sample_features):
        for col in sample_features.columns:
            assert pd.api.types.is_numeric_dtype(sample_features[col]), (
                f"Column {col} is not numeric"
            )
