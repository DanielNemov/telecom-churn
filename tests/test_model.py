import pickle

import numpy as np
import pandas as pd


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

    def test_model_serialization(self, trained_sklearn_model, sample_features):
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
            assert pd.api.types.is_numeric_dtype(
                sample_features[col]
            ), f"Column {col} is not numeric"
