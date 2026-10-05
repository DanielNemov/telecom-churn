import pandas as pd
import pytest

from telecom_churn.etl.load import load
from telecom_churn.etl.transform import transform


class TestTransform:
    def test_removes_customer_id(self, raw_df):
        result = transform(raw_df)
        assert "customerID" not in result.columns

    def test_churn_is_binary_int(self, raw_df):
        result = transform(raw_df)
        assert result["Churn"].dtype in (int, "int64", "int32")
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_gender_is_binary_int(self, raw_df):
        result = transform(raw_df)
        assert set(result["gender"].unique()).issubset({0, 1})

    def test_no_null_values(self, raw_df):
        result = transform(raw_df)
        assert result.isnull().sum().sum() == 0

    def test_total_charges_numeric(self, raw_df):
        result = transform(raw_df)
        assert pd.api.types.is_float_dtype(result["TotalCharges"])

    def test_output_shape(self, raw_df):
        result = transform(raw_df)
        assert result.shape[0] == len(raw_df)
        assert result.shape[1] == raw_df.shape[1] - 1

    def test_numeric_cols_scaled(self, raw_df):
        result = transform(raw_df)
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            assert result[col].mean() == pytest.approx(0.0, abs=1e-6)

    def test_binary_yes_no_cols(self, raw_df):
        result = transform(raw_df)
        for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            assert set(result[col].unique()).issubset({0, 1})

    def test_categorical_cols_encoded(self, raw_df):
        result = transform(raw_df)
        for col in ["Contract", "PaymentMethod", "InternetService"]:
            assert pd.api.types.is_integer_dtype(result[col])


class TestLoad:
    def test_load_creates_file(self, processed_df, tmp_path):
        dest = tmp_path / "test_output.csv"
        path = load(processed_df, dest=dest)
        assert path.exists()

    def test_load_file_has_correct_rows(self, processed_df, tmp_path):
        dest = tmp_path / "test_output.csv"
        load(processed_df, dest=dest)
        loaded = pd.read_csv(dest)
        assert len(loaded) == len(processed_df)

    def test_load_file_has_correct_columns(self, processed_df, tmp_path):
        dest = tmp_path / "test_output.csv"
        load(processed_df, dest=dest)
        loaded = pd.read_csv(dest)
        assert list(loaded.columns) == list(processed_df.columns)
