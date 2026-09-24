import pandas as pd
import pytest

from src.data_preprocessing import (
    build_preprocessor,
    clean_data,
    get_feature_columns,
    get_feature_frame,
    load_raw_data,
    validate_customer_input,
)


def test_clean_data_converts_total_charges_and_removes_identifier():
    raw = pd.DataFrame(
        {
            "customerID": ["001", "002"],
            "tenure": [1, 10],
            "MonthlyCharges": [20.0, 55.0],
            "TotalCharges": [" ", "550.0"],
            "Contract": ["Month-to-month", "One year"],
            "Churn": ["Yes", "No"],
        }
    )
    cleaned = clean_data(raw)
    assert "customerID" not in cleaned
    assert cleaned["TotalCharges"].isna().iloc[0]
    assert cleaned["TotalCharges"].iloc[1] == 550.0


def test_preprocessor_is_fitted_only_after_training_data_is_supplied():
    raw = load_raw_data()
    features, _ = get_feature_frame(raw)
    numerical, categorical = get_feature_columns(features)
    transformer = build_preprocessor(numerical, categorical)
    transformed = transformer.fit_transform(features.head(30))
    assert transformed.shape[0] == 30
    assert transformed.shape[1] > len(features.columns)


def test_validate_customer_input_rejects_missing_required_field():
    with pytest.raises(ValueError, match="Missing required customer fields"):
        validate_customer_input(pd.DataFrame([{"tenure": 4}]), ["tenure", "Contract"])

