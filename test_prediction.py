import json
from pathlib import Path

import joblib

from src.data_preprocessing import clean_data, load_raw_data, validate_customer_input


ROOT = Path(__file__).resolve().parents[1]


def test_saved_model_returns_binary_predictions_and_probability():
    model = joblib.load(ROOT / "models" / "best_churn_model.joblib")
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text())
    customer = clean_data(load_raw_data()).drop(columns="Churn").head(1)
    payload = validate_customer_input(customer, metadata["feature_columns"])
    probability = model.predict_proba(payload)[0, 1]
    prediction = model.predict(payload)[0]
    assert 0.0 <= probability <= 1.0
    assert prediction in {0, 1}

