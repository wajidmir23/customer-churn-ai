"""Dataset loading, validation, cleaning, and preprocessing pipeline helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_raw_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the source Telco dataset and fail with a useful message when absent."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Place the Telco churn CSV in data/."
        )
    return pd.read_csv(dataset_path)


def clean_data(data: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Standardise the source data without fitting anything on it.

    ``TotalCharges`` arrives as text because a small number of new customers have
    blank values. Those blanks become NaN and are imputed inside the training
    pipeline, so the test set never contributes to an imputation statistic.
    """
    frame = data.copy()
    frame.columns = frame.columns.str.strip()
    frame = frame.replace(r"^\s*$", np.nan, regex=True)

    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")

    if ID_COLUMN in frame.columns:
        frame = frame.drop(columns=ID_COLUMN)

    if TARGET_COLUMN in frame.columns and not include_target:
        frame = frame.drop(columns=TARGET_COLUMN)

    # Keep input types stable between model training and Streamlit form submission.
    if "SeniorCitizen" in frame.columns:
        frame["SeniorCitizen"] = pd.to_numeric(frame["SeniorCitizen"], errors="coerce").astype("Int64")

    return frame


def get_feature_columns(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numerical and categorical feature names after cleaning."""
    feature_columns = [
        column for column in data.columns if column not in {TARGET_COLUMN, ID_COLUMN}
    ]
    numerical = [column for column in NUMERICAL_COLUMNS if column in feature_columns]
    categorical = [column for column in feature_columns if column not in numerical]
    return numerical, categorical


def get_feature_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create model inputs and a 0/1 churn target from raw or cleaned data."""
    cleaned = clean_data(data, include_target=True)
    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in training data.")

    target = cleaned.pop(TARGET_COLUMN).map({"Yes": 1, "No": 0})
    if target.isna().any():
        unexpected = data[TARGET_COLUMN].dropna().unique().tolist()
        raise ValueError(f"Churn must contain Yes/No values; found {unexpected}.")
    return cleaned, target.astype(int)


def build_preprocessor(
    numerical_columns: Iterable[str], categorical_columns: Iterable[str]
) -> ColumnTransformer:
    """Build an unfitted transformer for a scikit-learn model pipeline.

    Fitting this object happens only after the train/test split. This is the key
    protection against data leakage in this project.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numerical_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def validate_customer_input(customer: pd.DataFrame, required_columns: Iterable[str]) -> pd.DataFrame:
    """Validate and order a one-or-more-row prediction payload.

    The function deliberately does not reject unseen categorical levels: the
    production encoder is configured with ``handle_unknown='ignore'``.
    """
    expected = list(required_columns)
    missing = [column for column in expected if column not in customer.columns]
    if missing:
        raise ValueError(f"Missing required customer fields: {', '.join(missing)}")
    if customer.empty:
        raise ValueError("Provide at least one customer record for prediction.")

    frame = customer.loc[:, expected].copy()
    for column in NUMERICAL_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "SeniorCitizen" in frame.columns:
        frame["SeniorCitizen"] = pd.to_numeric(frame["SeniorCitizen"], errors="coerce")
    return frame

