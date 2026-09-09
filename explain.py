"""SHAP explanation helpers that work with the fitted sklearn pipeline."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import shap


TREE_MODELS = {"DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier"}


def get_transformed_feature_names(pipeline: Any) -> list[str]:
    """Read one-hot-expanded feature names from a fitted preprocessing pipeline."""
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


def transform_features(pipeline: Any, features: pd.DataFrame) -> np.ndarray:
    """Apply the fitted preprocessing step only (required by SHAP)."""
    return np.asarray(pipeline.named_steps["preprocessor"].transform(features))


def build_shap_explainer(pipeline: Any, background: np.ndarray | None = None) -> Any:
    """Return the appropriate SHAP explainer for the selected sklearn estimator."""
    estimator = pipeline.named_steps["model"]
    if estimator.__class__.__name__ in TREE_MODELS:
        return shap.TreeExplainer(estimator)
    if background is None:
        raise ValueError("A transformed background sample is required for linear SHAP.")
    return shap.LinearExplainer(estimator, background)


def positive_class_values(shap_values: Any) -> np.ndarray:
    """Normalise SHAP's version/model-dependent outputs to positive-class values."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])

    values = np.asarray(shap_values)
    if values.ndim == 3:
        # Modern SHAP returns (rows, features, classes); some versions use classes first.
        if values.shape[-1] == 2:
            return values[:, :, 1]
        if values.shape[1] == 2:
            return values[:, 1, :]
    return values


def explain_rows(
    pipeline: Any, features: pd.DataFrame, background: np.ndarray | None = None
) -> tuple[np.ndarray, list[str]]:
    """Calculate per-feature contributions toward the churn (positive) class."""
    transformed = transform_features(pipeline, features)
    explainer = build_shap_explainer(pipeline, background)
    values = positive_class_values(explainer.shap_values(transformed))
    return np.asarray(values), get_transformed_feature_names(pipeline)


def source_feature_name(transformed_name: str, raw_feature_names: Iterable[str]) -> str:
    """Map ``categorical__Contract_Month-to-month`` back to ``Contract``."""
    short_name = transformed_name.split("__", maxsplit=1)[-1]
    for raw_name in sorted(raw_feature_names, key=len, reverse=True):
        if short_name == raw_name or short_name.startswith(f"{raw_name}_"):
            return raw_name
    return short_name


def aggregate_contributions(
    values: np.ndarray, transformed_names: Iterable[str], raw_feature_names: Iterable[str]
) -> pd.Series:
    """Aggregate one-hot SHAP values so users see original customer fields."""
    contributions = pd.Series(values, index=list(transformed_names), dtype=float)
    grouped = contributions.groupby(
        lambda name: source_feature_name(name, raw_feature_names)
    ).sum()
    return grouped.sort_values()


def global_feature_importance(
    pipeline: Any, sample_features: pd.DataFrame, background: np.ndarray | None = None
) -> pd.DataFrame:
    """Return mean absolute SHAP value per original feature across many customers."""
    values, names = explain_rows(pipeline, sample_features, background)
    raw_features = list(sample_features.columns)
    per_row = [
        aggregate_contributions(row, names, raw_features).abs()
        for row in values
    ]
    importance = pd.DataFrame(per_row).fillna(0).mean().sort_values(ascending=False)
    return importance.rename_axis("feature").reset_index(name="mean_abs_shap")

