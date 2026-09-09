"""Train, compare, explain, and save churn classification pipelines.

Run from the project root with: ``python -m src.train_model``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import (
    DEFAULT_DATA_PATH,
    build_preprocessor,
    get_feature_columns,
    get_feature_frame,
    load_raw_data,
)
from src.evaluate import evaluate_classifier, save_json
from src.explain import global_feature_importance, transform_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def make_candidates() -> dict[str, object]:
    """Define small, interpretable baselines plus two capable tree ensembles."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
        ),
    }


def choose_best_model(results: pd.DataFrame) -> str:
    """Select for churn-catching power, using AUC/F1 as quality tie-breakers.

    Recall and F2 are prioritised over accuracy because missing an imminent
    churner is usually more expensive than offering a retention action to a
    customer who stays.
    """
    ranked = results.sort_values(
        ["recall", "f2", "roc_auc", "f1"], ascending=False
    ).reset_index(drop=True)
    return str(ranked.loc[0, "model"])


def train(data_path: Path = DEFAULT_DATA_PATH, test_size: float = 0.20) -> pd.DataFrame:
    """Run the full reproducible training workflow and persist app artifacts."""
    MODELS_DIR.mkdir(exist_ok=True)
    raw_data = load_raw_data(data_path)
    features, target = get_feature_frame(raw_data)
    numerical_columns, categorical_columns = get_feature_columns(features)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    base_preprocessor = build_preprocessor(numerical_columns, categorical_columns)
    model_results: list[dict[str, float | str | int]] = []
    curves: dict[str, dict[str, list[float]]] = {}
    fitted_models: dict[str, Pipeline] = {}

    # Training weights correct the imbalance for estimators without class_weight.
    positive_weight = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weight = np.where(y_train.to_numpy() == 1, positive_weight, 1.0)

    for name, estimator in make_candidates().items():
        pipeline = Pipeline(
            steps=[("preprocessor", clone(base_preprocessor)), ("model", estimator)]
        )
        fit_parameters = {}
        if name == "Gradient Boosting":
            fit_parameters["model__sample_weight"] = sample_weight
        pipeline.fit(x_train, y_train, **fit_parameters)
        metrics, curve = evaluate_classifier(pipeline, x_test, y_test)
        model_results.append({"model": name, **metrics})
        curves[name] = curve
        fitted_models[name] = pipeline

    comparison = pd.DataFrame(model_results).sort_values("roc_auc", ascending=False)
    comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    best_name = choose_best_model(comparison)
    best_pipeline = fitted_models[best_name]
    joblib.dump(best_pipeline, MODELS_DIR / "best_churn_model.joblib")
    save_json(curves, MODELS_DIR / "roc_curves.json")

    # A bounded sample keeps SHAP artifact generation and reruns responsive.
    shap_sample = x_train.sample(min(600, len(x_train)), random_state=RANDOM_STATE)
    transformed_background = transform_features(
        best_pipeline, x_train.sample(min(200, len(x_train)), random_state=RANDOM_STATE)
    )
    importance = global_feature_importance(
        best_pipeline, shap_sample, background=transformed_background
    )
    importance.to_csv(MODELS_DIR / "global_feature_importance.csv", index=False)

    metadata = {
        "best_model": best_name,
        "selection_rule": "Highest test recall, then F2, ROC-AUC, and F1",
        "decision_threshold": 0.5,
        "data_file": str(Path(data_path).name),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_columns": list(features.columns),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "target_mapping": {"No": 0, "Yes": 1},
        "class_distribution": {
            "no_churn": int((target == 0).sum()),
            "churn": int((target == 1).sum()),
            "churn_rate": float(target.mean()),
        },
    }
    save_json(metadata, MODELS_DIR / "metadata.json")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the customer churn models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="CSV dataset path")
    parser.add_argument("--test-size", type=float, default=0.20, help="Held-out test proportion")
    arguments = parser.parse_args()
    comparison = train(arguments.data, arguments.test_size)
    print("\nModel comparison (held-out test set):")
    print(comparison[["model", "accuracy", "precision", "recall", "f1", "f2", "roc_auc"]].round(3).to_string(index=False))
    with (MODELS_DIR / "metadata.json").open(encoding="utf-8") as file:
        print(f"\nSaved selected model: {json.load(file)['best_model']}")


if __name__ == "__main__":
    main()

