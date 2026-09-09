"""Model evaluation and plotting helpers used during training and in the app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classifier(model: Any, features: pd.DataFrame, target: pd.Series) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Calculate business-relevant binary-classification metrics and ROC points."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    fpr, tpr, thresholds = roc_curve(target, probabilities)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        # F2 gives recall extra importance because missed churners are costly.
        "f2": float(fbeta_score(target, predictions, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    curve = {
        "fpr": np.round(fpr, 8).tolist(),
        "tpr": np.round(tpr, 8).tolist(),
        "thresholds": np.round(thresholds, 8).tolist(),
    }
    return metrics, curve


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Persist JSON with stable formatting for app consumption and code review."""
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

