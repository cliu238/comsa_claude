"""Metrics calculation for model comparison."""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_bootstrap: int = 100,
) -> Dict[str, float]:
    """Calculate comprehensive metrics with confidence intervals.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary containing metrics and confidence intervals
    """
    # Check input lengths
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")
    
    # Basic metrics
    cod_accuracy = accuracy_score(y_true, y_pred)
    csmf_accuracy = calculate_csmf_accuracy(y_true, y_pred)

    # Bootstrap confidence intervals
    cod_ci = bootstrap_metric(y_true, y_pred, accuracy_score, n_bootstrap)

    csmf_ci = bootstrap_metric(y_true, y_pred, calculate_csmf_accuracy, n_bootstrap)

    metrics = {
        "cod_accuracy": cod_accuracy,
        "cod_accuracy_ci_lower": cod_ci[0],
        "cod_accuracy_ci_upper": cod_ci[1],
        "csmf_accuracy": csmf_accuracy,
        "csmf_accuracy_ci_lower": csmf_ci[0],
        "csmf_accuracy_ci_upper": csmf_ci[1],
    }

    # Add per-cause metrics if space allows
    if len(np.unique(y_true)) <= 10:  # Only for small number of causes
        cause_accuracies = calculate_per_cause_accuracy(y_true, y_pred)
        for cause, acc in cause_accuracies.items():
            metrics[f"accuracy_{cause}"] = acc

    return metrics


def calculate_csmf_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Calculate CSMF (Cause-Specific Mortality Fraction) accuracy.

    CSMF accuracy measures how well the predicted distribution of causes
    matches the true distribution.

    Args:
        y_true: True cause labels
        y_pred: Predicted cause labels

    Returns:
        CSMF accuracy score between 0 and 1
    """
    # Handle empty data
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Cannot calculate CSMF accuracy for empty data")
    # Convert to pandas Series for easier manipulation
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, index=y_true.index)

    # Get true and predicted fractions
    true_fractions = y_true.value_counts(normalize=True)
    pred_fractions = y_pred.value_counts(normalize=True)

    # Align categories
    all_categories = sorted(set(true_fractions.index) | set(pred_fractions.index))
    true_fractions = true_fractions.reindex(all_categories, fill_value=0)
    pred_fractions = pred_fractions.reindex(all_categories, fill_value=0)

    # Calculate CSMF accuracy using the standard formula
    diff = np.abs(true_fractions - pred_fractions).sum()
    min_frac = true_fractions.min()

    # Handle edge case where there's only one cause
    if min_frac == 1:
        return 1.0 if diff == 0 else 0.0

    # Standard CSMF accuracy formula
    csmf_accuracy = 1 - diff / (2 * (1 - min_frac))

    return max(0, csmf_accuracy)  # Ensure non-negative


def bootstrap_metric(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metric_func: Callable,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Calculate bootstrap confidence intervals for a metric.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_func: Function to calculate metric
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level for intervals

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    scores = []
    n_samples = len(y_true)

    # Ensure y_pred is array-like with same length
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Set random seed for reproducibility
    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        # Resample indices with replacement
        indices = rng.choice(n_samples, n_samples, replace=True)

        # Calculate metric on resampled data
        y_true_boot = y_true.iloc[indices]
        y_pred_boot = y_pred[indices]

        try:
            score = metric_func(y_true_boot, y_pred_boot)
            scores.append(score)
        except Exception:
            # Skip if metric calculation fails (e.g., missing classes)
            continue

    if not scores:
        return (0.0, 0.0)

    # Calculate confidence intervals
    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def calculate_per_cause_accuracy(
    y_true: pd.Series, y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate accuracy for each cause.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary mapping cause to accuracy
    """
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, index=y_true.index)

    accuracies = {}
    for cause in y_true.unique():
        mask = y_true == cause
        if mask.sum() > 0:
            accuracies[str(cause)] = float((y_true[mask] == y_pred[mask]).mean())
        else:
            accuracies[str(cause)] = 0.0

    return accuracies


def calculate_confusion_matrix_metrics(
    y_true: pd.Series, y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate detailed confusion matrix based metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with precision, recall, f1 per class
    """
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Extract macro and weighted averages
    metrics = {
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }

    return metrics
