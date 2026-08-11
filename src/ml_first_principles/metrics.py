"""Regression and classification evaluation metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _paired_vectors(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[NDArray, NDArray]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1 or true.shape != pred.shape or true.size == 0:
        raise ValueError(
            "y_true and y_pred must be non-empty one-dimensional arrays of equal shape"
        )
    return true, pred


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean squared error."""
    true, pred = _paired_vectors(y_true, y_pred)
    return float(np.mean((true - pred) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean absolute error."""
    true, pred = _paired_vectors(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the coefficient of determination with finite constant-target behavior."""
    true, pred = _paired_vectors(y_true, y_pred)
    residual = float(np.sum((true - pred) ** 2))
    total = float(np.sum((true - np.mean(true)) ** 2))
    if np.isclose(total, 0.0):
        return 1.0 if np.isclose(residual, 0.0) else 0.0
    return 1.0 - residual / total


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the fraction of exactly matched labels."""
    true, pred = _paired_vectors(y_true, y_pred)
    return float(np.mean(true == pred))


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> NDArray[np.int64]:
    """Return a confusion matrix ordered by the sorted union of observed labels."""
    true, pred = _paired_vectors(y_true, y_pred)
    classes = np.unique(np.concatenate((true, pred)))
    class_to_index = {label: index for index, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, predicted in zip(true, pred, strict=True):
        matrix[class_to_index[actual], class_to_index[predicted]] += 1
    return matrix


def _binary_counts(
    y_true: ArrayLike, y_pred: ArrayLike, positive_label: object
) -> tuple[int, int, int]:
    true, pred = _paired_vectors(y_true, y_pred)
    true_positive = int(np.sum((true == positive_label) & (pred == positive_label)))
    false_positive = int(np.sum((true != positive_label) & (pred == positive_label)))
    false_negative = int(np.sum((true == positive_label) & (pred != positive_label)))
    return true_positive, false_positive, false_negative


def precision(y_true: ArrayLike, y_pred: ArrayLike, positive_label: object = 1) -> float:
    """Return binary precision, using zero when no positive prediction exists."""
    true_positive, false_positive, _ = _binary_counts(y_true, y_pred, positive_label)
    denominator = true_positive + false_positive
    return 0.0 if denominator == 0 else true_positive / denominator


def recall(y_true: ArrayLike, y_pred: ArrayLike, positive_label: object = 1) -> float:
    """Return binary recall, using zero when no positive target exists."""
    true_positive, _, false_negative = _binary_counts(y_true, y_pred, positive_label)
    denominator = true_positive + false_negative
    return 0.0 if denominator == 0 else true_positive / denominator


def f1_score(y_true: ArrayLike, y_pred: ArrayLike, positive_label: object = 1) -> float:
    """Return the harmonic mean of binary precision and recall."""
    precision_value = precision(y_true, y_pred, positive_label)
    recall_value = recall(y_true, y_pred, positive_label)
    denominator = precision_value + recall_value
    return 0.0 if denominator == 0.0 else 2.0 * precision_value * recall_value / denominator


def cross_entropy_loss(y_true: ArrayLike, y_pred_proba: ArrayLike, eps: float = 1e-15) -> float:
    """Return categorical cross-entropy for one-hot targets."""
    true = np.asarray(y_true, dtype=float)
    probability = np.asarray(y_pred_proba, dtype=float)
    if true.ndim != 2 or probability.shape != true.shape or true.shape[0] == 0:
        raise ValueError(
            "targets and probabilities must have the same non-empty two-dimensional shape"
        )
    if np.any(probability < 0.0) or np.any(~np.isfinite(probability)):
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = probability.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("each probability row must have positive mass")
    normalized = np.clip(probability / row_sums, eps, 1.0)
    return float(-np.mean(np.sum(true * np.log(normalized), axis=1)))


def log_loss(y_true: ArrayLike, y_pred_proba: ArrayLike, eps: float = 1e-15) -> float:
    """Return binary cross-entropy for targets encoded as zero and one."""
    true, probability = _paired_vectors(y_true, y_pred_proba)
    if np.any((true != 0) & (true != 1)) or np.any(~np.isfinite(probability)):
        raise ValueError("binary log loss requires finite probabilities and targets in {0, 1}")
    clipped = np.clip(probability.astype(float), eps, 1.0 - eps)
    return float(-np.mean(true * np.log(clipped) + (1.0 - true) * np.log(1.0 - clipped)))
