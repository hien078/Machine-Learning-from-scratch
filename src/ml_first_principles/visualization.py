from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray


class Predictor(Protocol):
    """Structural type for objects exposing ``predict``."""

    def predict(self, X: NDArray[np.float64]) -> NDArray: ...


def plot_regression(
    X: ArrayLike, y: ArrayLike, y_pred: ArrayLike, title: str = "Regression"
) -> Figure:
    """Plot one-dimensional observations and predictions."""
    features = np.asarray(X, dtype=float).reshape(-1)
    target = np.asarray(y, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if not features.shape == target.shape == prediction.shape:
        raise ValueError("X, y, and y_pred must contain the same number of values")
    order = np.argsort(features)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(features, target, color="tab:blue", alpha=0.5, label="Data")
    axis.plot(features[order], prediction[order], color="tab:red", linewidth=2, label="Prediction")
    axis.set(title=title, xlabel="Feature", ylabel="Target")
    axis.legend()
    axis.grid(True, alpha=0.3)
    return figure


def plot_decision_boundary(
    model: Predictor,
    X: ArrayLike,
    y: ArrayLike,
    title: str = "Decision Boundary",
    mesh_step: float = 0.05,
) -> Figure:
    """Plot the predicted regions of a two-feature classifier."""
    features = np.asarray(X, dtype=float)
    target = np.asarray(y)
    if features.ndim != 2 or features.shape[1] != 2 or target.shape[0] != features.shape[0]:
        raise ValueError("decision-boundary plots require X with two features and matching y")
    if mesh_step <= 0.0:
        raise ValueError("mesh_step must be positive")
    x_min, x_max = features[:, 0].min() - 1.0, features[:, 0].max() + 1.0
    y_min, y_max = features[:, 1].min() - 1.0, features[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step),
        np.arange(y_min, y_max, mesh_step),
    )
    mesh = np.column_stack((xx.ravel(), yy.ravel()))
    labels = np.unique(np.concatenate((target, np.asarray(model.predict(mesh)))))
    label_to_index = {label: index for index, label in enumerate(labels)}
    region = np.array([label_to_index[label] for label in model.predict(mesh)]).reshape(xx.shape)
    colors = np.array([label_to_index[label] for label in target])
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.contourf(xx, yy, region, alpha=0.3, cmap="coolwarm")
    axis.scatter(features[:, 0], features[:, 1], c=colors, edgecolors="k", cmap="coolwarm")
    axis.set(title=title, xlabel="Feature 1", ylabel="Feature 2")
    axis.grid(True, alpha=0.3)
    return figure


def plot_learning_curve(losses: Sequence[float], title: str = "Learning Curve") -> Figure:
    """Plot scalar loss against iteration or epoch."""
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("losses must be a non-empty finite one-dimensional sequence")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(np.arange(len(values)), values, color="tab:blue")
    axis.set(title=title, xlabel="Iteration", ylabel="Loss")
    axis.grid(True, alpha=0.3)
    return figure


def plot_confusion_matrix(
    matrix: ArrayLike, labels: Sequence[object] | None = None, title: str = "Confusion Matrix"
) -> Figure:
    """Plot a square confusion matrix with annotated counts."""
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    tick_labels = np.arange(values.shape[0]) if labels is None else np.asarray(labels)
    if len(tick_labels) != values.shape[0]:
        raise ValueError("labels must match the matrix dimensions")
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values, interpolation="nearest", cmap=plt.cm.Blues)
    figure.colorbar(image, ax=axis)
    threshold = float(values.max(initial=0)) / 2.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{int(values[row, column])}",
                ha="center",
                color="white" if values[row, column] > threshold else "black",
            )
    axis.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(values.shape[0]),
        yticks=np.arange(values.shape[0]),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    figure.tight_layout()
    return figure


def plot_regularization_path(
    alphas: ArrayLike, coefficients: ArrayLike, title: str = "Regularization Path"
) -> Figure:
    """Plot coefficients over positive regularization strengths."""
    strengths = np.asarray(alphas, dtype=float)
    values = np.asarray(coefficients, dtype=float)
    if strengths.ndim != 1 or values.ndim != 2 or values.shape[0] != strengths.size:
        raise ValueError("coefficients must have one row per alpha")
    if np.any(strengths <= 0.0):
        raise ValueError("alphas must be positive for logarithmic scaling")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(strengths, values)
    axis.set_xscale("log")
    axis.set(title=title, xlabel="Alpha", ylabel="Coefficient")
    axis.grid(True, alpha=0.3)
    return figure
