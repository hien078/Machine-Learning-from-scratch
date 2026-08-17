"""Data splitting, scaling, and synthetic dataset utilities."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_feature_matrix(X: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if matrix.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    return matrix


def train_test_split(
    X: ArrayLike,
    y: ArrayLike | None,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[NDArray[np.float64], ...]:
    """Split arrays into reproducible train and test subsets.

    Args:
        X: Feature matrix with shape ``(n_samples, n_features)``.
        y: Optional targets with ``n_samples`` entries.
        test_size: Fraction of samples assigned to the test set.
        random_state: Seed used by an isolated random generator.

    Returns:
        ``(X_train, X_test)`` when ``y`` is ``None``; otherwise
        ``(X_train, X_test, y_train, y_test)``.

    Raises:
        ValueError: If shapes or ``test_size`` are invalid.
    """
    features = _as_feature_matrix(X)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")

    n_samples = features.shape[0]
    n_test = int(np.ceil(n_samples * test_size))
    if n_test >= n_samples:
        raise ValueError("test_size leaves no training samples")

    indices = np.random.default_rng(random_state).permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    if y is None:
        return features[train_indices], features[test_indices]

    targets = np.asarray(y)
    if targets.ndim == 0 or targets.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of samples")
    return (
        features[train_indices],
        features[test_indices],
        targets[train_indices],
        targets[test_indices],
    )


def normalize(X: ArrayLike) -> NDArray[np.float64]:
    """Scale every row of a feature matrix to unit Euclidean norm."""
    features = _as_feature_matrix(X)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.where(norms == 0.0, 1.0, norms)


def standardize(
    X: ArrayLike,
    mean: ArrayLike | None = None,
    std: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Center features and scale them to unit population standard deviation.

    Args:
        X: Non-empty one- or two-dimensional array of samples.
        mean: Optional precomputed mean (per feature for 2-D ``X``). Provide
            together with ``std`` to apply training statistics to another
            fold instead of computing them from ``X``.
        std: Optional precomputed standard deviation matching ``mean``.
            Entries equal to zero are treated as one, exactly as in the
            computed case.

    Returns:
        ``(X_std, mean, std)`` where ``std`` has zero entries replaced by one.

    Raises:
        ValueError: If ``X`` is empty or not 1-D/2-D, if only one of ``mean``
            and ``std`` is provided, or if provided statistics are non-finite,
            negative (``std``), or shaped incompatibly with ``X``.
    """
    values = np.asarray(X, dtype=float)
    if values.ndim not in (1, 2) or values.size == 0:
        raise ValueError("X must be a non-empty one- or two-dimensional array")
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be provided together or both omitted")
    if mean is None:
        mean_value = np.mean(values, axis=0)
        std_value = np.std(values, axis=0)
    else:
        mean_value = np.asarray(mean, dtype=float)
        std_value = np.asarray(std, dtype=float)
        expected_shape = values.shape[1:]
        if mean_value.shape != expected_shape or std_value.shape != expected_shape:
            raise ValueError("mean and std must match the feature shape of X")
        if (
            np.any(~np.isfinite(mean_value))
            or np.any(~np.isfinite(std_value))
            or np.any(std_value < 0.0)
        ):
            raise ValueError("mean and std must be finite and std must be non-negative")
    safe_std = np.where(std_value == 0.0, 1.0, std_value)
    return (values - mean_value) / safe_std, np.asarray(mean_value), np.asarray(safe_std)


def generate_regression_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.1,
    random_state: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a linear regression problem with Gaussian features and noise."""
    if n_samples < 1 or n_features < 1 or noise < 0.0:
        raise ValueError("n_samples and n_features must be positive; noise cannot be negative")
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    weights = rng.standard_normal(n_features) * 5.0
    intercept = rng.standard_normal() * 2.0
    y = X @ weights + intercept + rng.standard_normal(n_samples) * noise
    return X, y


def generate_classification_data(
    n_samples: int = 100,
    n_features: int = 2,
    n_classes: int = 2,
    random_state: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate Gaussian clusters while preserving the requested sample count."""
    if n_samples < 1 or n_features < 1 or not 1 <= n_classes <= n_samples:
        raise ValueError("Require positive dimensions and 1 <= n_classes <= n_samples")
    rng = np.random.default_rng(random_state)
    class_sizes = np.full(n_classes, n_samples // n_classes, dtype=int)
    class_sizes[: n_samples % n_classes] += 1
    centers = rng.standard_normal((n_classes, n_features)) * 3.0
    blocks = [
        centers[c] + rng.standard_normal((size, n_features)) for c, size in enumerate(class_sizes)
    ]
    labels = [np.full(size, c, dtype=int) for c, size in enumerate(class_sizes)]
    X = np.vstack(blocks)
    y = np.concatenate(labels)
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def k_fold_split(
    X: ArrayLike,
    y: ArrayLike,
    k: int = 5,
    random_state: int | None = None,
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Yield ``k`` reproducible train/validation index pairs."""
    features = _as_feature_matrix(X)
    targets = np.asarray(y)
    n_samples = features.shape[0]
    if targets.ndim == 0 or targets.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of samples")
    if not 2 <= k <= n_samples:
        raise ValueError("k must satisfy 2 <= k <= n_samples")

    indices = np.random.default_rng(random_state).permutation(n_samples)
    folds = np.array_split(indices, k)
    for index, validation in enumerate(folds):
        training = np.concatenate([fold for i, fold in enumerate(folds) if i != index])
        yield training, validation
