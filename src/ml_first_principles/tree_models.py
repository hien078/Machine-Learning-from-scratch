from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_first_principles.metrics import accuracy


class DecisionTreeClassifier:
    """Binary-split classification tree using weighted Gini impurity."""

    def __init__(
        self,
        max_depth: int | None = None,
        max_features: int | None = None,
        min_samples_split: int = 2,
        random_state: int | None = None,
    ) -> None:
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be positive or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least two")
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.tree_: dict[str, Any] | None = None
        self.classes_: NDArray | None = None
        self._rng = np.random.default_rng(random_state)

    @staticmethod
    def _gini_from_counts(counts: NDArray[np.int64]) -> float:
        total = int(counts.sum())
        if total == 0:
            return 0.0
        probabilities = counts / total
        return float(1.0 - probabilities @ probabilities)

    def _candidate_features(self, n_features: int) -> NDArray[np.int64]:
        count = n_features if self.max_features is None else min(self.max_features, n_features)
        return np.sort(self._rng.choice(n_features, size=count, replace=False))

    def _best_split(
        self, X: NDArray[np.float64], y: NDArray[np.int64]
    ) -> tuple[int | None, float | None]:
        n_samples, n_features = X.shape
        parent_counts = np.bincount(y, minlength=len(self.classes_))
        best_impurity = self._gini_from_counts(parent_counts)
        best_feature: int | None = None
        best_threshold: float | None = None

        for feature in self._candidate_features(n_features):
            order = np.argsort(X[:, feature], kind="mergesort")
            values = X[order, feature]
            labels = y[order]
            left = np.zeros(len(self.classes_), dtype=int)
            right = parent_counts.copy()
            for split in range(1, n_samples):
                label = labels[split - 1]
                left[label] += 1
                right[label] -= 1
                if values[split] == values[split - 1]:
                    continue
                weighted = (
                    split * self._gini_from_counts(left)
                    + (n_samples - split) * self._gini_from_counts(right)
                ) / n_samples
                if weighted < best_impurity - 1e-15:
                    best_impurity = weighted
                    best_feature = int(feature)
                    best_threshold = float((values[split] + values[split - 1]) / 2.0)
        return best_feature, best_threshold

    def _build(self, X: NDArray[np.float64], y: NDArray[np.int64], depth: int) -> dict[str, Any]:
        counts = np.bincount(y, minlength=len(self.classes_))
        node: dict[str, Any] = {"class_index": int(np.argmax(counts))}
        depth_limit = self.max_depth is not None and depth >= self.max_depth
        if depth_limit or len(y) < self.min_samples_split or np.count_nonzero(counts) == 1:
            return node
        feature, threshold = self._best_split(X, y)
        if feature is None or threshold is None:
            return node
        left_mask = X[:, feature] < threshold
        node.update(
            {
                "feature": feature,
                "threshold": threshold,
                "left": self._build(X[left_mask], y[left_mask], depth + 1),
                "right": self._build(X[~left_mask], y[~left_mask], depth + 1),
            }
        )
        return node

    def fit(self, X: ArrayLike, y: ArrayLike) -> DecisionTreeClassifier:
        """Fit a tree and preserve arbitrary original class labels."""
        features = np.asarray(X, dtype=float)
        target = np.asarray(y)
        if features.ndim != 2 or target.ndim != 1 or features.shape[0] != target.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        if features.shape[0] == 0:
            raise ValueError("training data cannot be empty")
        self.classes_, encoded = np.unique(target, return_inverse=True)
        self._rng = np.random.default_rng(self.random_state)
        self.tree_ = self._build(features, encoded, 0)
        return self

    def _predict_index(self, sample: NDArray[np.float64], node: dict[str, Any]) -> int:
        while "threshold" in node:
            branch = "left" if sample[node["feature"]] < node["threshold"] else "right"
            node = node[branch]
        return int(node["class_index"])

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict original class labels."""
        if self.tree_ is None or self.classes_ is None:
            raise RuntimeError("fit must be called before predict")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2:
            raise ValueError("X must be two-dimensional")
        indices = np.array([self._predict_index(sample, self.tree_) for sample in features])
        return self.classes_[indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy."""
        return accuracy(y, self.predict(X))
