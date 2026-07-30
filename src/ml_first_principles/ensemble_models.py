from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_first_principles.metrics import accuracy
from ml_first_principles.tree_models import DecisionTreeClassifier


class RandomForestClassifier:
    """Random forest using bootstrap samples and per-split feature subsets."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: int | float | str | None = "sqrt",
        random_state: int | None = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: list[DecisionTreeClassifier] = []
        self.classes_: NDArray | None = None

    def _feature_count(self, n_features: int) -> int:
        value = self.max_features
        if value is None:
            return n_features
        if value == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if value == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(value, int) and 1 <= value <= n_features:
            return value
        if isinstance(value, float) and 0.0 < value <= 1.0:
            return max(1, int(np.ceil(value * n_features)))
        raise ValueError("max_features must be None, sqrt, log2, a valid count, or a fraction")

    def fit(self, X: ArrayLike, y: ArrayLike) -> RandomForestClassifier:
        """Fit independently seeded trees to bootstrap samples."""
        features = np.asarray(X, dtype=float)
        target = np.asarray(y)
        if features.ndim != 2 or target.ndim != 1 or features.shape[0] != target.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        if features.shape[0] == 0:
            raise ValueError("training data cannot be empty")
        self.classes_ = np.unique(target)
        rng = np.random.default_rng(self.random_state)
        feature_count = self._feature_count(features.shape[1])
        self.trees_ = []
        for _ in range(self.n_estimators):
            bootstrap = rng.integers(0, features.shape[0], size=features.shape[0])
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=feature_count,
                random_state=seed,
            )
            tree.fit(features[bootstrap], target[bootstrap])
            self.trees_.append(tree)
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict labels by majority vote with deterministic tie breaking."""
        if not self.trees_ or self.classes_ is None:
            raise RuntimeError("fit must be called before predict")
        predictions = np.asarray([tree.predict(X) for tree in self.trees_])
        output = []
        for column in predictions.T:
            counts = np.array([np.sum(column == label) for label in self.classes_])
            output.append(self.classes_[np.argmax(counts)])
        return np.asarray(output)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy."""
        return accuracy(y, self.predict(X))
