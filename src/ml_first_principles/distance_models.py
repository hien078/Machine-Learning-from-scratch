"""Distance-based models: k-nearest neighbors and k-means clustering."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_first_principles.metrics import accuracy


class KNeighborsClassifier:
    """Brute-force $k$-nearest-neighbors classifier with Euclidean distance."""

    def __init__(self, n_neighbors: int = 5) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        self.n_neighbors = n_neighbors
        self.X_train_: NDArray[np.float64] | None = None
        self.y_train_: NDArray | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> KNeighborsClassifier:
        """Store the training set."""
        features = np.asarray(X, dtype=float)
        target = np.asarray(y)
        if features.ndim != 2 or target.ndim != 1 or features.shape[0] != target.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        if not self.n_neighbors <= features.shape[0]:
            raise ValueError("n_neighbors cannot exceed the training sample count")
        self.X_train_ = features.copy()
        self.y_train_ = target.copy()
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict by an unweighted majority vote."""
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("fit must be called before predict")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.X_train_.shape[1]:
            raise ValueError("X has an incompatible feature shape")
        predictions = []
        for sample in features:
            distances = np.linalg.norm(self.X_train_ - sample, axis=1)
            neighbors = np.argpartition(distances, self.n_neighbors - 1)[: self.n_neighbors]
            labels, counts = np.unique(self.y_train_[neighbors], return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.asarray(predictions)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy."""
        return accuracy(y, self.predict(X))


class KMeans:
    """Lloyd's $k$-means algorithm with deterministic empty-cluster recovery."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        random_state: int | None = 42,
        tol: float = 1e-4,
    ) -> None:
        if n_clusters < 1 or max_iter < 1 or tol <= 0.0:
            raise ValueError("n_clusters, max_iter, and tol must be positive")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.cluster_centers_: NDArray[np.float64] | None = None
        self.labels_: NDArray[np.int64] | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0

    def fit(self, X: ArrayLike) -> KMeans:
        """Cluster observations by alternating assignment and mean updates."""
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if self.n_clusters > features.shape[0]:
            raise ValueError("n_clusters cannot exceed the sample count")
        rng = np.random.default_rng(self.random_state)
        initial = rng.choice(features.shape[0], self.n_clusters, replace=False)
        centers = features[initial].copy()

        for iteration in range(1, self.max_iter + 1):
            squared_distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(squared_distances, axis=1)
            new_centers = centers.copy()
            empty_clusters = []
            for cluster in range(self.n_clusters):
                members = features[labels == cluster]
                if members.size:
                    new_centers[cluster] = members.mean(axis=0)
                else:
                    empty_clusters.append(cluster)
            if empty_clusters:
                nearest_squared = squared_distances[np.arange(features.shape[0]), labels]
                candidates = np.argsort(nearest_squared)[::-1]
                used: set[int] = set()
                for cluster in empty_clusters:
                    sample_index = next(index for index in candidates if int(index) not in used)
                    used.add(int(sample_index))
                    new_centers[cluster] = features[sample_index]
            shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
            centers = new_centers
            self.n_iter_ = iteration
            if shift <= self.tol:
                break

        final_squared = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        self.cluster_centers_ = centers
        self.labels_ = np.argmin(final_squared, axis=1)
        self.inertia_ = float(np.sum(final_squared[np.arange(features.shape[0]), self.labels_]))
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Assign observations to the nearest fitted center."""
        if self.cluster_centers_ is None:
            raise RuntimeError("fit must be called before predict")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X has an incompatible feature shape")
        squared = np.sum((features[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=2)
        return np.argmin(squared, axis=1)

    def fit_predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Fit the model and return training assignments."""
        self.fit(X)
        if self.labels_ is None:
            raise RuntimeError("fit did not produce labels")
        return self.labels_
