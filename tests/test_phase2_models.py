import numpy as np

from ml_first_principles.data_utils import generate_classification_data, standardize
from ml_first_principles.distance_models import KMeans, KNeighborsClassifier
from ml_first_principles.ensemble_models import RandomForestClassifier
from ml_first_principles.probabilistic_models import GaussianNB
from ml_first_principles.svm_models import LinearSVC
from ml_first_principles.tree_models import DecisionTreeClassifier


def test_decision_tree():
    X, y = generate_classification_data(n_samples=100, n_features=2, n_classes=2, random_state=42)
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    assert model.score(X, y) > 0.8


def test_random_forest():
    X, y = generate_classification_data(n_samples=100, n_features=2, n_classes=2, random_state=42)
    model = RandomForestClassifier(n_estimators=5, max_depth=3, max_features=1, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.8
    assert all(tree.max_features == 1 for tree in model.trees_)


def test_knn():
    X, y = generate_classification_data(n_samples=100, n_features=2, n_classes=2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    assert model.score(X, y) > 0.8


def test_kmeans():
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(size=(50, 2)) + [2, 2], rng.normal(size=(50, 2)) + [-2, -2]])
    model = KMeans(n_clusters=2, random_state=0)
    model.fit(X)
    assert model.cluster_centers_.shape == (2, 2)
    centers = model.cluster_centers_[np.argsort(model.cluster_centers_[:, 0])]
    np.testing.assert_allclose(centers, [[-2.0, -2.0], [2.0, 2.0]], atol=0.5)
    assert model.labels_.shape == (100,)
    assert model.inertia_ > 0.0


def test_naive_bayes():
    X, y = generate_classification_data(n_samples=100, n_features=2, n_classes=2, random_state=42)
    model = GaussianNB()
    model.fit(X, y)
    assert model.score(X, y) > 0.8


def test_svm():
    X, y = generate_classification_data(n_samples=100, n_features=2, n_classes=2, random_state=42)
    X, _, _ = standardize(X)
    model = LinearSVC(C=1.0, lr=0.1, max_iter=200)
    model.fit(X, y)
    assert model.score(X, y) > 0.7


def test_models_preserve_nonzero_and_negative_labels():
    X = np.array([[0.0], [0.1], [0.9], [1.0]])

    tree_labels = np.array([2, 2, 4, 4])
    tree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, tree_labels)
    np.testing.assert_array_equal(tree.predict(X), tree_labels)

    knn_labels = np.array([-1, -1, 1, 1])
    knn = KNeighborsClassifier(n_neighbors=1).fit(X, knn_labels)
    np.testing.assert_array_equal(knn.predict(X), knn_labels)

    svm = LinearSVC(C=10.0, lr=0.05, max_iter=500, random_state=42).fit(X, knn_labels)
    np.testing.assert_array_equal(svm.predict(X), knn_labels)


def test_kmeans_recovers_from_empty_clusters():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    model = KMeans(n_clusters=3, max_iter=20, random_state=1).fit(X)
    assert np.all(np.isfinite(model.cluster_centers_))
    assert model.labels_.shape == (4,)


def test_gaussian_nb_remains_stable_for_extreme_values():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    y = np.array([0, 0, 1, 1])
    model = GaussianNB().fit(X, y)
    assert model.predict(np.array([[1_000_000.0]]))[0] == 1
