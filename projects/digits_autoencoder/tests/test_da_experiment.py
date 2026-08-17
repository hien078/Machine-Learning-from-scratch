"""Fast checks for da_experiment: shapes, training, PCA correctness, determinism."""

from __future__ import annotations

import da_experiment as da
import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

N_TINY = 120


@pytest.fixture(scope="module")
def tiny_data() -> tuple[np.ndarray, np.ndarray]:
    """A small deterministic slice of the digits training set."""
    x_train, _, y_train, _ = da.load_data()
    return x_train[:N_TINY], y_train[:N_TINY]


def test_network_shapes(tiny_data: tuple[np.ndarray, np.ndarray]) -> None:
    features, _ = tiny_data
    classifier = da.build_classifier()
    assert classifier.predict(features).shape == (N_TINY, da.N_CLASSES)
    autoencoder = da.build_autoencoder()
    assert autoencoder.predict(features).shape == (N_TINY, da.N_FEATURES)
    assert da.encode(autoencoder, features).shape == (N_TINY, da.LATENT_DIM)


def test_param_roundtrip() -> None:
    net = da.build_autoencoder()
    flat = da.get_params(net)
    da.set_params(net, flat.copy())
    assert np.array_equal(da.get_params(net), flat)
    with pytest.raises(ValueError):
        da.set_params(net, flat[:-1])


def test_adam_training_reduces_loss(tiny_data: tuple[np.ndarray, np.ndarray]) -> None:
    features, labels = tiny_data
    net = da.build_classifier()
    losses = da.train_adam(net, features, da.one_hot(labels), lr=0.01, max_iter=60)
    assert len(losses) == 60
    assert losses[-1] < 0.5 * losses[0]


def test_pca_matches_sklearn(tiny_data: tuple[np.ndarray, np.ndarray]) -> None:
    features, _ = tiny_data
    mean, components = da.pca_fit(features, 2)
    scores = da.pca_transform(features, mean, components)
    reconstructed = da.pca_reconstruct(scores, mean, components)
    error_scratch = float(np.mean((features - reconstructed) ** 2))

    library = SklearnPCA(n_components=2).fit(features)
    reconstructed_lib = library.inverse_transform(library.transform(features))
    error_library = float(np.mean((features - reconstructed_lib) ** 2))

    assert np.isclose(error_scratch, error_library, atol=1e-10)
    # Analytic check: residual MSE equals the sum of the discarded eigenvalues,
    # rescaled from the (n - 1)-normalized covariance to a per-element mean.
    covariance = np.cov(features, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    n_samples, n_features = features.shape
    analytic = eigenvalues[2:].sum() * (n_samples - 1) / (n_samples * n_features)
    assert np.isclose(error_scratch, analytic, atol=1e-10)


def test_training_is_deterministic(tiny_data: tuple[np.ndarray, np.ndarray]) -> None:
    features, labels = tiny_data
    runs = []
    for _ in range(2):
        net = da.build_classifier()
        losses = da.train_adam(net, features, da.one_hot(labels), lr=0.01, max_iter=30)
        runs.append((losses, da.get_params(net)))
    assert runs[0][0] == runs[1][0]
    assert np.array_equal(runs[0][1], runs[1][1])


def test_one_hot_encoding() -> None:
    labels = np.array([0, 3, 9])
    encoded = da.one_hot(labels)
    assert encoded.shape == (3, da.N_CLASSES)
    assert np.array_equal(encoded.argmax(axis=1), labels)
    assert np.array_equal(encoded.sum(axis=1), np.ones(3))
