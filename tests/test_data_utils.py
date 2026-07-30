import numpy as np
import pytest

from ml_first_principles.data_utils import (
    generate_classification_data,
    k_fold_split,
    standardize,
    train_test_split,
)


def test_classification_generator_preserves_requested_sample_count():
    X, y = generate_classification_data(
        n_samples=101,
        n_features=3,
        n_classes=4,
        random_state=42,
    )
    assert X.shape == (101, 3)
    assert y.shape == (101,)
    assert sorted(np.bincount(y)) == [25, 25, 25, 26]


def test_standardize_supports_one_dimensional_input():
    scaled, mean, std = standardize(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(mean, 2.0)
    assert np.isclose(std, np.std([1.0, 2.0, 3.0]))
    assert np.isclose(np.mean(scaled), 0.0, atol=1e-12)
    assert np.isclose(np.std(scaled), 1.0, atol=1e-12)


def test_split_is_reproducible_and_keeps_all_samples():
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10)
    first = train_test_split(X, y, test_size=0.3, random_state=7)
    second = train_test_split(X, y, test_size=0.3, random_state=7)
    for first_array, second_array in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_array, second_array)
    assert len(first[0]) == 7
    assert len(first[1]) == 3


def test_k_folds_partition_validation_indices_once():
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10)
    folds = list(k_fold_split(X, y, k=3, random_state=42))
    validation = np.concatenate([fold[1] for fold in folds])
    np.testing.assert_array_equal(np.sort(validation), np.arange(10))
    assert all(set(train).isdisjoint(valid) for train, valid in folds)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.1])
def test_split_rejects_invalid_test_size(test_size):
    with pytest.raises(ValueError, match="test_size"):
        train_test_split(np.ones((5, 2)), np.ones(5), test_size=test_size)
