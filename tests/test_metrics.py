"""
Unit tests for metrics.
"""

import numpy as np

from ml_first_principles.metrics import (
    accuracy,
    confusion_matrix,
    cross_entropy_loss,
    f1_score,
    log_loss,
    mae,
    mse,
    precision,
    r2_score,
    recall,
    rmse,
)


def test_regression_metrics():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    assert np.isclose(mse(y_true, y_pred), 0.375)
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(0.375))
    assert np.isclose(mae(y_true, y_pred), 0.5)

    # R2
    # mean(y) = 2.875
    # ss_tot = 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875
    # ss_res = 1.5
    # r2 = 1 - 1.5 / 29.1875 = 0.9486...
    assert np.isclose(r2_score(y_true, y_pred), 0.9486081370449678)


def test_classification_metrics():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])

    assert accuracy(y_true, y_pred) == 4 / 6

    # cm:
    #       pred_0  pred_1
    # true_0   2       1
    # true_1   1       2
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 2
    assert cm[1, 1] == 2
    assert cm[0, 1] == 1
    assert cm[1, 0] == 1

    # precision = TP / (TP + FP) = 2 / (2 + 1) = 2/3
    assert np.isclose(precision(y_true, y_pred), 2 / 3)

    # recall = TP / (TP + FN) = 2 / (2 + 1) = 2/3
    assert np.isclose(recall(y_true, y_pred), 2 / 3)

    # f1 = 2 * (2/3 * 2/3) / (4/3) = 2/3
    assert np.isclose(f1_score(y_true, y_pred), 2 / 3)


def test_loss_functions():
    y_true_bin = np.array([1, 0, 1])
    y_pred_prob = np.array([0.9, 0.1, 0.8])

    loss = log_loss(y_true_bin, y_pred_prob)
    # - (1*log(0.9) + 1*log(0.9) + 1*log(0.8)) / 3 = - (log(0.9) + log(0.9) + log(0.8)) / 3
    # 0.9*0.1 gives 1*log(0.9) for first, 1*log(0.9) for second
    expected = -(np.log(0.9) + np.log(0.9) + np.log(0.8)) / 3
    assert np.isclose(loss, expected)

    y_true_cat = np.array([[1, 0], [0, 1]])
    y_pred_cat = np.array([[0.9, 0.1], [0.2, 0.8]])
    ce_loss = cross_entropy_loss(y_true_cat, y_pred_cat)
    expected_ce = -(np.log(0.9) + np.log(0.8)) / 2
    assert np.isclose(ce_loss, expected_ce)


def test_binary_metrics_handle_absent_positive_label():
    y_true = np.array([2, 3])
    y_pred = np.array([2, 3])
    assert precision(y_true, y_pred, positive_label=1) == 0.0
    assert recall(y_true, y_pred, positive_label=1) == 0.0
    assert f1_score(y_true, y_pred, positive_label=1) == 0.0


def test_metrics_reject_incompatible_shapes():
    import pytest

    with pytest.raises(ValueError, match="equal shape"):
        mse(np.array([1.0]), np.array([1.0, 2.0]))


def test_regression_metrics_accept_matching_nd_arrays():
    rng = np.random.default_rng(1)
    y_true = rng.standard_normal((4, 5))
    y_pred = rng.standard_normal((4, 5))

    assert np.isclose(mse(y_true, y_pred), mse(y_true.ravel(), y_pred.ravel()))
    assert np.isclose(rmse(y_true, y_pred), rmse(y_true.ravel(), y_pred.ravel()))
    assert np.isclose(mae(y_true, y_pred), mae(y_true.ravel(), y_pred.ravel()))
    assert np.isclose(r2_score(y_true, y_pred), r2_score(y_true.ravel(), y_pred.ravel()))


def test_regression_metrics_reject_mismatched_nd_shapes():
    import pytest

    with pytest.raises(ValueError, match="equal shape"):
        mse(np.zeros((2, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="equal shape"):
        mae(np.zeros((2, 3)), np.zeros(6))
    with pytest.raises(ValueError, match="equal shape"):
        r2_score(np.zeros(0), np.zeros(0))
