import matplotlib.pyplot as plt
import numpy as np

from ml_first_principles.visualization import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_learning_curve,
    plot_regression,
    plot_regularization_path,
)


class _ThresholdClassifier:
    def predict(self, X):
        return (X[:, 0] + X[:, 1] > 0).astype(int)


def test_regression_plot_has_labels_and_two_artists():
    figure = plot_regression(
        np.array([[0.0], [1.0], [2.0]]),
        np.array([1.0, 3.0, 5.0]),
        np.array([1.0, 3.0, 5.0]),
    )
    axis = figure.axes[0]
    assert axis.get_xlabel() == "Feature"
    assert axis.get_ylabel() == "Target"
    assert len(axis.lines) == 1
    assert len(axis.collections) == 1
    plt.close(figure)


def test_learning_curve_plots_supplied_loss():
    losses = [2.0, 1.0, 0.5]
    figure = plot_learning_curve(losses)
    np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), losses)
    assert figure.axes[0].get_ylabel() == "Loss"
    plt.close(figure)


def test_confusion_matrix_contains_count_annotations():
    figure = plot_confusion_matrix(np.array([[3, 1], [2, 4]]), labels=["no", "yes"])
    assert {text.get_text() for text in figure.axes[0].texts} == {"1", "2", "3", "4"}
    plt.close(figure)


def test_decision_boundary_draws_regions_and_points():
    X = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    y = np.array([0, 1, 0, 1])
    figure = plot_decision_boundary(_ThresholdClassifier(), X, y, mesh_step=0.5)
    axis = figure.axes[0]
    assert axis.get_xlabel() == "Feature 1"
    assert axis.get_ylabel() == "Feature 2"
    assert len(axis.collections) >= 1
    plt.close(figure)


def test_regularization_path_uses_log_scale():
    alphas = np.array([0.01, 0.1, 1.0, 10.0])
    coefficients = np.array([[1.0, 2.0], [0.5, 1.0], [0.1, 0.2], [0.0, 0.0]])
    figure = plot_regularization_path(alphas, coefficients)
    axis = figure.axes[0]
    assert axis.get_xscale() == "log"
    assert len(axis.lines) == 2
    plt.close(figure)
