import numpy as np

from ml_first_principles.nn_core import Dense, ReLU, Sequential, Sigmoid


def test_xor_neural_network():
    # XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    net = Sequential()
    net.add(Dense(2, 4, random_state=42))
    net.add(Sigmoid())
    net.add(Dense(4, 1, random_state=43))
    net.add(Sigmoid())

    net.fit(X, y, epochs=1000, learning_rate=0.5)

    predictions = net.predict(X)

    assert predictions[0] < 0.5
    assert predictions[1] > 0.5
    assert predictions[2] > 0.5
    assert predictions[3] < 0.5


def test_dense_backward_matches_finite_difference():
    layer = Dense(2, 1, random_state=7)
    X = np.array([[0.2, -0.4], [1.0, 0.5]])
    upstream = np.array([[0.3], [-0.2]])
    original = layer.weights.copy()
    layer.forward(X)
    layer.backward(upstream, learning_rate=0.0)

    numerical = np.zeros_like(original)
    epsilon = 1e-6
    for index in np.ndindex(original.shape):
        layer.weights[index] = original[index] + epsilon
        plus = np.sum(layer.forward(X) * upstream)
        layer.weights[index] = original[index] - epsilon
        minus = np.sum(layer.forward(X) * upstream)
        numerical[index] = (plus - minus) / (2 * epsilon)
        layer.weights[index] = original[index]

    np.testing.assert_allclose(layer.weights_gradient_, numerical, atol=1e-7)


def test_relu_forward_and_backward():
    layer = ReLU()
    x = np.array([[-1.0, 0.0, 2.0]])
    np.testing.assert_allclose(layer.forward(x), [[0.0, 0.0, 2.0]])
    upstream = np.array([[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(layer.backward(upstream, learning_rate=0.0), [[0.0, 0.0, 1.0]])


def test_dense_backward_without_learning_rate_collects_gradients():
    X = np.array([[0.2, -0.4], [1.0, 0.5]])
    upstream = np.array([[0.3, -0.1], [-0.2, 0.4]])

    reference = Dense(2, 2, random_state=7)
    reference.forward(X)
    reference.backward(upstream, learning_rate=0.1)

    layer = Dense(2, 2, random_state=7)
    weights_before = layer.weights.copy()
    bias_before = layer.bias.copy()
    layer.forward(X)
    input_error = layer.backward(upstream)

    # Parameters untouched, gradients identical to the update-mode computation.
    np.testing.assert_array_equal(layer.weights, weights_before)
    np.testing.assert_array_equal(layer.bias, bias_before)
    np.testing.assert_allclose(layer.weights_gradient_, reference.weights_gradient_)
    np.testing.assert_allclose(layer.bias_gradient_, reference.bias_gradient_)
    np.testing.assert_allclose(input_error, upstream @ weights_before.T)


def test_activation_backward_without_learning_rate():
    layer = ReLU()
    layer.forward(np.array([[-1.0, 2.0]]))
    np.testing.assert_allclose(layer.backward(np.array([[1.0, 1.0]])), [[0.0, 1.0]])
