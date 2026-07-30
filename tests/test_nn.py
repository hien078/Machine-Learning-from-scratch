import numpy as np

from ml_first_principles.nn_core import Dense, Sequential, Sigmoid


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
