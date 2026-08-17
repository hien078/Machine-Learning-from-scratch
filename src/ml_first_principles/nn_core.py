"""Minimal feed-forward neural-network layers and training loop."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

LOG_INTERVAL_EPOCHS = 10


class Layer:
    """Base interface for educational neural-network layers."""

    def forward(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute layer outputs."""
        raise NotImplementedError

    def backward(
        self, output_error: NDArray[np.float64], learning_rate: float | None = None
    ) -> NDArray[np.float64]:
        """Backpropagate an output gradient and return the input gradient.

        Args:
            output_error: Upstream gradient with the layer's output shape.
            learning_rate: Step size for the in-place parameter update. ``None``
                (the default) computes and stores gradients without updating
                any trainable parameters.
        """
        raise NotImplementedError


class Dense(Layer):
    """Fully connected affine layer with He initialization."""

    def __init__(self, input_size: int, output_size: int, random_state: int | None = None) -> None:
        if input_size < 1 or output_size < 1:
            raise ValueError("input_size and output_size must be positive")
        rng = np.random.default_rng(random_state)
        self.weights = rng.standard_normal((input_size, output_size)) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.input_: NDArray[np.float64] | None = None
        self.weights_gradient_: NDArray[np.float64] | None = None
        self.bias_gradient_: NDArray[np.float64] | None = None

    def forward(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return $XW+b$ and cache $X$ for backpropagation."""
        values = np.asarray(input_data, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.weights.shape[0]:
            raise ValueError("input_data has an incompatible shape")
        self.input_ = values
        return values @ self.weights + self.bias

    def backward(
        self, output_error: NDArray[np.float64], learning_rate: float | None = None
    ) -> NDArray[np.float64]:
        """Return the input gradient, optionally applying a gradient-descent update.

        Args:
            output_error: Upstream gradient of shape ``(n_samples, output_size)``.
            learning_rate: Step size for the in-place parameter update. ``None``
                (the default) stores ``weights_gradient_`` and ``bias_gradient_``
                without modifying ``weights`` or ``bias``.
        """
        if self.input_ is None:
            raise RuntimeError("forward must be called before backward")
        gradient = np.asarray(output_error, dtype=float)
        if gradient.ndim != 2 or gradient.shape != (self.input_.shape[0], self.weights.shape[1]):
            raise ValueError("output_error has an incompatible shape")
        if learning_rate is not None and learning_rate < 0.0:
            raise ValueError("learning_rate cannot be negative")
        input_error = gradient @ self.weights.T
        self.weights_gradient_ = self.input_.T @ gradient
        self.bias_gradient_ = gradient.sum(axis=0, keepdims=True)
        if learning_rate is not None:
            self.weights -= learning_rate * self.weights_gradient_
            self.bias -= learning_rate * self.bias_gradient_
        return input_error


class Activation(Layer):
    """Element-wise activation defined by a function and its derivative."""

    def __init__(
        self,
        activation: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        activation_prime: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> None:
        self.activation = activation
        self.activation_prime = activation_prime
        self.input_: NDArray[np.float64] | None = None

    def forward(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the activation element-wise."""
        self.input_ = np.asarray(input_data, dtype=float)
        return self.activation(self.input_)

    def backward(
        self, output_error: NDArray[np.float64], learning_rate: float | None = None
    ) -> NDArray[np.float64]:
        """Multiply by the activation derivative; ``learning_rate`` is unused."""
        del learning_rate
        if self.input_ is None:
            raise RuntimeError("forward must be called before backward")
        return self.activation_prime(self.input_) * np.asarray(output_error, dtype=float)


class ReLU(Activation):
    """Rectified linear unit activation."""

    def __init__(self) -> None:
        super().__init__(
            lambda x: np.maximum(0.0, x),
            lambda x: (x > 0.0).astype(float),
        )


class Sigmoid(Activation):
    """Numerically clipped logistic sigmoid activation."""

    def __init__(self) -> None:
        def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return 1.0 / (1.0 + np.exp(-np.clip(x, -250.0, 250.0)))

        def sigmoid_prime(x: NDArray[np.float64]) -> NDArray[np.float64]:
            value = sigmoid(x)
            return value * (1.0 - value)

        super().__init__(sigmoid, sigmoid_prime)


class Sequential:
    """Minimal sequential network trained with per-sample MSE updates."""

    def __init__(self) -> None:
        self.layers: list[Layer] = []

    def add(self, layer: Layer) -> None:
        """Append a layer."""
        self.layers.append(layer)

    def predict(self, input_data: ArrayLike) -> NDArray[np.float64]:
        """Run a vectorized forward pass."""
        output = np.asarray(input_data, dtype=float)
        if output.ndim != 2:
            raise ValueError("input_data must be two-dimensional")
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        epochs: int,
        learning_rate: float,
        verbose: bool = False,
    ) -> list[float]:
        """Train with MSE and return mean epoch losses."""
        features = np.asarray(x_train, dtype=float)
        targets = np.asarray(y_train, dtype=float)
        if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
            raise ValueError("x_train and y_train must be matching two-dimensional arrays")
        if not self.layers or epochs < 1 or learning_rate <= 0.0:
            raise ValueError(
                "the network must have layers and training parameters must be positive"
            )
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for index in range(features.shape[0]):
                output = features[index : index + 1]
                for layer in self.layers:
                    output = layer.forward(output)
                difference = output - targets[index : index + 1]
                total_loss += float(np.mean(difference**2))
                gradient = 2.0 * difference / difference.size
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)
            loss = total_loss / features.shape[0]
            history.append(loss)
            if verbose and (epoch + 1) % LOG_INTERVAL_EPOCHS == 0:
                logger.info("epoch %d/%d loss %.6f", epoch + 1, epochs, loss)
        return history
