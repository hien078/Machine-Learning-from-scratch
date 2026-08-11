"""First-principles pure NumPy implementations of generative models (VAE, GAN)."""

from __future__ import annotations

import numpy as np

SIGMOID_CLIP = 15.0
LOG_EPS = 1e-12


def _he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


def _xavier_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(1.0 / fan_in)


class VAE:
    """Variational Autoencoder forward pass in pure NumPy.

    This is the inference-side reference implementation (encoder,
    reparameterization, decoder, ELBO); the full training loop with
    backpropagation is developed step by step in
    ``topics/19_generative_models/first_principles.ipynb``.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 64,
        latent_dim: int = 10,
        random_state: int | None = None,
    ) -> None:
        """Initialize encoder and decoder weights with fan-in scaled noise.

        Args:
            input_dim: Dimensionality of the (binary or unit-interval) data.
            hidden_dim: Width of the single hidden layer on each side.
            latent_dim: Dimensionality of the latent Gaussian.
            random_state: Seed for the isolated random generator, which also
                drives the reparameterization noise.

        Raises:
            ValueError: If any dimension is not positive.
        """
        if input_dim < 1 or hidden_dim < 1 or latent_dim < 1:
            raise ValueError("input_dim, hidden_dim, and latent_dim must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self._rng = np.random.default_rng(random_state)

        # Encoder parameters (He init before ReLU, Xavier for the linear heads).
        self.W_enc = _he_init(self._rng, input_dim, hidden_dim)
        self.b_enc = np.zeros(hidden_dim)
        self.W_mu = _xavier_init(self._rng, hidden_dim, latent_dim)
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = _xavier_init(self._rng, hidden_dim, latent_dim)
        self.b_logvar = np.zeros(latent_dim)

        # Decoder parameters.
        self.W_dec1 = _he_init(self._rng, latent_dim, hidden_dim)
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = _xavier_init(self._rng, hidden_dim, input_dim)
        self.b_dec2 = np.zeros(input_dim)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -SIGMOID_CLIP, SIGMOID_CLIP)))

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map inputs to the mean and log-variance of the latent Gaussian.

        Args:
            x: Input batch of shape ``(n_samples, input_dim)``.

        Returns:
            ``(mu, logvar)`` arrays of shape ``(n_samples, latent_dim)``.
        """
        h = self._relu(x @ self.W_enc + self.b_enc)
        mu = h @ self.W_mu + self.b_mu
        logvar = h @ self.W_logvar + self.b_logvar
        return mu, logvar

    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Sample ``z = mu + eps * std`` with the reparameterization trick.

        Args:
            mu: Latent means.
            logvar: Latent log-variances.

        Returns:
            A latent sample with the same shape as ``mu``.
        """
        std = np.exp(0.5 * logvar)
        eps = self._rng.standard_normal(std.shape)
        return mu + eps * std

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Map latent samples back to unit-interval reconstructions.

        Args:
            z: Latent batch of shape ``(n_samples, latent_dim)``.

        Returns:
            Reconstructions of shape ``(n_samples, input_dim)``.
        """
        h = self._relu(z @ self.W_dec1 + self.b_dec1)
        return self._sigmoid(h @ self.W_dec2 + self.b_dec2)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run encode, reparameterize, and decode in sequence.

        Args:
            x: Input batch of shape ``(n_samples, input_dim)``.

        Returns:
            ``(reconstructed_x, mu, logvar)``.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_elbo_loss(recon_x: np.ndarray, x: np.ndarray, mu: np.ndarray, logvar: np.ndarray) -> float:
    r"""Compute the negative ELBO: binary cross-entropy plus KL divergence.

    The KL term is the closed form
    :math:`-\tfrac{1}{2}\sum(1+\log\sigma^2-\mu^2-\sigma^2)` for a diagonal
    Gaussian against a standard normal prior. Both terms are summed over the
    whole batch, matching the common summed-ELBO convention.

    Args:
        recon_x: Decoder outputs in ``(0, 1)``.
        x: Targets in ``[0, 1]`` with the same shape as ``recon_x``.
        mu: Latent means.
        logvar: Latent log-variances.

    Returns:
        The total (batch-summed) negative ELBO.
    """
    bce = -np.sum(x * np.log(recon_x + LOG_EPS) + (1 - x) * np.log(1 - recon_x + LOG_EPS))
    kld = -0.5 * np.sum(1 + logvar - np.square(mu) - np.exp(logvar))
    return float(bce + kld)


class GANGenerator:
    """Single-layer GAN generator mapping noise to ``tanh`` samples."""

    def __init__(
        self, latent_dim: int = 10, output_dim: int = 16, random_state: int | None = None
    ) -> None:
        """Initialize the generator weight and bias.

        Args:
            latent_dim: Dimensionality of the input noise.
            output_dim: Dimensionality of the generated samples.
            random_state: Seed for the isolated random generator.

        Raises:
            ValueError: If any dimension is not positive.
        """
        if latent_dim < 1 or output_dim < 1:
            raise ValueError("latent_dim and output_dim must be positive")
        rng = np.random.default_rng(random_state)
        self.W = _xavier_init(rng, latent_dim, output_dim)
        self.b = np.zeros(output_dim)

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Map noise vectors to synthetic samples in ``(-1, 1)``.

        Args:
            z: Noise batch of shape ``(n_samples, latent_dim)``.

        Returns:
            Generated samples of shape ``(n_samples, output_dim)``.
        """
        return np.tanh(z @ self.W + self.b)


class GANDiscriminator:
    """Single-layer GAN discriminator returning real-sample probabilities."""

    def __init__(self, input_dim: int = 16, random_state: int | None = None) -> None:
        """Initialize the discriminator weight and bias.

        Args:
            input_dim: Dimensionality of the input samples.
            random_state: Seed for the isolated random generator.

        Raises:
            ValueError: If ``input_dim`` is not positive.
        """
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        rng = np.random.default_rng(random_state)
        self.W = _xavier_init(rng, input_dim, 1)
        self.b = np.zeros(1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Map samples to the probability of being real.

        Args:
            x: Sample batch of shape ``(n_samples, input_dim)``.

        Returns:
            Probabilities of shape ``(n_samples, 1)``.
        """
        logits = np.clip(x @ self.W + self.b, -SIGMOID_CLIP, SIGMOID_CLIP)
        return 1.0 / (1.0 + np.exp(-logits))


def gan_discriminator_loss(d_real: np.ndarray, d_fake: np.ndarray) -> float:
    r"""Compute the discriminator objective.

    Implements :math:`-\mathbb{E}[\log D(x)] - \mathbb{E}[\log(1 - D(G(z)))]`.

    Args:
        d_real: Discriminator probabilities on real samples.
        d_fake: Discriminator probabilities on generated samples.

    Returns:
        The mean discriminator loss.
    """
    d_real = np.asarray(d_real, dtype=float)
    d_fake = np.asarray(d_fake, dtype=float)
    return float(-np.mean(np.log(d_real + LOG_EPS)) - np.mean(np.log(1.0 - d_fake + LOG_EPS)))


def gan_generator_loss(d_fake: np.ndarray) -> float:
    r"""Compute the non-saturating generator objective :math:`-\mathbb{E}[\log D(G(z))]`.

    Args:
        d_fake: Discriminator probabilities on generated samples.

    Returns:
        The mean generator loss.
    """
    d_fake = np.asarray(d_fake, dtype=float)
    return float(-np.mean(np.log(d_fake + LOG_EPS)))
