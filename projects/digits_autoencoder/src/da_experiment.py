"""Digits capstone: MLP classifier and undercomplete autoencoder vs rank-2 PCA.

Everything runs on sklearn's bundled ``load_digits`` (8x8 images, no downloads).
Models are built from ``ml_first_principles.nn_core`` layers and trained with
``ml_first_principles.optimizers.adam`` through a flatten/unflatten parameter
bridge. Outputs (report.md, latent_space.png, reconstructions.png) are written
to the project's ``reports/`` directory.

Run from the repo root:  python projects/digits_autoencoder/src/da_experiment.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml_first_principles.metrics import accuracy, mse
from ml_first_principles.nn_core import Dense, ReLU, Sequential, Sigmoid
from ml_first_principles.optimizers import adam

SEED = 42
TEST_SIZE = 0.25
N_CLASSES = 10
N_FEATURES = 64
LATENT_DIM = 2

CLASSIFIER_HIDDEN = 32
CLASSIFIER_LR = 0.01
CLASSIFIER_ITERS = 400

AE_HIDDEN = 8
AE_LR = 0.01
AE_ITERS = 3000
N_ENCODER_LAYERS = 3  # Dense(64->8), ReLU, Dense(8->2)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

Array = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> tuple[Array, Array, NDArray[np.int64], NDArray[np.int64]]:
    """Return (x_train, x_test, y_train, y_test) with pixels scaled to [0, 1]."""
    digits = load_digits()
    features = digits.data / 16.0
    labels = digits.target
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
    )
    return x_train, x_test, y_train, y_test


def one_hot(labels: NDArray[np.int64], n_classes: int = N_CLASSES) -> Array:
    """Encode integer labels as one-hot rows."""
    return np.eye(n_classes)[labels]


# ---------------------------------------------------------------------------
# Networks from nn_core pieces
# ---------------------------------------------------------------------------
def build_classifier(seed: int = SEED) -> Sequential:
    """MLP 64 -> 32 (ReLU) -> 10 (Sigmoid), trained on one-hot targets with MSE."""
    net = Sequential()
    net.add(Dense(N_FEATURES, CLASSIFIER_HIDDEN, random_state=seed))
    net.add(ReLU())
    net.add(Dense(CLASSIFIER_HIDDEN, N_CLASSES, random_state=seed + 1))
    net.add(Sigmoid())
    return net


def build_autoencoder(seed: int = SEED) -> Sequential:
    """Undercomplete autoencoder 64 -> 8 (ReLU) -> 2 -> 8 (ReLU) -> 64 (Sigmoid)."""
    net = Sequential()
    net.add(Dense(N_FEATURES, AE_HIDDEN, random_state=seed))
    net.add(ReLU())
    net.add(Dense(AE_HIDDEN, LATENT_DIM, random_state=seed + 1))
    net.add(Dense(LATENT_DIM, AE_HIDDEN, random_state=seed + 2))
    net.add(ReLU())
    net.add(Dense(AE_HIDDEN, N_FEATURES, random_state=seed + 3))
    net.add(Sigmoid())
    return net


def encode(net: Sequential, features: Array) -> Array:
    """Map inputs to the 2-D latent space using the encoder half of the autoencoder."""
    output = np.asarray(features, dtype=float)
    for layer in net.layers[:N_ENCODER_LAYERS]:
        output = layer.forward(output)
    return output


# ---------------------------------------------------------------------------
# Parameter bridge: Sequential <-> flat vector, for optimizers.adam
# ---------------------------------------------------------------------------
def get_params(net: Sequential) -> Array:
    """Flatten all Dense weights and biases into one vector."""
    parts = []
    for layer in net.layers:
        if isinstance(layer, Dense):
            parts.append(layer.weights.ravel())
            parts.append(layer.bias.ravel())
    return np.concatenate(parts)


def set_params(net: Sequential, flat: Array) -> None:
    """Write a flat parameter vector back into the network's Dense layers."""
    offset = 0
    for layer in net.layers:
        if isinstance(layer, Dense):
            for attr in ("weights", "bias"):
                param = getattr(layer, attr)
                size = param.size
                setattr(layer, attr, flat[offset : offset + size].reshape(param.shape))
                offset += size
    if offset != flat.size:
        raise ValueError("flat parameter vector does not match the network size")


def loss_and_grad(
    net: Sequential, features: Array, targets: Array, flat: Array
) -> tuple[float, Array]:
    """Full-batch MSE loss and its gradient w.r.t. the flat parameter vector.

    Layers are run with ``backward(gradient, learning_rate=0.0)`` so they compute
    and cache gradients without updating parameters; ``optimizers.adam`` owns the
    update instead.
    """
    set_params(net, flat)
    output = features
    for layer in net.layers:
        output = layer.forward(output)
    difference = output - targets
    loss = float(np.mean(difference**2))
    gradient = 2.0 * difference / difference.size
    for layer in reversed(net.layers):
        gradient = layer.backward(gradient, 0.0)
    parts = []
    for layer in net.layers:
        if isinstance(layer, Dense):
            parts.append(layer.weights_gradient_.ravel())
            parts.append(layer.bias_gradient_.ravel())
    return loss, np.concatenate(parts)


def train_adam(
    net: Sequential, features: Array, targets: Array, lr: float, max_iter: int
) -> list[float]:
    """Train a Sequential network with full-batch Adam; return the loss history."""
    losses: list[float] = []

    def gradient_fn(flat: Array) -> Array:
        loss, grad = loss_and_grad(net, features, targets, flat)
        losses.append(loss)
        return grad

    final, _ = adam(gradient_fn, get_params(net), lr=lr, max_iter=max_iter, tol=1e-12)
    set_params(net, final)
    return losses


# ---------------------------------------------------------------------------
# PCA from scratch (numpy eigh)
# ---------------------------------------------------------------------------
def pca_fit(features: Array, n_components: int) -> tuple[Array, Array]:
    """Return (mean, components) via eigendecomposition of the covariance matrix.

    ``components`` has shape (n_features, n_components), columns sorted by
    decreasing eigenvalue.
    """
    mean = features.mean(axis=0)
    centered = features - mean
    covariance = centered.T @ centered / (features.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return mean, eigenvectors[:, order[:n_components]]


def pca_transform(features: Array, mean: Array, components: Array) -> Array:
    """Project onto the principal components."""
    return (features - mean) @ components


def pca_reconstruct(scores: Array, mean: Array, components: Array) -> Array:
    """Map principal-component scores back to feature space."""
    return scores @ components.T + mean


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_latent_spaces(
    z_ae: Array,
    z_pca: Array,
    labels: NDArray[np.int64],
    path: Path,
) -> None:
    """Side-by-side 2-D scatter: autoencoder latent space vs PCA projection."""
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
    panels = [
        (axes[0], z_ae, "Autoencoder latent space (test set)", "latent $z_1$", "latent $z_2$"),
        (axes[1], z_pca, "PCA projection (test set)", "PC 1", "PC 2"),
    ]
    for ax, coords, title, xlabel, ylabel in panels:
        for digit in range(N_CLASSES):
            mask = labels == digit
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=14,
                color=cmap(digit),
                alpha=0.65,
                linewidths=0,
                label=str(digit),
            )
            # Secondary encoding: annotate each class at its median position.
            cx, cy = np.median(coords[mask, 0]), np.median(coords[mask, 1])
            ax.text(
                cx,
                cy,
                str(digit),
                fontsize=13,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"boxstyle": "circle,pad=0.15", "fc": "white", "ec": cmap(digit), "lw": 1.2},
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linewidth=0.5)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, title="digit", loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(rect=(0.0, 0.0, 0.93, 1.0))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reconstructions(
    x_test: Array,
    x_ae: Array,
    x_pca: Array,
    labels: NDArray[np.int64],
    path: Path,
) -> None:
    """Original vs autoencoder vs rank-2 PCA reconstructions, one column per digit."""
    indices = [int(np.flatnonzero(labels == digit)[0]) for digit in range(N_CLASSES)]
    rows = [("Original", x_test), ("Autoencoder", x_ae), ("PCA (rank 2)", x_pca)]
    fig, axes = plt.subplots(3, N_CLASSES, figsize=(12, 4.2))
    for row, (row_name, images) in enumerate(rows):
        for col, index in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(images[index].reshape(8, 8), cmap="gray_r", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(str(labels[index]), fontsize=10)
            if col == 0:
                ax.set_ylabel(row_name, fontsize=10)
    fig.suptitle("Test-set reconstructions: 2-D bottleneck", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(results: dict[str, float], wall_time: float, path: Path) -> None:
    """Write reports/report.md with tables, embedded figures, and findings."""
    ae_vs_pca = results["mse_ae"] / results["mse_pca"]
    lines = [
        "# Digits capstone report",
        "",
        f"Dataset: sklearn `load_digits` (1797 samples, 8x8 images, pixels scaled to [0, 1]), "
        f"{int(results['n_train'])} train / {int(results['n_test'])} test "
        f"(stratified, seed {SEED}).",
        "",
        "## (a) Classification — from-scratch MLP vs sklearn baseline",
        "",
        "| Model | Test accuracy |",
        "|---|---|",
        f"| MLP 64-{CLASSIFIER_HIDDEN}-10 (nn_core + Adam, MSE on one-hot) | "
        f"{results['acc_mlp']:.4f} |",
        f"| sklearn `LogisticRegression` (lbfgs) | {results['acc_logreg']:.4f} |",
        "",
        f"Final MLP training loss: {results['mlp_final_loss']:.6f} "
        f"after {CLASSIFIER_ITERS} full-batch Adam steps (lr = {CLASSIFIER_LR}).",
        "",
        "## (b) 2-D latent space — autoencoder vs PCA",
        "",
        f"Undercomplete autoencoder 64 -> {AE_HIDDEN} (ReLU) -> {LATENT_DIM} -> "
        f"{AE_HIDDEN} (ReLU) -> 64 (Sigmoid), trained with full-batch Adam "
        f"(lr = {AE_LR}, {AE_ITERS} steps). PCA is computed from scratch via "
        "`numpy.linalg.eigh` on the training covariance matrix.",
        "",
        "![2-D latent space, autoencoder vs PCA](latent_space.png)",
        "",
        "## (c) Reconstruction error — autoencoder vs rank-2 PCA",
        "",
        "| Model (2-D bottleneck) | Test MSE per pixel |",
        "|---|---|",
        f"| Autoencoder | {results['mse_ae']:.5f} |",
        f"| PCA (rank 2) | {results['mse_pca']:.5f} |",
        "",
        f"AE / PCA error ratio: {ae_vs_pca:.3f}",
        "",
        "![Original vs reconstructed digits](reconstructions.png)",
        "",
        "## Findings",
        "",
        f"- The from-scratch MLP reaches {results['acc_mlp']:.1%} test accuracy, "
        f"{'above' if results['acc_mlp'] >= results['acc_logreg'] else 'below'} the "
        f"logistic-regression baseline ({results['acc_logreg']:.1%}), despite using "
        "plain MSE on one-hot targets rather than cross-entropy.",
        "- Both 2-D embeddings separate visually distinct digits (0, 6, 4); the "
        "autoencoder's nonlinear encoder curves the space and pulls some classes "
        "apart more than PCA's linear projection, but with only 2 latent units "
        "several digit pairs (e.g. the loopy 8/9/3 group) still overlap in both.",
        "- "
        + (
            "Through the same 2-D bottleneck, the nonlinear autoencoder reconstructs "
            f"with lower error than rank-2 PCA (ratio {ae_vs_pca:.3f}): the ReLU "
            "layers let it use a curved 2-D manifold, while PCA is restricted to the "
            "best flat 2-D subspace."
            if ae_vs_pca < 1.0
            else "Rank-2 PCA edges out this small autoencoder "
            f"(ratio {ae_vs_pca:.3f}): PCA is the optimal *linear* rank-2 "
            "reconstruction, and the narrow 8-unit hidden layer plus finite "
            "training budget keep the nonlinear model from fully exploiting its "
            "curved manifold."
        ),
        "- Reconstructions from 2 dimensions are heavily smoothed prototypes: "
        "per-sample stroke detail is lost, which is expected from a 32x "
        "compression.",
        "",
        f"Total wall time: {wall_time:.1f} s (single CPU, deterministic, seed {SEED}).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full experiment and write reports/ artifacts."""
    start = time.perf_counter()
    matplotlib.use("Agg")
    np.random.seed(SEED)
    REPORTS_DIR.mkdir(exist_ok=True)

    x_train, x_test, y_train, y_test = load_data()

    # (a) Classifier: from-scratch MLP vs sklearn LogisticRegression.
    classifier = build_classifier()
    mlp_losses = train_adam(
        classifier, x_train, one_hot(y_train), lr=CLASSIFIER_LR, max_iter=CLASSIFIER_ITERS
    )
    y_pred_mlp = classifier.predict(x_test).argmax(axis=1)
    acc_mlp = accuracy(y_test, y_pred_mlp)

    baseline = LogisticRegression(max_iter=2000, random_state=SEED)
    baseline.fit(x_train, y_train)
    acc_logreg = accuracy(y_test, baseline.predict(x_test))

    # (b, c) Autoencoder vs from-scratch rank-2 PCA.
    autoencoder = build_autoencoder()
    ae_losses = train_adam(autoencoder, x_train, x_train, lr=AE_LR, max_iter=AE_ITERS)
    x_test_ae = autoencoder.predict(x_test)
    z_test_ae = encode(autoencoder, x_test)
    # metrics.mse accepts only 1-D arrays, so image matrices are raveled.
    mse_ae = mse(x_test.ravel(), x_test_ae.ravel())

    pca_mean, pca_components = pca_fit(x_train, LATENT_DIM)
    z_test_pca = pca_transform(x_test, pca_mean, pca_components)
    x_test_pca = pca_reconstruct(z_test_pca, pca_mean, pca_components)
    mse_pca = mse(x_test.ravel(), x_test_pca.ravel())

    plot_latent_spaces(z_test_ae, z_test_pca, y_test, REPORTS_DIR / "latent_space.png")
    plot_reconstructions(x_test, x_test_ae, x_test_pca, y_test, REPORTS_DIR / "reconstructions.png")

    wall_time = time.perf_counter() - start
    results = {
        "n_train": float(x_train.shape[0]),
        "n_test": float(x_test.shape[0]),
        "acc_mlp": acc_mlp,
        "acc_logreg": acc_logreg,
        "mlp_final_loss": mlp_losses[-1],
        "ae_final_loss": ae_losses[-1],
        "mse_ae": mse_ae,
        "mse_pca": mse_pca,
    }
    write_report(results, wall_time, REPORTS_DIR / "report.md")

    print(f"MLP test accuracy:        {acc_mlp:.4f}")
    print(f"LogReg test accuracy:     {acc_logreg:.4f}")
    print(f"AE test MSE (2-D):        {mse_ae:.5f}")
    print(f"PCA rank-2 test MSE:      {mse_pca:.5f}")
    print(f"Wall time: {wall_time:.1f} s — outputs in {REPORTS_DIR}")


if __name__ == "__main__":
    main()
