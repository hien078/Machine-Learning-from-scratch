"""From-scratch educational machine-learning implementations."""

from __future__ import annotations

from ml_first_principles.data_utils import (
    generate_classification_data,
    generate_regression_data,
    k_fold_split,
    normalize,
    standardize,
    train_test_split,
)
from ml_first_principles.distance_models import KMeans, KNeighborsClassifier
from ml_first_principles.ensemble_models import RandomForestClassifier
from ml_first_principles.generative_models import (
    VAE,
    GANDiscriminator,
    GANGenerator,
    gan_discriminator_loss,
    gan_generator_loss,
    vae_elbo_loss,
)
from ml_first_principles.gnn_models import GATLayer, GCNLayer
from ml_first_principles.linear_models import (
    LassoRegression,
    LinearRegression,
    LogisticRegression,
    PolynomialFeatures,
    RidgeRegression,
)
from ml_first_principles.llm_models import BPETokenizer, LoRALinear, dpo_loss
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
from ml_first_principles.nn_core import Activation, Dense, Layer, ReLU, Sequential, Sigmoid
from ml_first_principles.optimizers import (
    SGD,
    Adam,
    adam,
    coordinate_descent_lasso,
    gradient_descent,
    sgd,
)
from ml_first_principles.probabilistic_models import GaussianNB
from ml_first_principles.rl_models import GridWorldEnv, QLearningAgent
from ml_first_principles.ssl_models import InfoNCELoss, PatchMasking
from ml_first_principles.svm_models import LinearSVC
from ml_first_principles.transformer_core import (
    CausalSelfAttention,
    Embedding,
    LayerNorm,
    TransformerBlock,
    softmax_cross_entropy,
)
from ml_first_principles.tree_models import DecisionTreeClassifier
from ml_first_principles.visualization import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_learning_curve,
    plot_regression,
    plot_regularization_path,
)

__version__ = "0.3.0"  # keep in sync with pyproject.toml [project] version

__all__ = [
    "VAE",
    "Activation",
    "Adam",
    "BPETokenizer",
    "CausalSelfAttention",
    "DecisionTreeClassifier",
    "Dense",
    "Embedding",
    "GANDiscriminator",
    "GANGenerator",
    "GATLayer",
    "GCNLayer",
    "GaussianNB",
    "GridWorldEnv",
    "InfoNCELoss",
    "KMeans",
    "KNeighborsClassifier",
    "LassoRegression",
    "Layer",
    "LayerNorm",
    "LinearRegression",
    "LinearSVC",
    "LoRALinear",
    "LogisticRegression",
    "PatchMasking",
    "PolynomialFeatures",
    "QLearningAgent",
    "RandomForestClassifier",
    "ReLU",
    "RidgeRegression",
    "SGD",
    "Sequential",
    "Sigmoid",
    "TransformerBlock",
    "__version__",
    "accuracy",
    "adam",
    "confusion_matrix",
    "coordinate_descent_lasso",
    "cross_entropy_loss",
    "dpo_loss",
    "f1_score",
    "gan_discriminator_loss",
    "gan_generator_loss",
    "generate_classification_data",
    "generate_regression_data",
    "gradient_descent",
    "k_fold_split",
    "log_loss",
    "mae",
    "mse",
    "normalize",
    "plot_confusion_matrix",
    "plot_decision_boundary",
    "plot_learning_curve",
    "plot_regression",
    "plot_regularization_path",
    "precision",
    "r2_score",
    "recall",
    "rmse",
    "sgd",
    "softmax_cross_entropy",
    "standardize",
    "train_test_split",
    "vae_elbo_loss",
]
