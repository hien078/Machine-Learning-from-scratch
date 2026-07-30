"""From-scratch educational machine-learning implementations."""

from ml_first_principles.distance_models import KMeans, KNeighborsClassifier
from ml_first_principles.ensemble_models import RandomForestClassifier
from ml_first_principles.linear_models import (
    LassoRegression,
    LinearRegression,
    LogisticRegression,
    PolynomialFeatures,
    RidgeRegression,
)
from ml_first_principles.probabilistic_models import GaussianNB
from ml_first_principles.svm_models import LinearSVC
from ml_first_principles.tree_models import DecisionTreeClassifier

__all__ = [
    "DecisionTreeClassifier",
    "GaussianNB",
    "KMeans",
    "KNeighborsClassifier",
    "LassoRegression",
    "LinearRegression",
    "LinearSVC",
    "LogisticRegression",
    "PolynomialFeatures",
    "RandomForestClassifier",
    "RidgeRegression",
]
