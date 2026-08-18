"""Package-level export, version, and dependency consistency tests.

These replace rules that used to be prose reminders ("keep in sync manually")
with gates the CI run enforces.
"""

import re
import tomllib
from pathlib import Path

import ml_first_principles

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _distribution_names(specs):
    return {re.split(r"[><=!~\[]", spec)[0].strip().lower() for spec in specs if spec.strip()}


def test_every_name_in_all_resolves():
    for name in ml_first_principles.__all__:
        assert hasattr(ml_first_principles, name), name


def test_key_symbols_are_exported():
    expected = (
        "Dense",
        "GridWorldEnv",
        "adam",
        "accuracy",
        "gan_generator_loss",
        "plot_learning_curve",
        "train_test_split",
    )
    for name in expected:
        assert name in ml_first_principles.__all__, name


def test_package_version_matches_pyproject():
    """The release procedure bumps two files; this makes forgetting one fail CI."""
    assert ml_first_principles.__version__ == _pyproject()["project"]["version"]


def test_requirements_txt_matches_pyproject_runtime_dependencies():
    """The pinned environment and the declared runtime set must name the same
    packages. pyproject.toml owns *which* packages; requirements.txt owns *which
    versions*. Optional extras (torch, dev tooling) live only in pyproject."""
    declared = _distribution_names(_pyproject()["project"]["dependencies"])
    pinned = _distribution_names(
        line
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert pinned == declared, (
        f"only in requirements.txt: {pinned - declared}; only in pyproject: {declared - pinned}"
    )
