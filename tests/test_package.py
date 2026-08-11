"""Package-level export consistency tests."""

import ml_first_principles


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
