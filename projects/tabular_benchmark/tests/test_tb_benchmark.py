"""Fast checks for the tabular benchmark (small pieces, not the full run)."""

from __future__ import annotations

import numpy as np
import pytest
import tb_benchmark as tb


@pytest.fixture(scope="module")
def reg_data() -> tb.SplitData:
    return tb.load_dataset(tb.REGRESSION)


@pytest.fixture(scope="module")
def clf_data() -> tb.SplitData:
    return tb.load_dataset(tb.CLASSIFICATION)


def test_load_dataset_shapes_and_standardization(reg_data: tb.SplitData) -> None:
    assert reg_data.X_train.shape[0] == reg_data.y_train.shape[0]
    assert reg_data.X_test.shape[0] == reg_data.y_test.shape[0]
    assert reg_data.X_train.shape[1] == reg_data.X_test.shape[1] == 10
    assert np.allclose(reg_data.X_train.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(reg_data.X_train.std(axis=0), 1.0, atol=1e-10)


def test_evaluate_regression_pair_returns_sane_metrics(reg_data: tb.SplitData) -> None:
    pair = tb.make_model_pairs(tb.REGRESSION)[0]  # LinearRegression
    metrics = tb.evaluate_model(pair.scratch(), reg_data, tb.REGRESSION)
    assert set(metrics) == {"r2", "rmse", "fit_time_s", "predict_time_s"}
    assert 0.2 < metrics["r2"] < 0.8
    assert metrics["rmse"] > 0.0
    assert metrics["fit_time_s"] >= 0.0 and metrics["predict_time_s"] >= 0.0


def test_scratch_linear_regression_matches_sklearn(reg_data: tb.SplitData) -> None:
    pair = tb.make_model_pairs(tb.REGRESSION)[0]
    scratch = tb.evaluate_model(pair.scratch(), reg_data, tb.REGRESSION)
    sk = tb.evaluate_model(pair.sklearn(), reg_data, tb.REGRESSION)
    assert np.isclose(scratch["r2"], sk["r2"], atol=1e-8)
    assert np.isclose(scratch["rmse"], sk["rmse"], atol=1e-6)


def test_classification_pair_is_accurate(clf_data: tb.SplitData) -> None:
    pair = next(p for p in tb.make_model_pairs(tb.CLASSIFICATION) if p.name == "GaussianNB")
    scratch = tb.evaluate_model(pair.scratch(), clf_data, tb.CLASSIFICATION)
    sk = tb.evaluate_model(pair.sklearn(), clf_data, tb.CLASSIFICATION)
    assert scratch["accuracy"] > 0.85 and sk["accuracy"] > 0.85
    assert np.isclose(scratch["accuracy"], sk["accuracy"], atol=0.02)
    assert np.isclose(scratch["f1"], sk["f1"], atol=0.02)


def test_seeded_model_is_deterministic(clf_data: tb.SplitData) -> None:
    pair = next(p for p in tb.make_model_pairs(tb.CLASSIFICATION) if p.name == "LinearSVC")
    first = tb.evaluate_model(pair.scratch(), clf_data, tb.CLASSIFICATION)
    second = tb.evaluate_model(pair.scratch(), clf_data, tb.CLASSIFICATION)
    assert first["accuracy"] == second["accuracy"]
    assert first["f1"] == second["f1"]


def test_report_contains_expected_tables(reg_data: tb.SplitData, clf_data: tb.SplitData) -> None:
    fake = {"fit_time_s": 0.001, "predict_time_s": 0.0005}
    reg_results = {
        "LinearRegression": {s: {"r2": 0.5, "rmse": 55.0, **fake} for s in ("scratch", "sklearn")}
    }
    clf_results = {
        "GaussianNB": {s: {"accuracy": 0.93, "f1": 0.94, **fake} for s in ("scratch", "sklearn")}
    }
    report = tb.format_report(reg_data, reg_results, clf_data, clf_results)
    assert report.startswith("# Tabular benchmark")
    assert "| Model | r2 (scratch) | r2 (sklearn) | rmse (scratch) | rmse (sklearn) |" in report
    assert "| Model | accuracy (scratch) | accuracy (sklearn) | f1 (scratch) | f1 (sklearn) |" in (
        report
    )
    assert "## Timing" in report and "## Findings" in report
    assert "| LinearRegression | 0.5000 | 0.5000 | 55.0000 | 55.0000 |" in report
