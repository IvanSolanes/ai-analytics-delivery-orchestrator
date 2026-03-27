"""
test_profiling.py
-----------------
Tests for the data profiling node.

All tests are fully deterministic — no LLM calls, no mocks needed.
We test with a small synthetic CSV that mirrors the Give Me Some Credit
schema, so tests are fast and self-contained.

What we test:
  1. Node writes a valid DataProfile to state
  2. Node only writes its own state field
  3. Null rates are computed correctly
  4. High-null columns are flagged correctly
  5. Target distribution is computed correctly
  6. Class imbalance warning is generated
  7. Top correlations are returned in descending order
  8. Quality warnings are human-readable strings
  9. Column type inference works correctly
  10. Profile summary JSON is saved to disk
"""

import json
import os
import tempfile

import pandas as pd
import pytest

from orchestrator.evaluation.golden_cases import CREDIT_RISK_EXPECTED_SCOPE
from orchestrator.nodes.profiling import (
    _compute_target_distribution,
    _compute_top_correlations,
    _generate_quality_warnings,
    _infer_column_type,
    profiling_node,
)
from orchestrator.state import ColumnType, DataProfile, TaskType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def credit_risk_sample_df() -> pd.DataFrame:
    """
    Synthetic dataframe mirroring the Give Me Some Credit schema.
    150 rows — enough to produce meaningful statistics, fast to compute.
    """
    import numpy as np
    rng = np.random.default_rng(seed=42)
    n = 150

    return pd.DataFrame({
        "Unnamed: 0": range(n),
        "SeriousDlqin2yrs": rng.choice([0, 1], size=n, p=[0.93, 0.07]),
        "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1, n),
        "age": rng.integers(20, 80, n),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 5, n),
        "DebtRatio": rng.uniform(0, 2, n),
        # MonthlyIncome has ~20% nulls — mirrors real dataset
        "MonthlyIncome": [
            None if rng.random() < 0.20 else float(rng.integers(1000, 15000))
            for _ in range(n)
        ],
        "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n),
        "NumberOfTimes90DaysLate": rng.integers(0, 5, n),
        "NumberRealEstateLoansOrLines": rng.integers(0, 5, n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 5, n),
        # NumberOfDependents has ~2.6% nulls — mirrors real dataset
        "NumberOfDependents": [
            None if rng.random() < 0.026 else float(rng.integers(0, 5))
            for _ in range(n)
        ],
    })


@pytest.fixture
def sample_csv_path(credit_risk_sample_df, tmp_path) -> str:
    """Write the sample dataframe to a temp CSV and return the path."""
    path = tmp_path / "credit_risk_sample.csv"
    credit_risk_sample_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def profiling_state(sample_csv_path) -> dict:
    """Minimal valid state dict for the profiling node."""
    return {
        "business_brief": "Predict credit risk.",
        "dataset_path": sample_csv_path,
        "retry_count": 0,
        "scoped_problem": CREDIT_RISK_EXPECTED_SCOPE,
    }


# ---------------------------------------------------------------------------
# Tests: profiling_node
# ---------------------------------------------------------------------------

def test_profiling_node_writes_data_profile(profiling_state):
    """Node must return a dict containing a valid DataProfile."""
    result = profiling_node(profiling_state)
    assert "data_profile" in result
    assert isinstance(result["data_profile"], DataProfile)


def test_profiling_node_only_writes_data_profile(profiling_state):
    """Node must only return its own state field."""
    result = profiling_node(profiling_state)
    assert list(result.keys()) == ["data_profile"]


def test_profiling_node_correct_row_count(profiling_state):
    """n_rows in DataProfile must match the actual CSV row count."""
    result = profiling_node(profiling_state)
    assert result["data_profile"].n_rows == 150


def test_profiling_node_correct_column_count(profiling_state):
    """n_columns must match the number of columns in the CSV."""
    result = profiling_node(profiling_state)
    assert result["data_profile"].n_columns == 12


def test_profiling_node_flags_monthly_income_as_high_null(profiling_state):
    """MonthlyIncome (~20% nulls) must appear in high_null_columns."""
    result = profiling_node(profiling_state)
    assert "MonthlyIncome" in result["data_profile"].high_null_columns


def test_profiling_node_target_distribution_sums_to_one(profiling_state):
    """Target distribution proportions must sum to approximately 1.0."""
    result = profiling_node(profiling_state)
    total = sum(result["data_profile"].target_distribution.values())
    assert abs(total - 1.0) < 0.01


def test_profiling_node_generates_quality_warnings(profiling_state):
    """At least one quality warning must be generated for this dataset."""
    result = profiling_node(profiling_state)
    assert len(result["data_profile"].quality_warnings) > 0
    # All warnings must be non-empty strings
    for w in result["data_profile"].quality_warnings:
        assert isinstance(w, str) and len(w) > 0


def test_profiling_node_top_correlations_descending(profiling_state):
    """Correlations must be sorted by absolute value, descending."""
    result = profiling_node(profiling_state)
    correlations = result["data_profile"].top_correlations
    if len(correlations) > 1:
        for i in range(len(correlations) - 1):
            assert (
                abs(correlations[i]["correlation"])
                >= abs(correlations[i + 1]["correlation"])
            )


def test_profiling_node_saves_profile_summary(profiling_state):
    """A profile summary JSON must be written to the profiles output dir."""
    result = profiling_node(profiling_state)
    path = result["data_profile"].profile_report_path
    assert path is not None
    assert os.path.exists(path)

    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

    assert "n_rows" in summary
    assert "target_distribution" in summary
    assert "quality_warnings" in summary


# ---------------------------------------------------------------------------
# Tests: helper functions (unit tested independently)
# ---------------------------------------------------------------------------

def test_infer_column_type_target():
    """Target column must always return ColumnType.TARGET."""
    series = pd.Series([0, 1, 0, 1])
    assert _infer_column_type("SeriousDlqin2yrs", series, "SeriousDlqin2yrs") == ColumnType.TARGET


def test_infer_column_type_id():
    """Unnamed: 0 must be identified as an ID column."""
    series = pd.Series([1, 2, 3, 4])
    assert _infer_column_type("Unnamed: 0", series, "target") == ColumnType.ID


def test_infer_column_type_numeric():
    """Integer/float columns must return ColumnType.NUMERIC."""
    series = pd.Series([1.0, 2.5, 3.1])
    assert _infer_column_type("age", series, "target") == ColumnType.NUMERIC


def test_infer_column_type_categorical():
    """Low-cardinality object column must return ColumnType.CATEGORICAL."""
    series = pd.Series(["yes", "no", "yes", "no"] * 10)
    assert _infer_column_type("approved", series, "target") == ColumnType.CATEGORICAL


def test_compute_target_distribution_classification():
    """Binary target distribution must sum to 1 and include both classes."""
    series = pd.Series([0] * 93 + [1] * 7)
    dist = _compute_target_distribution(series, "binary_classification")
    assert set(dist.keys()) == {"0", "1"}
    assert abs(sum(dist.values()) - 1.0) < 0.01


def test_compute_target_distribution_regression():
    """Regression distribution must return mean, std, min, max."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    dist = _compute_target_distribution(series, "regression")
    assert set(dist.keys()) == {"mean", "std", "min", "max"}
    assert dist["mean"] == 3.0


def test_compute_top_correlations_sorted():
    """Correlations must be sorted by absolute value descending."""
    # strong: perfectly correlated with target (corr=1.0)
    # medium: partially correlated (corr~0.5)
    # weak: no variation — will be excluded (NaN correlation)
    df = pd.DataFrame({
        "target": [0, 0, 0, 0, 1, 1, 1, 1],
        "strong": [0, 0, 0, 0, 1, 1, 1, 1],   # perfect corr = 1.0
        "medium": [0, 1, 0, 0, 1, 1, 0, 1],   # partial corr ~ 0.5
    })
    result = _compute_top_correlations(df, "target")
    assert len(result) >= 1
    assert result[0]["feature"] == "strong"
    # verify descending order
    for i in range(len(result) - 1):
        assert abs(result[i]["correlation"]) >= abs(result[i+1]["correlation"])


def test_generate_quality_warnings_imbalance():
    """Class imbalance below threshold must trigger a warning."""
    warnings = _generate_quality_warnings(
        df=pd.DataFrame({"target": [0] * 95 + [1] * 5}),
        target_column="target",
        task_type_value="binary_classification",
        high_null_columns=[],
        high_cardinality_columns=[],
        target_distribution={"0": 0.95, "1": 0.05},
    )
    assert any("imbalance" in w.lower() for w in warnings)


def test_generate_quality_warnings_null_columns():
    """High-null column must appear in a warning message."""
    df = pd.DataFrame({
        "MonthlyIncome": [None] * 30 + [5000.0] * 70,
        "target": [0] * 100,
    })
    warnings = _generate_quality_warnings(
        df=df,
        target_column="target",
        task_type_value="binary_classification",
        high_null_columns=["MonthlyIncome"],
        high_cardinality_columns=[],
        target_distribution={"0": 1.0},
    )
    assert any("MonthlyIncome" in w for w in warnings)
