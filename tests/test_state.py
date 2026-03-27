"""
test_state.py
-------------
Tests for every Pydantic model in state.py.

Philosophy: test the contract, not the implementation.
These tests verify that:
  1. Valid data is accepted correctly.
  2. Invalid data raises a ValidationError (not silently accepted).
  3. Enum values reject arbitrary strings.
  4. Numeric constraints (ge, le, gt) are enforced.

These tests run without any LLM calls or file I/O — they are pure, fast,
and should pass before any node code is written.
"""

import pytest
from pydantic import ValidationError

from orchestrator.state import (
    AnalyticsState,
    AssertionResult,
    ColumnProfile,
    ColumnType,
    DataProfile,
    ETLArtifacts,
    FinalReport,
    ModelResults,
    QAResult,
    ReviewDimension,
    ReviewScores,
    ScopedProblem,
    TaskType,
)


# =============================================================================
# FIXTURES — reusable valid model instances
# =============================================================================

@pytest.fixture
def valid_scoped_problem():
    return ScopedProblem(
        target_column="SeriousDlqin2yrs",
        task_type=TaskType.BINARY_CLASSIFICATION,
        success_metric="roc_auc",
        problem_statement=(
            "Predict whether a borrower will experience financial distress "
            "(90+ days past due) within two years, using financial history "
            "and credit utilisation features."
        ),
        features_to_exclude=[],
        known_constraints=[
            "No external data sources available.",
            "Model must be explainable to credit analysts.",
        ],
        out_of_scope=["Loan approval policy decisions."],
        limitations=[
            "Baseline model only — no hyperparameter tuning.",
            "MonthlyIncome and NumberOfDependents have significant missing values.",
            "Severe class imbalance (~93% non-default) may affect recall.",
        ],
    )


@pytest.fixture
def valid_data_profile():
    return DataProfile(
        n_rows=150000,
        n_columns=11,
        columns=[
            ColumnProfile(
                name="RevolvingUtilizationOfUnsecuredLines",
                column_type=ColumnType.NUMERIC,
                null_rate=0.0,
                n_unique=27116,
                sample_values=["0.766", "0.957", "0.658", "0.233", "0.907"],
            ),
            ColumnProfile(
                name="MonthlyIncome",
                column_type=ColumnType.NUMERIC,
                null_rate=0.198,
                n_unique=13414,
                sample_values=["5400", "2916", "8333", "3500", "7500"],
            ),
            ColumnProfile(
                name="SeriousDlqin2yrs",
                column_type=ColumnType.TARGET,
                null_rate=0.0,
                n_unique=2,
                sample_values=["0", "1"],
            ),
        ],
        high_null_columns=["MonthlyIncome", "NumberOfDependents"],
        high_cardinality_columns=[],
        target_distribution={"0": 0.933, "1": 0.067},
        quality_warnings=[
            "MonthlyIncome has 19.8% missing values.",
            "NumberOfDependents has 2.6% missing values.",
            "Class imbalance detected: 93.3% negative class.",
        ],
    )


@pytest.fixture
def valid_etl_artifacts():
    return ETLArtifacts(
        feature_columns=[
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        ],
        target_column="SeriousDlqin2yrs",
        preprocessing_steps=[
            "SimpleImputer(strategy=median) on MonthlyIncome and NumberOfDependents",
            "StandardScaler on all numeric columns",
        ],
        n_rows_after_cleaning=150000,
        n_rows_dropped=0,
        pipeline_path="/outputs/models/pipeline.joblib",
        processed_data_path="/outputs/models/processed_data.csv",
    )


@pytest.fixture
def valid_model_results():
    return ModelResults(
        algorithm="LogisticRegression",
        task_type=TaskType.BINARY_CLASSIFICATION,
        cv_scores=[0.856, 0.861, 0.849, 0.858, 0.863],
        cv_mean=0.857,
        cv_std=0.005,
        test_score=0.854,
        primary_metric="roc_auc",
        feature_importances=[
            {"feature": "RevolvingUtilizationOfUnsecuredLines", "importance": 0.41},
            {"feature": "NumberOfTimes90DaysLate", "importance": 0.29},
        ],
        model_path="/outputs/models/model.joblib",
        training_notes=["class_weight=balanced used due to 93.3% class imbalance."],
    )


@pytest.fixture
def valid_review_scores():
    dim = ReviewDimension(score=0.8, justification="Well defined.", flags=[])
    return ReviewScores(
        problem_clarity=dim,
        data_quality=dim,
        model_appropriateness=dim,
        dashboard_completeness=dim,
        limitations_acknowledged=dim,
        composite_score=0.8,
        retry_recommended=False,
        overall_summary="Delivery is solid with minor gaps.",
    )


# =============================================================================
# TESTS: TaskType enum
# =============================================================================

def test_task_type_valid_values():
    assert TaskType.BINARY_CLASSIFICATION == "binary_classification"
    assert TaskType.REGRESSION == "regression"

def test_task_type_rejects_arbitrary_string():
    with pytest.raises(ValidationError):
        ScopedProblem(
            target_column="churn",
            task_type="not_a_real_task",   # invalid enum value
            success_metric="roc_auc",
            problem_statement="Test.",
        )


# =============================================================================
# TESTS: ScopedProblem
# =============================================================================

def test_scoped_problem_valid(valid_scoped_problem):
    assert valid_scoped_problem.target_column == "SeriousDlqin2yrs"
    assert valid_scoped_problem.task_type == TaskType.BINARY_CLASSIFICATION

def test_scoped_problem_default_lists():
    """Fields with default_factory=list should initialise as empty lists, not None."""
    p = ScopedProblem(
        target_column="price",
        task_type=TaskType.REGRESSION,
        success_metric="rmse",
        problem_statement="Predict house prices.",
    )
    assert p.features_to_exclude == []
    assert p.limitations == []
    assert p.known_constraints == []

def test_scoped_problem_requires_target_column():
    with pytest.raises(ValidationError):
        ScopedProblem(
            task_type=TaskType.REGRESSION,
            success_metric="rmse",
            problem_statement="Missing target.",
        )


# =============================================================================
# TESTS: DataProfile
# =============================================================================

def test_data_profile_valid(valid_data_profile):
    assert valid_data_profile.n_rows == 150000
    assert len(valid_data_profile.columns) == 3

def test_data_profile_null_rate_constraint():
    """null_rate must be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        ColumnProfile(
            name="bad_col",
            column_type=ColumnType.NUMERIC,
            null_rate=1.5,   # > 1.0 — invalid
            n_unique=10,
        )

def test_data_profile_n_rows_must_be_positive():
    with pytest.raises(ValidationError):
        DataProfile(n_rows=0, n_columns=5, columns=[])  # gt=0 violated


# =============================================================================
# TESTS: ETLArtifacts
# =============================================================================

def test_etl_artifacts_valid(valid_etl_artifacts):
    assert "RevolvingUtilizationOfUnsecuredLines" in valid_etl_artifacts.feature_columns
    assert valid_etl_artifacts.target_column == "SeriousDlqin2yrs"

def test_etl_n_rows_dropped_cannot_be_negative():
    with pytest.raises(ValidationError):
        ETLArtifacts(
            feature_columns=["a"],
            target_column="y",
            preprocessing_steps=["step1"],
            n_rows_after_cleaning=100,
            n_rows_dropped=-1,   # ge=0 violated
            pipeline_path="/p.joblib",
            processed_data_path="/d.csv",
        )


# =============================================================================
# TESTS: ModelResults
# =============================================================================

def test_model_results_valid(valid_model_results):
    assert valid_model_results.cv_mean == 0.857
    assert valid_model_results.algorithm == "LogisticRegression"

def test_model_results_feature_importances_default_empty():
    m = ModelResults(
        algorithm="LinearRegression",
        task_type=TaskType.REGRESSION,
        cv_scores=[0.75],
        cv_mean=0.75,
        cv_std=0.0,
        test_score=0.74,
        primary_metric="rmse",
        model_path="/model.joblib",
    )
    assert m.feature_importances == []


# =============================================================================
# TESTS: ReviewScores
# =============================================================================

def test_review_dimension_score_bounds():
    """Score must be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        ReviewDimension(score=1.5, justification="Too high.", flags=[])

def test_review_composite_score_bounds():
    dim = ReviewDimension(score=0.7, justification="OK.", flags=[])
    with pytest.raises(ValidationError):
        ReviewScores(
            problem_clarity=dim,
            data_quality=dim,
            model_appropriateness=dim,
            dashboard_completeness=dim,
            limitations_acknowledged=dim,
            composite_score=-0.1,   # ge=0.0 violated
            retry_recommended=True,
            overall_summary="Bad.",
        )


# =============================================================================
# TESTS: QAResult
# =============================================================================

def test_qa_result_valid():
    result = QAResult(
        assertions=[
            AssertionResult(
                name="target_column_match",
                passed=True,
                detail="scoped target 'churn' matches etl target 'churn'.",
            )
        ],
        assertions_passed=1,
        assertions_failed=0,
        hallucination_flags=[],
        overall_passed=True,
        qa_summary="All checks passed.",
    )
    assert result.overall_passed is True
    assert result.assertions_failed == 0


# =============================================================================
# TESTS: AnalyticsState as a TypedDict
# =============================================================================

def test_analytics_state_accepts_partial_state():
    state: AnalyticsState = {
        "business_brief": (
            "Predict whether a borrower will experience financial distress "
            "within two years."
        ),
        "dataset_path": "/data/credit_risk_sample.csv",
        "retry_count": 0,
    }
    assert "business_brief" in state
    assert "scoped_problem" not in state

def test_analytics_state_can_hold_pydantic_models(
    valid_scoped_problem, valid_data_profile
):
    state: AnalyticsState = {
        "business_brief": (
            "Predict whether a borrower will experience financial distress "
            "within two years."
        ),
        "dataset_path": "/data/credit_risk_sample.csv",
        "retry_count": 0,
        "scoped_problem": valid_scoped_problem,
        "data_profile": valid_data_profile,
    }
    assert state["scoped_problem"].target_column == "SeriousDlqin2yrs"
    assert state["data_profile"].n_rows == 150000
