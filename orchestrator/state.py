"""
state.py
--------
The shared state that flows through every node in the LangGraph graph.

Design principles:
  1. Every node boundary is a Pydantic model — no raw dicts or untyped strings.
  2. Every field has a clear owner (which node writes it).
  3. Optional fields start as None and are populated as the graph progresses.
  4. Control-flow fields (retry_count, human_feedback) live here too,
     so the graph's routing logic can read them from state without side effects.

Reading this file alone should give a complete picture of the system's data contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# =============================================================================
# ENUMS — shared vocabulary used across multiple models
# =============================================================================

class TaskType(str, Enum):
    """
    The type of ML task inferred from the business brief.
    Using an Enum (not a free string) means the scoping node cannot return
    something ambiguous like "binary classification" vs "classification" —
    it must pick one of these values. This makes downstream branching logic
    (e.g. choosing the right model) deterministic.
    """
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class ColumnType(str, Enum):
    """
    Inferred semantic type of a dataset column.
    Separate from pandas dtype — a column can be dtype=object
    but semantically be an ID (which should be dropped) or a category.
    """
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    ID = "id"           # high-cardinality identifier — should be dropped
    TARGET = "target"   # the column being predicted


# =============================================================================
# NODE 1 OUTPUT: ScopedProblem
# Written by: scoping node (LLM)
# =============================================================================

class ScopedProblem(BaseModel):
    """
    The output of the LLM scoping node.

    Why Pydantic here?
    The LLM uses .with_structured_output(ScopedProblem) from LangChain, which
    instructs the model to return JSON matching this schema. If it does not,
    LangChain raises a validation error rather than passing bad data downstream.

    This model is the contract every other node checks against. For example:
    - The model node checks task_type to choose the right algorithm.
    - The QA node checks that primary_metric matches the one used in ModelResults.
    - The review node checks that limitations are non-empty.

    Example for credit risk:
    - target_column: "SeriousDlqin2yrs"
    - task_type: TaskType.BINARY_CLASSIFICATION
    - success_metric: "roc_auc"
    - features_to_exclude: [] (no ID column in this dataset)
    """
    target_column: str = Field(
        description="The exact column name in the dataset to be predicted."
    )
    task_type: TaskType = Field(
        description="The ML task type inferred from the brief and target column."
    )
    success_metric: str = Field(
        description=(
            "The primary metric used to evaluate the model. "
            "Must match the task type: e.g. 'roc_auc' for binary classification, "
            "'rmse' for regression."
        )
    )
    problem_statement: str = Field(
        description="One or two sentences restating the business problem in precise, testable terms."
    )
    features_to_exclude: list[str] = Field(
        default_factory=list,
        description=(
            "Column names that should be excluded from features. "
            "Typically IDs, free-text fields, or columns that would cause data leakage."
        )
    )
    known_constraints: list[str] = Field(
        default_factory=list,
        description="Business or technical constraints mentioned in the brief."
    )
    out_of_scope: list[str] = Field(
        default_factory=list,
        description="Things explicitly NOT addressed by this analysis."
    )
    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Anticipated limitations of the approach given the brief. "
            "Non-empty list required — the review node flags an empty limitations field."
        )
    )


# =============================================================================
# NODE 2 OUTPUT: DataProfile
# Written by: profiling node (deterministic — ydata-profiling + pandas)
# =============================================================================

class ColumnProfile(BaseModel):
    """Profile statistics for a single column."""
    name: str
    column_type: ColumnType
    null_rate: float = Field(ge=0.0, le=1.0, description="Fraction of missing values.")
    n_unique: int = Field(ge=0)
    sample_values: list[str] = Field(
        default_factory=list,
        description="Up to 5 representative values as strings, for human review."
    )


class DataProfile(BaseModel):
    """
    Deterministic statistical summary of the input dataset.

    Why deterministic?
    Data facts must never come from an LLM. Null rates, distributions,
    and correlations are computed directly with pandas and stored here.
    The review node can then quote these numbers knowing they are ground truth.

    Implementation note:
    We use pandas directly instead of ydata-profiling to avoid dependency
    conflicts on Python 3.12+. The profiling node computes:
      null rates          -> df.isnull().mean()
      unique counts       -> df.nunique()
      distributions       -> df.describe() + value_counts()
      correlations        -> df.corr()[target].abs().sort_values()
      quality warnings    -> custom logic (imbalance, high nulls, cardinality)
    """
    n_rows: int = Field(gt=0)
    n_columns: int = Field(gt=0)
    columns: list[ColumnProfile]
    high_null_columns: list[str] = Field(
        default_factory=list,
        description="Columns with null_rate > 0.2 — flagged for ETL attention."
    )
    high_cardinality_columns: list[str] = Field(
        default_factory=list,
        description="Categorical columns with n_unique > 20 — may need special encoding."
    )
    target_distribution: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "For classification: class label -> proportion. "
            "For regression: {'mean': x, 'std': y, 'min': z, 'max': w}."
        )
    )
    top_correlations: list[dict[str, float]] = Field(
        default_factory=list,
        description=(
            "Top 5 absolute correlations with the target column. "
            "Each dict: {'feature': str, 'correlation': float}."
        )
    )
    quality_warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable warnings: e.g. "
            "'MonthlyIncome has 19.8% missing values', "
            "'Class imbalance detected: 93.3% negative class'."
        )
    )
    profile_report_path: Optional[str] = Field(
        default=None,
        description="Reserved for future use. Always None in current implementation."
    )


# =============================================================================
# NODE 3 OUTPUT: ETLArtifacts
# Written by: ETL node (deterministic — sklearn Pipeline + joblib)
# =============================================================================

class ETLArtifacts(BaseModel):
    """
    What the ETL node produced: a fitted sklearn Pipeline and the column contract.

    Why store feature_columns and target_column here?
    The model node and dashboard node both need to know exactly which columns
    were used. Storing them in state ensures consistency and makes the QA
    consistency check possible: QA can assert that
    model_results.feature_columns == etl_artifacts.feature_columns.
    """
    feature_columns: list[str] = Field(
        description="Ordered list of feature column names fed into the pipeline."
    )
    target_column: str = Field(
        description="The target column name — must match scoped_problem.target_column."
    )
    preprocessing_steps: list[str] = Field(
        description=(
            "Human-readable description of each pipeline step in order. "
            "e.g. ['SimpleImputer(strategy=median) on numeric columns', "
            "'StandardScaler on numeric columns'] — "
            "no categorical columns in Give Me Some Credit dataset."
        )
    )
    n_rows_after_cleaning: int = Field(gt=0)
    n_rows_dropped: int = Field(ge=0)
    pipeline_path: str = Field(
        description="Absolute path to the joblib-serialised sklearn Pipeline."
    )
    processed_data_path: str = Field(
        description="Absolute path to the processed CSV (X_train, X_test, y_train, y_test)."
    )


# =============================================================================
# NODE 4 OUTPUT: ModelResults
# Written by: model node (deterministic — sklearn)
# =============================================================================

class ModelResults(BaseModel):
    """
    Training and evaluation results for the baseline model.

    Why 'baseline'?
    This system makes no claim to produce the best possible model.
    The goal is a reproducible, explainable starting point. The review
    node is explicitly asked to flag if the chosen algorithm is appropriate
    for the task type and data profile.
    """
    algorithm: str = Field(
        description="Name of the algorithm used. e.g. 'LogisticRegression', 'RandomForestClassifier'."
    )
    task_type: TaskType = Field(
        description="Copied from scoped_problem — ensures the model matches the scope."
    )
    cv_scores: list[float] = Field(
        description="Cross-validation scores (one per fold) on the primary metric."
    )
    cv_mean: float = Field(description="Mean of cv_scores.")
    cv_std: float = Field(description="Standard deviation of cv_scores.")
    test_score: float = Field(
        description="Score on the held-out test set, on the primary metric."
    )
    primary_metric: str = Field(
        description="The metric used — must match scoped_problem.success_metric."
    )
    feature_importances: list[dict] = Field(
        default_factory=list,
        description=(
            "Top 10 features by importance. "
            "Each dict: {'feature': str, 'importance': float}."
        )
    )
    model_path: str = Field(
        description="Absolute path to the joblib-serialised trained model."
    )
    training_notes: list[str] = Field(
        default_factory=list,
        description=(
            "Automatically generated notes about the training run. "
            "e.g. 'Class imbalance detected — used class_weight=balanced'."
        )
    )


# =============================================================================
# NODE 6 OUTPUT: ReviewScores
# Written by: review node (LLM with structured output)
# =============================================================================

class ReviewDimension(BaseModel):
    """Score and justification for one rubric dimension."""
    score: float = Field(ge=0.0, le=1.0, description="Score from 0.0 (poor) to 1.0 (excellent).")
    justification: str = Field(description="One or two sentences explaining the score.")
    flags: list[str] = Field(
        default_factory=list,
        description="Specific issues found. Empty list means no issues."
    )


class ReviewScores(BaseModel):
    """
    Rubric-based evaluation of all prior node outputs.

    Five dimensions are scored independently so that a weak score in one
    area does not mask strong performance in others. The composite score
    drives the retry conditional edge in the graph.

    Why LLM for review?
    Judgment calls — 'is this problem statement precise enough?',
    'are the limitations realistic?' — require language understanding.
    But the output is structured (Pydantic), so scores are comparable
    across runs and can be stored, diffed, and plotted.
    """
    problem_clarity: ReviewDimension = Field(
        description=(
            "Is the problem statement precise and testable? "
            "Is the target column and success metric well-defined?"
        )
    )
    data_quality: ReviewDimension = Field(
        description=(
            "Are data quality warnings present and reasonable? "
            "Are high-null and high-cardinality columns flagged?"
        )
    )
    model_appropriateness: ReviewDimension = Field(
        description=(
            "Is the chosen algorithm appropriate for the task type and data size? "
            "Is class imbalance handled if present?"
        )
    )
    dashboard_completeness: ReviewDimension = Field(
        description=(
            "Does the dashboard surface the key metric, feature importances, "
            "and at least one chart? Is it self-contained?"
        )
    )
    limitations_acknowledged: ReviewDimension = Field(
        description=(
            "Are limitations listed realistic and specific to this dataset and brief? "
            "Generic boilerplate ('results may vary') should score low."
        )
    )
    composite_score: float = Field(
        ge=0.0, le=1.0,
        description="Mean of the five dimension scores. Drives the retry edge."
    )
    retry_recommended: bool = Field(
        description="True if composite_score < REVIEW_SCORE_THRESHOLD."
    )
    overall_summary: str = Field(
        description="Two or three sentences summarising the delivery quality."
    )


# =============================================================================
# NODE 8 OUTPUT: QAResult
# Written by: QA node (deterministic assertions + LLM flag scan)
# =============================================================================

class AssertionResult(BaseModel):
    """Result of a single deterministic assertion."""
    name: str = Field(description="Short name for the assertion. e.g. 'target_column_match'.")
    passed: bool
    detail: str = Field(description="What was checked and what value was found.")


class QAResult(BaseModel):
    """
    Quality assurance results combining deterministic checks and LLM flag scanning.

    Why split into deterministic + LLM?
    Some checks are binary and unambiguous — the target column either matches
    the scope or it does not. These are pytest-style assertions. Other checks
    require reading text — does the review text make claims not supported by
    the data profile? That is where the LLM flag scan adds value.
    """
    assertions: list[AssertionResult] = Field(
        description="Results of all deterministic consistency checks."
    )
    assertions_passed: int
    assertions_failed: int
    hallucination_flags: list[str] = Field(
        default_factory=list,
        description=(
            "LLM-identified claims in the review text that are not supported "
            "by the data profile or model results. Empty list = clean."
        )
    )
    overall_passed: bool = Field(
        description="True if assertions_failed == 0 and no hallucination_flags."
    )
    qa_summary: str = Field(
        description="One or two sentences summarising QA outcome."
    )


# =============================================================================
# FINAL OUTPUT: FinalReport
# Written by: QA node (assembles from all prior state fields)
# =============================================================================

class FinalReport(BaseModel):
    """
    The delivery package. Assembled after QA passes (or after human approval).
    Contains no raw data — only summaries and artifact paths.
    """
    problem_statement: str
    success_metric: str
    task_type: TaskType
    model_algorithm: str
    test_score: float
    primary_metric: str
    top_features: list[str] = Field(description="Top 5 feature names by importance.")
    data_quality_warnings: list[str]
    preprocessing_summary: list[str]
    review_composite_score: float
    qa_passed: bool
    limitations: list[str]
    artifacts: dict[str, str] = Field(
        description=(
            "Paths to all generated artifacts. "
            "Keys: 'pipeline', 'model', 'dashboard', 'profile_report'."
        )
    )
    delivery_notes: str = Field(
        description="Any human reviewer notes from the interrupt checkpoint."
    )


# =============================================================================
# THE GRAPH STATE — the TypedDict flowing through every node
# =============================================================================

class AnalyticsState(TypedDict, total=False):
    """
    The shared state object passed between all LangGraph nodes.

    Why TypedDict (not a Pydantic model)?
    LangGraph requires a TypedDict for state — it uses the type annotations
    for its own internal merging and checkpointing logic. Pydantic models
    live at the node boundaries (inputs/outputs), not in the container itself.

    Why total=False?
    Fields are populated incrementally as the graph runs. At start, only
    business_brief and dataset_path are set. total=False means all fields
    are implicitly optional — no need to pre-populate every field with None.

    Field ownership (which node writes each field):
      business_brief      -> user input (main.py)
      dataset_path        -> user input (main.py)
      scoped_problem      -> scoping node
      data_profile        -> profiling node
      etl_artifacts       -> etl node
      model_results       -> model node
      dashboard_code      -> dashboard node
      review_scores       -> review node
      human_feedback      -> human review interrupt
      qa_result           -> qa node
      final_report        -> qa node (assembles delivery)
      retry_count         -> graph routing logic
    """

    # --- Inputs (set by the caller before graph.invoke()) ---
    business_brief: str
    dataset_path: str

    # --- Node outputs (populated as the graph progresses) ---
    scoped_problem: ScopedProblem
    data_profile: DataProfile
    etl_artifacts: ETLArtifacts
    model_results: ModelResults
    dashboard_code: str           # Raw Streamlit Python code as a string
    review_scores: ReviewScores

    # --- Control flow ---
    retry_count: int              # Incremented by the graph router on each retry
    human_feedback: Optional[str] # Set at the human review interrupt; None = approved

    # --- Final output ---
    qa_result: QAResult
    final_report: FinalReport
