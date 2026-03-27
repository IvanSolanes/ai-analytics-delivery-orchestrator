"""
nodes/profiling.py
------------------
Node 2: Data Profiling

Responsibility:
  Read the dataset from disk, compute deterministic statistics,
  and write a validated DataProfile to state.

Why fully deterministic?
  Data facts must never come from an LLM. Null rates, distributions,
  and correlations computed here are ground truth — the review node
  quotes them, the QA node asserts against them.

Why pandas directly (no ydata-profiling)?
  ydata-profiling has unstable support on Python 3.12+. We implement
  the same outputs — null rates, distributions, correlations, warnings —
  in ~60 lines of readable pandas code. A reviewer can audit every number.

Inputs from state:
  - dataset_path: str
  - scoped_problem: ScopedProblem

Outputs to state:
  - data_profile: DataProfile
"""

from pathlib import Path

import pandas as pd
from rich.console import Console

from orchestrator.config import settings
from orchestrator.state import (
    AnalyticsState,
    ColumnProfile,
    ColumnType,
    DataProfile,
)

console = Console()

# Thresholds — centralised here so they are easy to find and adjust
_NULL_RATE_THRESHOLD = 0.20      # columns above this are flagged
_HIGH_CARDINALITY_THRESHOLD = 20 # categorical columns above this are flagged
_IMBALANCE_THRESHOLD = 0.10      # minority class below this triggers a warning
_TOP_N_CORRELATIONS = 5          # how many top correlations to store


# ---------------------------------------------------------------------------
# Column type inference
# ---------------------------------------------------------------------------
# We infer a semantic ColumnType from the pandas dtype and column name.
# This is separate from dtype — a column can be dtype=int64 but semantically
# be an ID that should be excluded from features.
# ---------------------------------------------------------------------------

def _infer_column_type(
    col: str,
    series: pd.Series,
    target_column: str,
) -> ColumnType:
    """
    Infer the semantic type of a column.

    Priority order:
      1. If it is the target column -> TARGET
      2. If the name looks like an ID (unnamed index, ends with _id) -> ID
      3. If dtype is numeric -> NUMERIC
      4. If dtype is datetime -> DATETIME
      5. If dtype is object with many unique values -> TEXT
      6. Otherwise -> CATEGORICAL
    """
    if col == target_column:
        return ColumnType.TARGET

    col_lower = col.lower()
    if col_lower in ("unnamed: 0", "id") or col_lower.endswith("_id"):
        return ColumnType.ID

    if pd.api.types.is_numeric_dtype(series):
        return ColumnType.NUMERIC

    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    # Object dtype: distinguish free text from categorical
    # Heuristic: if more than 50% of values are unique, treat as text
    if series.dtype == object:
        uniqueness_ratio = series.nunique() / max(len(series.dropna()), 1)
        if uniqueness_ratio > 0.5:
            return ColumnType.TEXT

    return ColumnType.CATEGORICAL


# ---------------------------------------------------------------------------
# Target distribution
# ---------------------------------------------------------------------------

def _compute_target_distribution(
    series: pd.Series,
    task_type_value: str,
) -> dict[str, float]:
    """
    Compute target distribution.

    For classification: {label: proportion} e.g. {"0": 0.933, "1": 0.067}
    For regression: {stat: value} e.g. {"mean": 5.2, "std": 1.1, ...}
    """
    if "classification" in task_type_value:
        counts = series.value_counts(normalize=True)
        return {str(k): round(float(v), 4) for k, v in counts.items()}
    else:
        desc = series.describe()
        return {
            "mean": round(float(desc["mean"]), 4),
            "std": round(float(desc["std"]), 4),
            "min": round(float(desc["min"]), 4),
            "max": round(float(desc["max"]), 4),
        }


# ---------------------------------------------------------------------------
# Quality warnings
# ---------------------------------------------------------------------------

def _generate_quality_warnings(
    df: pd.DataFrame,
    target_column: str,
    task_type_value: str,
    high_null_columns: list[str],
    high_cardinality_columns: list[str],
    target_distribution: dict[str, float],
) -> list[str]:
    """
    Generate human-readable quality warnings.

    These warnings are surfaced in the dashboard and delivery report.
    They are deterministic — same dataset always produces same warnings.
    """
    warnings = []

    # Null warnings
    for col in high_null_columns:
        null_rate = df[col].isnull().mean()
        warnings.append(
            f"'{col}' has {null_rate:.1%} missing values — imputation required."
        )

    # Class imbalance warning (classification only)
    if "classification" in task_type_value:
        minority_proportion = min(target_distribution.values())
        if minority_proportion < _IMBALANCE_THRESHOLD:
            warnings.append(
                f"Class imbalance detected: minority class is "
                f"{minority_proportion:.1%} of the dataset. "
                f"Consider class_weight='balanced' in the model."
            )

    # High cardinality warning
    for col in high_cardinality_columns:
        n_unique = df[col].nunique()
        warnings.append(
            f"'{col}' has {n_unique} unique values — "
            f"consider target encoding or grouping rare categories."
        )

    # Duplicate rows
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        warnings.append(
            f"{n_duplicates} duplicate rows detected — "
            f"review before training to avoid data leakage."
        )

    return warnings


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def _compute_top_correlations(
    df: pd.DataFrame,
    target_column: str,
) -> list[dict]:
    """
    Compute top N absolute correlations between numeric features and target.

    Returns a list of dicts: [{"feature": str, "correlation": float}, ...]
    sorted by absolute correlation descending.

    Only numeric columns are included — non-numeric columns cannot be
    correlated with the target using Pearson correlation.
    """
    # Encode target as numeric if needed (for binary classification)
    target_series = pd.to_numeric(df[target_column], errors="coerce")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_column]

    if not feature_cols or target_series.isna().all():
        return []

    correlations = []
    for col in feature_cols:
        feature_series = pd.to_numeric(df[col], errors="coerce")
        corr = feature_series.corr(target_series)
        if pd.notna(corr):
            correlations.append({
                "feature": col,
                "correlation": round(float(corr), 4),
            })

    # Sort by absolute correlation, descending
    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return correlations[:_TOP_N_CORRELATIONS]


# ---------------------------------------------------------------------------
# The node function
# ---------------------------------------------------------------------------

def profiling_node(state: AnalyticsState) -> dict:
    """
    Data profiling node.

    Reads the dataset, computes deterministic statistics, and returns
    a validated DataProfile. No LLM calls — all facts, all pandas.

    Returns:
        dict with key "data_profile" containing a DataProfile instance.

    Raises:
        FileNotFoundError: if dataset_path does not exist.
        KeyError: if target_column from scoped_problem is not in the dataset.
    """
    console.rule("[bold]Node 2: Data Profiling[/bold]")

    dataset_path = state["dataset_path"]
    scoped_problem = state["scoped_problem"]
    target_column = scoped_problem.target_column
    task_type_value = scoped_problem.task_type.value

    # --- Step 1: Load the full dataset ---
    console.print(f"  Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    n_rows, n_cols = df.shape
    console.print(f"  Shape: {n_rows:,} rows x {n_cols} columns")

    # --- Step 2: Profile each column ---
    column_profiles = []
    high_null_columns = []
    high_cardinality_columns = []

    for col in df.columns:
        series = df[col]
        null_rate = float(series.isnull().mean())
        n_unique = int(series.nunique())
        col_type = _infer_column_type(col, series, target_column)

        # Sample up to 5 representative non-null values
        sample_values = (
            series.dropna()
            .astype(str)
            .unique()[:5]
            .tolist()
        )

        column_profiles.append(ColumnProfile(
            name=col,
            column_type=col_type,
            null_rate=round(null_rate, 4),
            n_unique=n_unique,
            sample_values=sample_values,
        ))

        # Flag columns needing attention
        if null_rate > _NULL_RATE_THRESHOLD and col != target_column:
            high_null_columns.append(col)

        if (
            col_type == ColumnType.CATEGORICAL
            and n_unique > _HIGH_CARDINALITY_THRESHOLD
            and col != target_column
        ):
            high_cardinality_columns.append(col)

    console.print(f"  High-null columns (>{_NULL_RATE_THRESHOLD:.0%}): "
                  f"{high_null_columns or 'none'}")
    console.print(f"  High-cardinality columns: "
                  f"{high_cardinality_columns or 'none'}")

    # --- Step 3: Target distribution ---
    target_distribution = _compute_target_distribution(
        df[target_column], task_type_value
    )
    console.print(f"  Target distribution: {target_distribution}")

    # --- Step 4: Top correlations with target ---
    top_correlations = _compute_top_correlations(df, target_column)
    console.print(
        f"  Top correlation: "
        f"{top_correlations[0] if top_correlations else 'n/a'}"
    )

    # --- Step 5: Quality warnings ---
    quality_warnings = _generate_quality_warnings(
        df=df,
        target_column=target_column,
        task_type_value=task_type_value,
        high_null_columns=high_null_columns,
        high_cardinality_columns=high_cardinality_columns,
        target_distribution=target_distribution,
    )

    for warning in quality_warnings:
        console.print(f"  [yellow]Warning:[/yellow] {warning}")

    # --- Step 6: Save a lightweight profile summary to disk ---
    # Not the full HTML report (no ydata-profiling), but a JSON summary
    # that the dashboard can reference.
    profile_path = settings.profiles_dir / "data_profile_summary.json"
    import json
    summary = {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "target_column": target_column,
        "target_distribution": target_distribution,
        "high_null_columns": high_null_columns,
        "quality_warnings": quality_warnings,
        "top_correlations": top_correlations,
    }
    profile_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"  Profile summary saved to: {profile_path}")
    console.print("  [green]Profiling complete.[/green]")

    # --- Step 7: Build and return the DataProfile ---
    data_profile = DataProfile(
        n_rows=n_rows,
        n_columns=n_cols,
        columns=column_profiles,
        high_null_columns=high_null_columns,
        high_cardinality_columns=high_cardinality_columns,
        target_distribution=target_distribution,
        top_correlations=top_correlations,
        quality_warnings=quality_warnings,
        profile_report_path=str(profile_path),
    )

    return {"data_profile": data_profile}
