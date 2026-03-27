"""
nodes/etl.py
------------
Node 3: ETL Pipeline

Responsibility:
  Build a reproducible sklearn Pipeline, fit it on training data,
  serialise it to disk with joblib, and save the processed splits.

Why sklearn Pipeline?
  Chaining steps in a Pipeline ensures that all transformations are fitted
  only on training data and applied identically to test data. This prevents
  data leakage — a common mistake when preprocessing outside a pipeline.

Why joblib for serialisation?
  joblib handles numpy arrays and sklearn objects more efficiently than
  pickle. The fitted pipeline is loaded by the model node, dashboard node,
  and QA node without re-fitting.

Why save processed splits as CSV?
  Inspectability. Any node (or human reviewer) can open the CSVs and
  verify the split looks correct — no black box.

Inputs from state:
  - dataset_path: str
  - scoped_problem: ScopedProblem
  - data_profile: DataProfile

Outputs to state:
  - etl_artifacts: ETLArtifacts
"""

import json
import math

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from rich.console import Console

from orchestrator.config import settings
from orchestrator.state import (
    AnalyticsState,
    ColumnType,
    ETLArtifacts,
    FeatureSelectionStrategy,
    TaskType,
)

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_SIZE = 0.2       # 80/20 train/test split
RANDOM_STATE = 42     # fixed seed for reproducibility
DEFAULT_CORRELATION_THRESHOLD = 0.05
DEFAULT_TOP_K_CAP = 10
DEFAULT_TOP_K_FLOOR = 3


# ---------------------------------------------------------------------------
# Raw feature selection
# ---------------------------------------------------------------------------

def _select_features(
    df: pd.DataFrame,
    target_column: str,
    features_to_exclude: list[str],
    column_profiles,
) -> tuple[list[str], list[str]]:
    """
    Split columns into numeric and categorical feature lists.

    Excludes:
      - The target column
      - Columns explicitly listed in scoped_problem.features_to_exclude
      - Columns inferred as ID type in the data profile

    Returns:
        (numeric_features, categorical_features)
    """
    type_lookup = {cp.name: cp.column_type for cp in column_profiles}
    exclude = set(features_to_exclude) | {target_column}

    numeric_features = []
    categorical_features = []

    for col in df.columns:
        if col in exclude:
            continue
        col_type = type_lookup.get(col, ColumnType.NUMERIC)
        if col_type == ColumnType.ID:
            continue
        if col_type == ColumnType.CATEGORICAL:
            categorical_features.append(col)
        elif col_type in (ColumnType.NUMERIC, ColumnType.TARGET):
            if col != target_column:
                numeric_features.append(col)

    return numeric_features, categorical_features


def _resolve_select_k(n_numeric_features: int, requested_k: int | None) -> int:
    if n_numeric_features <= 0:
        return 0
    if requested_k is not None:
        return max(1, min(requested_k, n_numeric_features))
    heuristic = math.ceil(n_numeric_features * 0.5)
    heuristic = max(DEFAULT_TOP_K_FLOOR, heuristic)
    heuristic = min(DEFAULT_TOP_K_CAP, heuristic)
    return min(heuristic, n_numeric_features)


def _apply_select_k_best(
    X_train_numeric: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    task_type: TaskType,
    requested_k: int | None,
) -> list[str]:
    """Apply SelectKBest on raw numeric columns using training data only."""
    if not numeric_features:
        return []

    k = _resolve_select_k(len(numeric_features), requested_k)
    if k >= len(numeric_features):
        return list(numeric_features)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_train_numeric[numeric_features])

    score_func = f_regression if task_type == TaskType.REGRESSION else f_classif
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X_imputed, y_train)

    support = selector.get_support()
    selected = [col for col, keep in zip(numeric_features, support) if keep]
    return selected or list(numeric_features[:k])


def _coerce_target_for_correlation(y_train: pd.Series) -> pd.Series:
    """Convert a target series to numeric codes for simple correlation filtering."""
    y_series = pd.Series(y_train).copy()
    if pd.api.types.is_numeric_dtype(y_series):
        return y_series.astype(float)
    codes, _ = pd.factorize(y_series)
    return pd.Series(codes, index=y_series.index, dtype=float)


def _apply_correlation_filter(
    X_train_numeric: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    threshold: float | None,
) -> list[str]:
    """Keep numeric columns whose absolute train-time correlation crosses a threshold."""
    if not numeric_features:
        return []

    resolved_threshold = (
        DEFAULT_CORRELATION_THRESHOLD if threshold is None else float(threshold)
    )

    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(
        imputer.fit_transform(X_train_numeric[numeric_features]),
        columns=numeric_features,
        index=X_train_numeric.index,
    )
    y_numeric = _coerce_target_for_correlation(y_train)
    correlations = imputed.corrwith(y_numeric).abs().fillna(0.0)

    selected = [
        col for col in numeric_features
        if float(correlations.get(col, 0.0)) >= resolved_threshold
    ]
    if selected:
        return selected

    fallback_k = min(_resolve_select_k(len(numeric_features), None), len(numeric_features))
    ranked = correlations.sort_values(ascending=False).index.tolist()
    fallback = ranked[: max(1, fallback_k)]
    return [col for col in numeric_features if col in fallback]


def _apply_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    scoped_problem,
) -> tuple[list[str], list[str]]:
    strategy = scoped_problem.feature_selection_strategy

    if strategy == FeatureSelectionStrategy.NONE or not numeric_features:
        return list(numeric_features), []

    if strategy == FeatureSelectionStrategy.SELECT_K_BEST:
        selected = _apply_select_k_best(
            X_train_numeric=X_train[numeric_features],
            y_train=y_train,
            numeric_features=numeric_features,
            task_type=scoped_problem.task_type,
            requested_k=scoped_problem.feature_selection_k,
        )
    elif strategy == FeatureSelectionStrategy.CORRELATION_FILTER:
        selected = _apply_correlation_filter(
            X_train_numeric=X_train[numeric_features],
            y_train=y_train,
            numeric_features=numeric_features,
            threshold=scoped_problem.feature_selection_threshold,
        )
    else:
        selected = list(numeric_features)

    dropped = [col for col in numeric_features if col not in selected]
    return selected, dropped


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def _build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[Pipeline, list[str]]:
    """
    Build a sklearn ColumnTransformer pipeline.

    Numeric features:
      1. SimpleImputer(strategy='median') — robust to outliers
      2. StandardScaler — required for logistic regression / distance-based models

    Categorical features (if any):
      1. SimpleImputer(strategy='most_frequent')
      2. OneHotEncoder(handle_unknown='ignore') — safe for unseen categories

    Returns:
        (fitted_pipeline_object, preprocessing_steps_description)
    """
    transformers = []
    step_descriptions = []

    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_transformer, numeric_features))
        step_descriptions.append(
            f"SimpleImputer(strategy='median') + StandardScaler "
            f"on {len(numeric_features)} numeric columns"
        )

    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_transformer, categorical_features))
        step_descriptions.append(
            f"SimpleImputer(strategy='most_frequent') + OneHotEncoder "
            f"on {len(categorical_features)} categorical columns"
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    full_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    return full_pipeline, step_descriptions


# ---------------------------------------------------------------------------
# The node function
# ---------------------------------------------------------------------------

def etl_node(state: AnalyticsState) -> dict:
    """
    ETL pipeline node.

    Builds and fits a sklearn Pipeline on training data, saves processed
    splits to disk, and serialises the pipeline with joblib.

    Returns:
        dict with key "etl_artifacts" containing an ETLArtifacts instance.
    """
    console.rule("[bold]Node 3: ETL Pipeline[/bold]")

    scoped_problem = state["scoped_problem"]
    data_profile = state["data_profile"]
    target_column = scoped_problem.target_column

    # --- Step 1: Load dataset ---
    console.print(f"  Loading dataset from: {state['dataset_path']}")
    df = pd.read_csv(state["dataset_path"])
    n_rows_original = len(df)

    # --- Step 2: Drop exact duplicate rows ---
    df = df.drop_duplicates()
    n_rows_dropped = n_rows_original - len(df)
    if n_rows_dropped > 0:
        console.print(f"  Dropped {n_rows_dropped} duplicate rows")

    # --- Step 3: Select candidate raw features ---
    numeric_features, categorical_features = _select_features(
        df=df,
        target_column=target_column,
        features_to_exclude=scoped_problem.features_to_exclude,
        column_profiles=data_profile.columns,
    )

    console.print(f"  Numeric features before selection ({len(numeric_features)}): {numeric_features}")
    console.print(
        f"  Categorical features ({len(categorical_features)}): "
        f"{categorical_features or 'none'}"
    )

    X = df[numeric_features + categorical_features]
    y = df[target_column]

    # --- Step 4: Split into X and y ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if "classification" in scoped_problem.task_type.value else None,
    )

    console.print(f"  Train size: {len(X_train):,} rows | Test size: {len(X_test):,} rows")

    # --- Step 5: Apply deterministic raw numeric feature selection on train only ---
    selected_numeric_features, dropped_numeric_features = _apply_feature_selection(
        X_train=X_train,
        y_train=y_train,
        numeric_features=numeric_features,
        scoped_problem=scoped_problem,
    )
    feature_selection_strategy = scoped_problem.feature_selection_strategy
    all_features = selected_numeric_features + categorical_features

    selection_steps: list[str] = []
    if feature_selection_strategy == FeatureSelectionStrategy.NONE:
        selection_steps.append("Feature selection: none")
    elif feature_selection_strategy == FeatureSelectionStrategy.SELECT_K_BEST:
        selection_steps.append(
            f"Feature selection: SelectKBest kept {len(selected_numeric_features)} "
            f"of {len(numeric_features)} numeric columns"
        )
    elif feature_selection_strategy == FeatureSelectionStrategy.CORRELATION_FILTER:
        threshold = scoped_problem.feature_selection_threshold
        threshold_text = (
            f"threshold={threshold}" if threshold is not None else f"threshold={DEFAULT_CORRELATION_THRESHOLD}"
        )
        selection_steps.append(
            f"Feature selection: correlation filter kept {len(selected_numeric_features)} "
            f"of {len(numeric_features)} numeric columns ({threshold_text})"
        )

    console.print(
        f"  Numeric features after selection ({len(selected_numeric_features)}): "
        f"{selected_numeric_features or 'none'}"
    )
    if dropped_numeric_features:
        console.print(f"  Dropped numeric features: {dropped_numeric_features}")

    # --- Step 6: Build and fit the pipeline (on training data only) ---
    pipeline, preprocessing_steps = _build_pipeline(
        selected_numeric_features,
        categorical_features,
    )
    all_steps = selection_steps + preprocessing_steps

    X_train_processed = pipeline.fit_transform(X_train[all_features])
    X_test_processed = pipeline.transform(X_test[all_features])

    console.print(f"  Pipeline fitted. Output shape: {X_train_processed.shape}")
    for step in all_steps:
        console.print(f"  Step: {step}")

    # --- Step 7: Save processed splits to disk ---
    processed_dir = settings.models_dir

    try:
        feature_names_out = pipeline.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        feature_names_out = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

    train_df = pd.DataFrame(X_train_processed, columns=feature_names_out)
    train_df["__target__"] = y_train.values
    train_df["__split__"] = "train"

    test_df = pd.DataFrame(X_test_processed, columns=feature_names_out)
    test_df["__target__"] = y_test.values
    test_df["__split__"] = "test"

    processed_data_path = processed_dir / "processed_splits.csv"
    pd.concat([train_df, test_df], ignore_index=True).to_csv(
        processed_data_path,
        index=False,
    )

    # --- Step 8: Serialise the fitted pipeline ---
    pipeline_path = processed_dir / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    console.print(f"  Pipeline saved to: {pipeline_path}")

    # --- Step 9: Save split metadata for downstream nodes ---
    split_meta = {
        "train_indices": list(X_train.index.astype(int)),
        "test_indices": list(X_test.index.astype(int)),
        "feature_columns": all_features,
        "target_column": target_column,
        "feature_selection_strategy": feature_selection_strategy.value,
        "selected_numeric_features": selected_numeric_features,
        "dropped_numeric_features": dropped_numeric_features,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    split_meta_path = processed_dir / "split_metadata.json"
    split_meta_path.write_text(json.dumps(split_meta, indent=2), encoding="utf-8")

    console.print("  [green]ETL complete.[/green]")

    return {
        "etl_artifacts": ETLArtifacts(
            feature_columns=all_features,
            target_column=target_column,
            feature_selection_strategy=feature_selection_strategy,
            selected_numeric_features=selected_numeric_features,
            dropped_numeric_features=dropped_numeric_features,
            preprocessing_steps=all_steps,
            n_rows_after_cleaning=len(df),
            n_rows_dropped=n_rows_dropped,
            pipeline_path=str(pipeline_path),
            processed_data_path=str(processed_data_path),
        )
    }
