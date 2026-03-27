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
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
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
)

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_SIZE = 0.2       # 80/20 train/test split
RANDOM_STATE = 42     # fixed seed for reproducibility


# ---------------------------------------------------------------------------
# Feature selection
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
    # Build a lookup: column name -> ColumnType from the profile
    type_lookup = {cp.name: cp.column_type for cp in column_profiles}

    exclude = set(features_to_exclude) | {target_column}

    numeric_features = []
    categorical_features = []

    for col in df.columns:
        if col in exclude:
            continue
        col_type = type_lookup.get(col, ColumnType.NUMERIC)
        if col_type == ColumnType.ID:
            continue   # skip ID columns even if not in features_to_exclude
        elif col_type == ColumnType.CATEGORICAL:
            categorical_features.append(col)
        elif col_type in (ColumnType.NUMERIC, ColumnType.TARGET):
            # TARGET type shouldn't appear in features, but guard anyway
            if col != target_column:
                numeric_features.append(col)
        # TEXT and DATETIME columns are skipped — they need custom handling
        # beyond the scope of a baseline pipeline

    return numeric_features, categorical_features


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

    Why median imputation?
    Financial datasets often have outliers (extreme incomes, debt ratios).
    Median is robust to these outliers — mean imputation would skew the
    distribution in the direction of outliers.

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
        remainder="drop",   # drop any columns not explicitly handled
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

    # --- Step 3: Select features ---
    numeric_features, categorical_features = _select_features(
        df=df,
        target_column=target_column,
        features_to_exclude=scoped_problem.features_to_exclude,
        column_profiles=data_profile.columns,
    )
    all_features = numeric_features + categorical_features

    console.print(f"  Numeric features  ({len(numeric_features)}): {numeric_features}")
    console.print(f"  Categorical features ({len(categorical_features)}): "
                  f"{categorical_features or 'none'}")

    # --- Step 4: Split into X and y ---
    X = df[all_features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        # Stratify for classification to preserve class balance in both splits
        stratify=y if "classification" in scoped_problem.task_type.value else None,
    )

    console.print(f"  Train size: {len(X_train):,} rows | "
                  f"Test size: {len(X_test):,} rows")

    # --- Step 5: Build and fit the pipeline (on training data only) ---
    pipeline, step_descriptions = _build_pipeline(
        numeric_features, categorical_features
    )

    # fit_transform on training data — this is the only call that sees y_train
    # indirectly (through stratified split, not through the pipeline itself)
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)   # transform only — no fit

    console.print(f"  Pipeline fitted. Output shape: {X_train_processed.shape}")
    for step in step_descriptions:
        console.print(f"  Step: {step}")

    # --- Step 6: Save processed splits to disk ---
    # Store as DataFrames with the original feature names for readability.
    # Post-encoding column names are constructed from the pipeline.
    processed_dir = settings.models_dir

    # Get output column names after encoding
    try:
        feature_names_out = pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    except Exception:
        feature_names_out = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

    train_df = pd.DataFrame(X_train_processed, columns=feature_names_out)
    train_df["__target__"] = y_train.values
    test_df = pd.DataFrame(X_test_processed, columns=feature_names_out)
    test_df["__target__"] = y_test.values

    processed_data_path = processed_dir / "processed_splits.csv"
    # Save train and test stacked with a split indicator
    train_df["__split__"] = "train"
    test_df["__split__"] = "test"
    pd.concat([train_df, test_df], ignore_index=True).to_csv(
        processed_data_path, index=False
    )

    # --- Step 7: Serialise the fitted pipeline ---
    pipeline_path = processed_dir / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    console.print(f"  Pipeline saved to: {pipeline_path}")

    # --- Step 8: Save split metadata for downstream nodes ---
    # The model node will re-load the splits from processed_splits.csv,
    # so we also save the index mappings for train/test.
    split_meta = {
        "train_indices": list(X_train.index.astype(int)),
        "test_indices": list(X_test.index.astype(int)),
        "feature_columns": all_features,
        "target_column": target_column,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    split_meta_path = processed_dir / "split_metadata.json"
    split_meta_path.write_text(
        json.dumps(split_meta, indent=2), encoding="utf-8"
    )

    console.print("  [green]ETL complete.[/green]")

    return {
        "etl_artifacts": ETLArtifacts(
            feature_columns=all_features,
            target_column=target_column,
            preprocessing_steps=step_descriptions,
            n_rows_after_cleaning=len(df),
            n_rows_dropped=n_rows_dropped,
            pipeline_path=str(pipeline_path),
            processed_data_path=str(processed_data_path),
        )
    }
