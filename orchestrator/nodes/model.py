"""
nodes/model.py
--------------
Node 4: Baseline Model

Responsibility:
  Load the processed splits, select and train the appropriate baseline
  algorithm, run cross-validation, extract feature importances, and
  serialise the fitted model with joblib.

Why 'baseline'?
  This system makes no claim to produce the best possible model.
  The goal is a reproducible, explainable starting point that the
  review node can evaluate honestly. The word 'baseline' is in the
  node name so that expectation is set at every level.

Algorithm selection logic:
  binary_classification      -> LogisticRegression(class_weight='balanced')
  multiclass_classification  -> LogisticRegression(class_weight='balanced',
                                                   multi_class='multinomial')
  regression                 -> Ridge (regularised linear regression)

Why these choices?
  - Interpretable: coefficients map directly to feature importances
  - Fast: suitable for 150k rows without GPU
  - Honest: a linear baseline is a fair starting point before tree models
  - Handles imbalance: class_weight='balanced' rescales loss per class

Inputs from state:
  - scoped_problem: ScopedProblem
  - etl_artifacts: ETLArtifacts

Outputs to state:
  - model_results: ModelResults
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline

from orchestrator.config import settings
from orchestrator.state import (
    AnalyticsState,
    ModelResults,
    TaskType,
)

console = Console()

RANDOM_STATE = 42
N_CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Algorithm factory
# ---------------------------------------------------------------------------

def _get_algorithm(task_type: TaskType):
    """
    Return the appropriate sklearn estimator for the task type.

    Why not just use RandomForest?
    A linear model is more interpretable, faster, and gives a cleaner
    baseline. If the linear model performs well, the problem is likely
    linearly separable and doesn't need a complex model. If it performs
    poorly, that's useful signal — not a failure of the system.
    """
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return LogisticRegression(
            class_weight="balanced",  # handles 93/7 imbalance
            max_iter=1000,            # sufficient for convergence on scaled data
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return LogisticRegression(
            class_weight="balanced",
            multi_class="multinomial",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
    elif task_type == TaskType.REGRESSION:
        return Ridge(
            alpha=1.0,               # L2 regularisation — robust to correlated features
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _get_cv_scorer(task_type: TaskType) -> str:
    """
    Return the sklearn scorer string for cross_val_score.

    These match scoped_problem.success_metric so the QA node can
    assert consistency between the scope and the model evaluation.
    """
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return "roc_auc"
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return "f1_weighted"
    elif task_type == TaskType.REGRESSION:
        return "neg_root_mean_squared_error"
    else:
        raise ValueError(f"No scorer for task type: {task_type}")


def _get_cv_splitter(task_type: TaskType, n_splits: int):
    """
    Return the appropriate CV splitter.

    StratifiedKFold for classification — preserves class proportions
    in each fold, critical for imbalanced datasets.

    KFold for regression — no class to stratify on.
    """
    if "classification" in task_type.value:
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------

def _extract_feature_importances(
    model,
    feature_names: list[str],
    task_type: TaskType,
    top_n: int = 10,
) -> list[dict]:
    """
    Extract feature importances from the fitted model.

    For linear models (Logistic, Ridge):
      Use absolute coefficient values. For binary classification,
      coef_ has shape (1, n_features) — we flatten it.

    For tree models (future extension):
      Use feature_importances_ directly.

    Returns list of dicts: [{"feature": str, "importance": float}, ...]
    sorted descending, limited to top_n.
    """
    try:
        if hasattr(model, "coef_"):
            coef = np.abs(model.coef_)
            if coef.ndim == 2:
                # Binary classification: shape (1, n_features)
                # Multiclass: shape (n_classes, n_features) — take mean
                coef = coef.mean(axis=0)
            importances = coef
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            return []

        # Normalise to sum to 1 for comparability
        total = importances.sum()
        if total > 0:
            importances = importances / total

        # Pair with feature names and sort
        paired = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {"feature": feat, "importance": round(float(imp), 4)}
            for feat, imp in paired[:top_n]
        ]

    except Exception as e:
        console.print(f"  [yellow]Warning: could not extract importances: {e}[/yellow]")
        return []


# ---------------------------------------------------------------------------
# The node function
# ---------------------------------------------------------------------------

def model_node(state: AnalyticsState) -> dict:
    """
    Baseline model node.

    Loads processed splits, trains the appropriate baseline algorithm,
    runs stratified cross-validation, extracts feature importances,
    and serialises the model with joblib.

    Returns:
        dict with key "model_results" containing a ModelResults instance.
    """
    console.rule("[bold]Node 4: Baseline Model[/bold]")

    scoped_problem = state["scoped_problem"]
    etl_artifacts = state["etl_artifacts"]
    task_type = scoped_problem.task_type

    # --- Step 1: Load processed splits ---
    console.print(f"  Loading processed splits from: "
                  f"{etl_artifacts.processed_data_path}")

    splits_df = pd.read_csv(etl_artifacts.processed_data_path)
    train_df = splits_df[splits_df["__split__"] == "train"]
    test_df = splits_df[splits_df["__split__"] == "test"]

    # Drop control columns before feeding to model
    drop_cols = ["__split__", "__target__"]
    feature_cols = [c for c in splits_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df["__target__"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["__target__"].values

    console.print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # --- Step 2: Select algorithm and scorer ---
    algorithm = _get_algorithm(task_type)
    scorer = _get_cv_scorer(task_type)
    cv_splitter = _get_cv_splitter(task_type, N_CV_FOLDS)

    console.print(f"  Algorithm : {algorithm.__class__.__name__}")
    console.print(f"  Scorer    : {scorer}")

    # --- Step 3: Cross-validation on training data ---
    # cross_val_score fits and scores on each fold internally.
    # It never touches X_test — that is reserved for the final evaluation.
    console.print(f"  Running {N_CV_FOLDS}-fold cross-validation...")

    cv_scores = cross_val_score(
        algorithm,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring=scorer,
        n_jobs=-1,   # use all available CPU cores
    )

    # For neg_rmse, flip sign back to positive
    if scorer == "neg_root_mean_squared_error":
        cv_scores = np.abs(cv_scores)

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    console.print(f"  CV {scorer}: {cv_mean:.4f} ± {cv_std:.4f}")

    # --- Step 4: Fit on full training set ---
    algorithm.fit(X_train, y_train)

    # --- Step 5: Evaluate on held-out test set ---
    if task_type == TaskType.BINARY_CLASSIFICATION:
        y_proba = algorithm.predict_proba(X_test)[:, 1]
        test_score = float(roc_auc_score(y_test, y_proba))
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        y_pred = algorithm.predict(X_test)
        test_score = float(f1_score(y_test, y_pred, average="weighted"))
    else:
        y_pred = algorithm.predict(X_test)
        test_score = float(
            np.sqrt(mean_squared_error(y_test, y_pred))
        )

    console.print(f"  Test {scorer}: {test_score:.4f}")

    # --- Step 6: Extract feature importances ---
    # feature_cols contains post-encoding column names (e.g. numeric__age)
    # We use them as-is — the dashboard will display them directly
    feature_importances = _extract_feature_importances(
        model=algorithm,
        feature_names=feature_cols,
        task_type=task_type,
    )

    if feature_importances:
        top = feature_importances[0]
        console.print(f"  Top feature: {top['feature']} "
                      f"(importance={top['importance']:.4f})")

    # --- Step 7: Build training notes ---
    training_notes = []
    if task_type in (TaskType.BINARY_CLASSIFICATION,
                     TaskType.MULTICLASS_CLASSIFICATION):
        training_notes.append(
            "class_weight='balanced' applied — "
            "loss rescaled inversely proportional to class frequency."
        )
    if cv_std > 0.05:
        training_notes.append(
            f"High CV variance (std={cv_std:.3f}) — "
            "model performance is unstable across folds."
        )

    # --- Step 8: Serialise the model ---
    model_path = settings.models_dir / "model.joblib"
    joblib.dump(algorithm, model_path)
    console.print(f"  Model saved to: {model_path}")
    console.print("  [green]Model training complete.[/green]")

    return {
        "model_results": ModelResults(
            algorithm=algorithm.__class__.__name__,
            task_type=task_type,
            cv_scores=cv_scores.tolist(),
            cv_mean=round(cv_mean, 4),
            cv_std=round(cv_std, 4),
            test_score=round(test_score, 4),
            primary_metric=scorer.replace("neg_", ""),
            feature_importances=feature_importances,
            model_path=str(model_path),
            training_notes=training_notes,
        )
    }
