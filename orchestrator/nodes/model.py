"""
nodes/model.py
--------------
Node 4: Baseline Model

Trains the baseline model, runs cross-validation, computes feature
importances via coefficient magnitude AND SHAP values, and serialises
everything with joblib.

Why SHAP?
  Coefficients from LogisticRegression give a rough importance ranking
  but they depend on feature scale. SHAP (SHapley Additive exPlanations)
  gives theoretically grounded, model-agnostic importance values that
  account for feature interactions. LinearExplainer is fast and exact
  for linear models — no sampling required.

Feature name cleaning:
  sklearn's ColumnTransformer prefixes output columns with the
  transformer name (e.g. "numeric__age"). We strip this prefix so
  the dashboard and reports show clean column names.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
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
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
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
        return Ridge(alpha=1.0, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def _get_cv_scorer(task_type: TaskType) -> str:
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return "roc_auc"
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return "f1_weighted"
    elif task_type == TaskType.REGRESSION:
        return "neg_root_mean_squared_error"
    else:
        raise ValueError(f"No scorer for: {task_type}")


def _get_cv_splitter(task_type: TaskType, n_splits: int):
    if "classification" in task_type.value:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Feature name cleaning
# ---------------------------------------------------------------------------

def _clean_feature_names(feature_names: list[str]) -> list[str]:
    """
    Strip sklearn ColumnTransformer prefixes from feature names.

    sklearn's ColumnTransformer produces names like:
      "numeric__age"         -> "age"
      "categorical__gender"  -> "gender"

    We strip everything up to and including the first "__" so the
    dashboard and reports show the original column names.
    """
    cleaned = []
    for name in feature_names:
        if "__" in name:
            cleaned.append(name.split("__", 1)[1])
        else:
            cleaned.append(name)
    return cleaned


# ---------------------------------------------------------------------------
# Feature importances (coefficient-based)
# ---------------------------------------------------------------------------

def _extract_feature_importances(
    model,
    feature_names: list[str],
    task_type: TaskType,
    top_n: int = 10,
) -> list[dict]:
    """Extract normalised absolute coefficient importances."""
    try:
        if hasattr(model, "coef_"):
            coef = np.abs(model.coef_)
            if coef.ndim == 2:
                coef = coef.mean(axis=0)
            importances = coef
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            return []

        total = importances.sum()
        if total > 0:
            importances = importances / total

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
# SHAP values
# ---------------------------------------------------------------------------

def _compute_shap_values(
    model,
    X_train: np.ndarray,
    feature_names: list[str],
    task_type: TaskType,
    top_n: int = 10,
) -> list[dict]:
    """
    Compute mean absolute SHAP values using LinearExplainer.

    Why LinearExplainer?
      For linear models (LogisticRegression, Ridge), LinearExplainer
      computes exact SHAP values analytically — no sampling, no
      approximation. It is fast even on 120k rows because it uses
      the training data mean as the background.

    Why SHAP over coefficients?
      Coefficients are scale-dependent and assume feature independence.
      SHAP values are scale-independent and account for the correlation
      structure of the training data. They are also directly interpretable:
      a SHAP value of 0.05 means that feature increased the prediction
      by 0.05 on average.

    Returns top_n features sorted by mean absolute SHAP value descending.
    """
    try:
        import shap

        # Use a sample for speed — 5k rows is sufficient for stable mean SHAP
        sample_size = min(5000, X_train.shape[0])
        rng = np.random.default_rng(seed=RANDOM_STATE)
        idx = rng.choice(X_train.shape[0], size=sample_size, replace=False)
        X_sample = X_train[idx]

        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values is a 2D array (n_samples, n_features)
        if isinstance(shap_values, list):
            # Multiclass: list of arrays, one per class — take mean across classes
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)

        mean_shap = shap_values.mean(axis=0)

        # Normalise
        total = mean_shap.sum()
        if total > 0:
            mean_shap = mean_shap / total

        paired = sorted(
            zip(feature_names, mean_shap),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {"feature": feat, "shap_importance": round(float(val), 4)}
            for feat, val in paired[:top_n]
        ]

    except Exception as e:
        console.print(f"  [yellow]SHAP computation failed (non-fatal): {e}[/yellow]")
        return []


# ---------------------------------------------------------------------------
# The node function
# ---------------------------------------------------------------------------

def model_node(state: AnalyticsState) -> dict:
    """
    Baseline model node.

    Loads processed splits, trains the algorithm, runs CV,
    computes both coefficient importances and SHAP values,
    serialises the model.
    """
    console.rule("[bold]Node 4: Baseline Model[/bold]")

    scoped_problem = state["scoped_problem"]
    etl_artifacts = state["etl_artifacts"]
    task_type = scoped_problem.task_type

    # --- Load splits ---
    splits_df = pd.read_csv(etl_artifacts.processed_data_path)
    train_df = splits_df[splits_df["__split__"] == "train"]
    test_df = splits_df[splits_df["__split__"] == "test"]

    drop_cols = ["__split__", "__target__"]
    raw_feature_cols = [c for c in splits_df.columns if c not in drop_cols]

    # Clean feature names — strip "numeric__" / "categorical__" prefixes
    clean_feature_cols = _clean_feature_names(raw_feature_cols)

    X_train = train_df[raw_feature_cols].values
    y_train = train_df["__target__"].values
    X_test = test_df[raw_feature_cols].values
    y_test = test_df["__target__"].values

    console.print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # --- Algorithm and scorer ---
    algorithm = _get_algorithm(task_type)
    scorer = _get_cv_scorer(task_type)
    cv_splitter = _get_cv_splitter(task_type, N_CV_FOLDS)

    console.print(f"  Algorithm : {algorithm.__class__.__name__}")
    console.print(f"  Scorer    : {scorer}")

    # --- Cross-validation ---
    console.print(f"  Running {N_CV_FOLDS}-fold cross-validation...")
    cv_scores = cross_val_score(
        algorithm, X_train, y_train,
        cv=cv_splitter, scoring=scorer, n_jobs=-1,
    )
    if scorer == "neg_root_mean_squared_error":
        cv_scores = np.abs(cv_scores)

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    console.print(f"  CV {scorer}: {cv_mean:.4f} ± {cv_std:.4f}")

    # --- Fit on full training set ---
    algorithm.fit(X_train, y_train)

    # --- Test set evaluation ---
    if task_type == TaskType.BINARY_CLASSIFICATION:
        y_proba = algorithm.predict_proba(X_test)[:, 1]
        test_score = float(roc_auc_score(y_test, y_proba))
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        y_pred = algorithm.predict(X_test)
        test_score = float(f1_score(y_test, y_pred, average="weighted"))
    else:
        y_pred = algorithm.predict(X_test)
        test_score = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    console.print(f"  Test {scorer}: {test_score:.4f}")

    # --- Coefficient importances (using clean names) ---
    feature_importances = _extract_feature_importances(
        model=algorithm,
        feature_names=clean_feature_cols,
        task_type=task_type,
    )
    if feature_importances:
        console.print(
            f"  Top feature: {feature_importances[0]['feature']} "
            f"(importance={feature_importances[0]['importance']:.4f})"
        )

    # --- SHAP values ---
    console.print("  Computing SHAP values...")
    shap_importances = _compute_shap_values(
        model=algorithm,
        X_train=X_train,
        feature_names=clean_feature_cols,
        task_type=task_type,
    )
    if shap_importances:
        console.print(
            f"  Top SHAP feature: {shap_importances[0]['feature']} "
            f"(shap={shap_importances[0]['shap_importance']:.4f})"
        )

    # --- Training notes ---
    training_notes = []
    if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        training_notes.append(
            "class_weight='balanced' applied — loss rescaled by class frequency."
        )
    if cv_std > 0.05:
        training_notes.append(
            f"High CV variance (std={cv_std:.3f}) — performance unstable across folds."
        )

    # Save SHAP values alongside model for dashboard use
    shap_path = settings.models_dir / "shap_values.json"
    shap_path.write_text(
        json.dumps(shap_importances, indent=2), encoding="utf-8"
    )

    # --- Serialise model ---
    model_path = settings.models_dir / "model.joblib"
    joblib.dump(algorithm, model_path)
    console.print(f"  Model saved to: {model_path}")
    console.print("  [green]Model training complete.[/green]")

    # Store SHAP importances in feature_importances field
    # (we use SHAP as the primary importance metric going forward)
    final_importances = shap_importances if shap_importances else feature_importances
    # Convert shap_importance key to importance key for schema compatibility
    final_importances = [
        {
            "feature": item.get("feature"),
            "importance": item.get("shap_importance", item.get("importance", 0.0)),
        }
        for item in final_importances
    ]

    return {
        "model_results": ModelResults(
            algorithm=algorithm.__class__.__name__,
            task_type=task_type,
            cv_scores=cv_scores.tolist(),
            cv_mean=round(cv_mean, 4),
            cv_std=round(cv_std, 4),
            test_score=round(test_score, 4),
            primary_metric=scorer.replace("neg_", ""),
            feature_importances=final_importances,
            model_path=str(model_path),
            training_notes=training_notes,
        )
    }
