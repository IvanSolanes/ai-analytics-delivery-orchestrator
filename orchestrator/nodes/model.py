"""
nodes/model.py
--------------
Node 4: Baseline Model

Trains the algorithm recommended by the scoping node, runs
cross-validation, computes SHAP values, and serialises everything.

Supported algorithms (driven by scoped_problem.model_recommendation):
  logistic_regression  -> LogisticRegression (linear, interpretable)
  random_forest        -> RandomForestClassifier / RandomForestRegressor
  gradient_boosting    -> GradientBoostingClassifier / GradientBoostingRegressor
  ridge                -> Ridge (regression only)

Why keep the LLM recommendation but execute deterministically?
  The LLM picks the strategy from a controlled vocabulary based on the
  brief and data profile. Deterministic code executes it. This gives
  the system genuine adaptability without LLM hallucination risk in
  the actual training code.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from orchestrator.config import settings
from orchestrator.state import (
    AnalyticsState,
    ModelRecommendation,
    ModelResults,
    TaskType,
)

console = Console()

RANDOM_STATE = 42
N_CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Algorithm factory — expanded
# ---------------------------------------------------------------------------

def _get_algorithm(task_type: TaskType, recommendation: ModelRecommendation):
    """
    Return the appropriate sklearn estimator based on task type and
    the LLM's model recommendation.

    The recommendation comes from scoped_problem.model_recommendation —
    the scoping LLM picked it from a controlled enum. We map it to a
    real sklearn estimator here. The LLM never writes ML code.

    Class imbalance handling:
      LogisticRegression and RandomForest support class_weight='balanced'.
      GradientBoosting does not — it handles imbalance via its loss function.
    """
    is_classification = "classification" in task_type.value
    is_binary = task_type == TaskType.BINARY_CLASSIFICATION

    if recommendation == ModelRecommendation.LOGISTIC_REGRESSION:
        if not is_classification:
            console.print(
                "  [yellow]Note: logistic_regression recommended for regression "
                "task — falling back to Ridge.[/yellow]"
            )
            return Ridge(alpha=1.0, random_state=RANDOM_STATE)
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )

    elif recommendation == ModelRecommendation.RANDOM_FOREST:
        if is_classification:
            return RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                max_depth=8,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif recommendation == ModelRecommendation.GRADIENT_BOOSTING:
        if is_classification:
            return GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=RANDOM_STATE,
            )
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )

    elif recommendation == ModelRecommendation.RIDGE:
        if is_classification:
            console.print(
                "  [yellow]Note: ridge recommended for classification task "
                "— falling back to LogisticRegression.[/yellow]"
            )
            return LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )
        return Ridge(alpha=1.0, random_state=RANDOM_STATE)

    else:
        raise ValueError(f"Unknown model recommendation: {recommendation}")


def _get_cv_scorer(task_type: TaskType) -> str:
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return "roc_auc"
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return "f1_weighted"
    elif task_type == TaskType.REGRESSION:
        return "neg_root_mean_squared_error"
    raise ValueError(f"No scorer for: {task_type}")


def _get_cv_splitter(task_type: TaskType, n_splits: int):
    if "classification" in task_type.value:
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Feature name cleaning
# ---------------------------------------------------------------------------

def _clean_feature_names(feature_names: list[str]) -> list[str]:
    """Strip sklearn ColumnTransformer prefixes (numeric__, categorical__)."""
    return [
        name.split("__", 1)[1] if "__" in name else name
        for name in feature_names
    ]


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------

def _extract_feature_importances(
    model, feature_names: list[str], task_type: TaskType, top_n: int = 10
) -> list[dict]:
    """
    Extract normalised feature importances.

    Linear models (Logistic, Ridge): use absolute coefficients.
    Tree models (RandomForest, GradientBoosting): use feature_importances_.
    Both are normalised to sum to 1 for comparability across model types.
    """
    try:
        if hasattr(model, "feature_importances_"):
            # Tree models — Gini/variance-based importances
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            # Linear models — absolute coefficients
            coef = np.abs(model.coef_)
            if coef.ndim == 2:
                coef = coef.mean(axis=0)
            importances = coef
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
    model, X_train: np.ndarray, feature_names: list[str],
    task_type: TaskType, top_n: int = 10,
) -> list[dict]:
    """
    Compute mean absolute SHAP values.

    Uses the appropriate explainer for each model type:
      Linear models  -> LinearExplainer (exact, fast)
      Tree models    -> TreeExplainer (exact for trees, fast)
    """
    try:
        import shap

        sample_size = min(5000, X_train.shape[0])
        rng = np.random.default_rng(seed=RANDOM_STATE)
        idx = rng.choice(X_train.shape[0], size=sample_size, replace=False)
        X_sample = X_train[idx]

        # Pick the right explainer for the model type
        if hasattr(model, "feature_importances_"):
            # Tree-based models — TreeExplainer is exact and fast
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Linear models — LinearExplainer is exact
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)

        # Handle multiclass / binary outputs
        if isinstance(shap_values, list):
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)

        mean_shap = shap_values.mean(axis=0)
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

    Reads the model recommendation from scoped_problem, trains the
    appropriate algorithm, runs cross-validation, computes SHAP values,
    and serialises the model.
    """
    console.rule("[bold]Node 4: Baseline Model[/bold]")

    scoped_problem = state["scoped_problem"]
    etl_artifacts = state["etl_artifacts"]
    task_type = scoped_problem.task_type
    recommendation = scoped_problem.model_recommendation

    # --- Load splits ---
    splits_df = pd.read_csv(etl_artifacts.processed_data_path)
    train_df = splits_df[splits_df["__split__"] == "train"]
    test_df = splits_df[splits_df["__split__"] == "test"]

    drop_cols = ["__split__", "__target__"]
    raw_feature_cols = [c for c in splits_df.columns if c not in drop_cols]
    clean_feature_cols = _clean_feature_names(raw_feature_cols)

    X_train = train_df[raw_feature_cols].values
    y_train = train_df["__target__"].values
    X_test = test_df[raw_feature_cols].values
    y_test = test_df["__target__"].values

    console.print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # --- Algorithm selection ---
    algorithm = _get_algorithm(task_type, recommendation)
    scorer = _get_cv_scorer(task_type)
    cv_splitter = _get_cv_splitter(task_type, N_CV_FOLDS)

    console.print(f"  Algorithm   : {algorithm.__class__.__name__}")
    console.print(f"  Recommended : {recommendation.value}")
    console.print(f"  Scorer      : {scorer}")

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

    # --- Coefficient/tree importances ---
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
    training_notes = [f"Model selected by scoping LLM: {recommendation.value}"]
    if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        if recommendation in (
            ModelRecommendation.LOGISTIC_REGRESSION,
            ModelRecommendation.RANDOM_FOREST,
        ):
            training_notes.append(
                "class_weight='balanced' applied — handles class imbalance."
            )
    if cv_std > 0.05:
        training_notes.append(
            f"High CV variance (std={cv_std:.3f}) — performance unstable."
        )

    # Save SHAP values to disk
    shap_path = settings.models_dir / "shap_values.json"
    shap_path.write_text(
        json.dumps(shap_importances, indent=2), encoding="utf-8"
    )

    # Serialise model
    model_path = settings.models_dir / "model.joblib"
    joblib.dump(algorithm, model_path)
    console.print(f"  Model saved to: {model_path}")
    console.print("  [green]Model training complete.[/green]")

    # Use SHAP importances as primary, fall back to coefficient importances
    final_importances = shap_importances if shap_importances else feature_importances
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
