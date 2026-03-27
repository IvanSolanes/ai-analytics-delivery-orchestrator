"""
feedback.py
-----------
Deterministic parsing of human review feedback into actionable state changes.

Why a rule-based parser first?
  - Review comments like "try fewer features" or "use random forest" are
    high-value and repetitive.
  - We want predictable behaviour with no extra LLM call.
  - The output is a typed FeedbackAction, so the graph can resume from the
    earliest node that actually needs to rerun.
"""

from __future__ import annotations

import re
from typing import Optional

from orchestrator.state import (
    AnalyticsState,
    FeatureSelectionStrategy,
    FeedbackAction,
    ModelRecommendation,
    ResumeFrom,
)

_STAGE_ORDER = {
    ResumeFrom.SCOPING: 0,
    ResumeFrom.ETL: 1,
    ResumeFrom.MODEL: 2,
}

_METRIC_ALIASES = {
    "roc_auc": ["roc_auc", "roc auc", "auc"],
    "f1_weighted": ["f1_weighted", "weighted f1", "f1 weighted"],
    "f1": ["f1", "f-1"],
    "accuracy": ["accuracy", "accurate"],
    "rmse": ["rmse", "root mean squared error"],
    "mae": ["mae", "mean absolute error"],
}


def _choose_earliest_resume(
    current: Optional[ResumeFrom],
    candidate: ResumeFrom,
) -> ResumeFrom:
    if current is None:
        return candidate
    return candidate if _STAGE_ORDER[candidate] < _STAGE_ORDER[current] else current


def _extract_first_int(text: str) -> Optional[int]:
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None


def _extract_explicit_columns(feedback: str, candidate_columns: list[str]) -> list[str]:
    """Match known column names in feedback using case-insensitive exact-ish search."""
    text = feedback.lower()
    matched: list[str] = []

    for column in sorted(candidate_columns, key=len, reverse=True):
        escaped = re.escape(column.lower())
        pattern = rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])"
        if re.search(pattern, text):
            matched.append(column)

    # preserve dataset order while removing duplicates
    seen = set()
    ordered = []
    for column in candidate_columns:
        if column in matched and column not in seen:
            ordered.append(column)
            seen.add(column)
    return ordered


def parse_human_feedback(feedback: str, state: AnalyticsState) -> FeedbackAction:
    """
    Parse free-text human feedback into a deterministic action plan.

    Supported v1 actions:
      - "try fewer features" / "reduce features"
      - correlation-based feature filtering
      - exclude explicit columns mentioned by name
      - switch model algorithm
      - change primary metric
      - fallback to re-scope when feedback is broad or ambiguous
    """
    text = (feedback or "").strip().lower()
    scoped_problem = state.get("scoped_problem")
    data_profile = state.get("data_profile")

    candidate_columns = []
    if data_profile is not None:
        candidate_columns = [cp.name for cp in data_profile.columns]

    resume_from: Optional[ResumeFrom] = None
    rationale_parts: list[str] = []
    matched_any = False

    action = FeedbackAction()

    def mark(stage: ResumeFrom, reason: str) -> None:
        nonlocal matched_any, resume_from
        matched_any = True
        resume_from = _choose_earliest_resume(resume_from, stage)
        rationale_parts.append(reason)

    # ------------------------------------------------------------------
    # Feature-selection feedback
    # ------------------------------------------------------------------
    if any(phrase in text for phrase in [
        "fewer features",
        "reduce features",
        "too many features",
        "simpler feature set",
        "try fewer predictors",
        "use fewer columns",
        "reduce the number of features",
    ]):
        action.set_feature_selection_strategy = FeatureSelectionStrategy.SELECT_K_BEST
        inferred_k = _extract_first_int(text)
        if inferred_k is not None:
            action.set_feature_selection_k = inferred_k
            mark(ResumeFrom.ETL, f"feature reduction requested (k={inferred_k})")
        else:
            mark(ResumeFrom.ETL, "feature reduction requested")

    if any(phrase in text for phrase in [
        "correlation filter",
        "correlation-based",
        "correlation based",
        "filter by correlation",
    ]):
        action.set_feature_selection_strategy = FeatureSelectionStrategy.CORRELATION_FILTER
        maybe_threshold = re.search(r"(?:threshold|corr(?:elation)?)\s*[=:]?\s*(0?\.\d+|1\.0|1)", text)
        if maybe_threshold:
            action.set_feature_selection_threshold = float(maybe_threshold.group(1))
            mark(ResumeFrom.ETL, f"correlation filter requested (threshold={maybe_threshold.group(1)})")
        else:
            mark(ResumeFrom.ETL, "correlation filter requested")

    # ------------------------------------------------------------------
    # Explicit column exclusions
    # ------------------------------------------------------------------
    if any(keyword in text for keyword in ["drop ", "exclude ", "remove ", "ignore "]):
        mentioned_columns = _extract_explicit_columns(text, candidate_columns)
        protected_columns = {scoped_problem.target_column} if scoped_problem is not None else set()
        explicit_exclusions = [col for col in mentioned_columns if col not in protected_columns]
        if explicit_exclusions:
            action.add_features_to_exclude.extend(explicit_exclusions)
            mark(ResumeFrom.ETL, f"explicit exclusions requested: {', '.join(explicit_exclusions)}")

    # ------------------------------------------------------------------
    # Model changes
    # ------------------------------------------------------------------
    if "logistic regression" in text:
        action.set_model_recommendation = ModelRecommendation.LOGISTIC_REGRESSION
        mark(ResumeFrom.MODEL, "switch model to logistic regression")
    elif "random forest" in text:
        action.set_model_recommendation = ModelRecommendation.RANDOM_FOREST
        mark(ResumeFrom.MODEL, "switch model to random forest")
    elif "gradient boosting" in text or "gbm" in text:
        action.set_model_recommendation = ModelRecommendation.GRADIENT_BOOSTING
        mark(ResumeFrom.MODEL, "switch model to gradient boosting")
    elif "ridge" in text:
        action.set_model_recommendation = ModelRecommendation.RIDGE
        mark(ResumeFrom.MODEL, "switch model to ridge")

    # ------------------------------------------------------------------
    # Metric changes
    # ------------------------------------------------------------------
    for metric, aliases in _METRIC_ALIASES.items():
        if any(alias in text for alias in aliases):
            action.set_success_metric = metric
            mark(ResumeFrom.MODEL, f"change success metric to {metric}")
            break

    # ------------------------------------------------------------------
    # Re-scope triggers
    # ------------------------------------------------------------------
    if any(phrase in text for phrase in [
        "re-scope",
        "rescope",
        "target is wrong",
        "wrong target",
        "change the target",
        "rewrite the problem",
        "problem statement is wrong",
        "constraint changed",
        "business objective changed",
    ]):
        action.update_business_brief = feedback
        mark(ResumeFrom.SCOPING, "feedback requires re-scoping")

    # ------------------------------------------------------------------
    # Fallback: if nothing matched, preserve the previous behaviour
    # ------------------------------------------------------------------
    if not matched_any:
        action.update_business_brief = feedback
        resume_from = ResumeFrom.SCOPING
        rationale_parts.append("no deterministic rule matched; fallback to full re-scope")

    action.resume_from = resume_from or ResumeFrom.SCOPING
    action.rationale = "; ".join(rationale_parts)
    return action
