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
    ResumeFrom.DASHBOARD: 3,
}

_METRIC_ALIASES = {
    "roc_auc": ["roc_auc", "roc auc", "auc"],
    "f1_weighted": ["f1_weighted", "weighted f1", "f1 weighted"],
    "f1": ["f1", "f-1"],
    "accuracy": ["accuracy", "accurate"],
    "rmse": ["rmse", "root mean squared error"],
    "mae": ["mae", "mean absolute error"],
}

_DASHBOARD_RULES = [
    (
        ["confusion matrix", "confusion heatmap"],
        "Add a confusion matrix on the held-out test split for classification tasks.",
    ),
    (
        ["roc curve", "auc curve"],
        "Add an ROC curve on the held-out test split for binary classification when probabilities are available.",
    ),
    (
        ["precision recall curve", "precision-recall curve", "pr curve"],
        "Add a precision-recall curve on the held-out test split for classification when probabilities are available.",
    ),
    (
        ["calibration plot", "calibration curve"],
        "Add a calibration plot when the model exposes probabilities.",
    ),
    (
        ["feature importance", "important features", "top features", "top drivers"],
        "Highlight the most important features in a dedicated dashboard section.",
    ),
    (
        ["executive summary", "business summary", "business-friendly", "stakeholder-friendly", "non-technical"],
        "Add a plain-English executive summary aimed at business stakeholders.",
    ),
    (
        ["limitations", "caveats"],
        "Surface model limitations clearly in the dashboard.",
    ),
    (
        ["recommendation", "recommendations", "next steps"],
        "Add a recommendations or next-steps section grounded in the current results.",
    ),
    (
        ["layout", "cleaner layout", "improve layout"],
        "Improve the dashboard layout and section ordering for readability.",
    ),
    (
        ["class balance", "class distribution"],
        "Add a class-balance view using the available target data.",
    ),
]

_DASHBOARD_CONTEXT_TERMS = [
    "dashboard",
    "streamlit",
    "app",
    "chart",
    "plot",
    "visual",
    "visualization",
    "layout",
    "summary",
]


def _choose_earliest_resume(
    current: Optional[ResumeFrom],
    candidate: ResumeFrom,
) -> ResumeFrom:
    if current is None:
        return candidate
    return candidate if _STAGE_ORDER[candidate] < _STAGE_ORDER[current] else current


def _extract_first_int(text: str) -> Optional[int]:
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _extract_first_float(text: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
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


def _append_dashboard_request(action: FeedbackAction, request: str) -> None:
    if request not in action.dashboard_requests:
        action.dashboard_requests.append(request)


def parse_human_feedback(feedback: str, state: AnalyticsState) -> FeedbackAction:
    """
    Parse free-text human feedback into a deterministic action plan.

    Supported v1 actions:
      - "try fewer features" / "reduce features"
      - correlation-based feature filtering
      - exclude explicit columns mentioned by name
      - switch model algorithm
      - change primary metric
      - add dashboard-only requests like confusion matrix / executive summary
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
    # Feature-selection changes
    # ------------------------------------------------------------------
    if any(phrase in text for phrase in [
        "fewer features",
        "reduce features",
        "too many features",
        "simpler feature set",
        "use fewer variables",
    ]):
        action.set_feature_selection_strategy = FeatureSelectionStrategy.SELECT_K_BEST
        requested_k = _extract_first_int(text)
        if requested_k is not None:
            action.set_feature_selection_k = requested_k
        mark(ResumeFrom.ETL, "reduce numeric features with SelectKBest")

    if "correlation filter" in text or "correlation-based" in text:
        action.set_feature_selection_strategy = FeatureSelectionStrategy.CORRELATION_FILTER
        threshold = _extract_first_float(text)
        if threshold is not None and threshold <= 1.0:
            action.set_feature_selection_threshold = threshold
        mark(ResumeFrom.ETL, "use correlation-based numeric feature filtering")

    # ------------------------------------------------------------------
    # Explicit feature exclusions
    # ------------------------------------------------------------------
    if any(token in text for token in ["drop ", "exclude ", "remove "]):
        mentioned_columns = _extract_explicit_columns(feedback, candidate_columns)
        protected_columns = set()
        if scoped_problem is not None:
            protected_columns.add(scoped_problem.target_column)
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
    # Dashboard-only changes
    # ------------------------------------------------------------------
    matched_dashboard_rule = False
    for aliases, request in _DASHBOARD_RULES:
        if any(alias in text for alias in aliases):
            _append_dashboard_request(action, request)
            matched_dashboard_rule = True

    if matched_dashboard_rule:
        mark(ResumeFrom.DASHBOARD, "dashboard-specific review requests detected")

    if any(term in text for term in _DASHBOARD_CONTEXT_TERMS) and not matched_dashboard_rule:
        _append_dashboard_request(action, feedback.strip())
        mark(ResumeFrom.DASHBOARD, "generic dashboard/app feedback detected")

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
