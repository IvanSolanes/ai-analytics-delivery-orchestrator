"""
evaluation/rubric.py
--------------------
Rubric baseline and side-by-side comparison utilities.

Three tools for evaluating the system over time:

1. RubricBaseline
   A fixed expected minimum score per dimension. If a real run produces
   scores below these baselines, something regressed. Store baselines
   in version control — they represent the quality floor.

2. compare_review_scores()
   Takes two ReviewScores objects (e.g. prompt v1 vs prompt v2) and
   produces a structured diff. Use this when iterating on the review
   prompt to verify that changes improve scores.

3. score_report()
   Human-readable string summary of a ReviewScores object. Used in
   the final delivery report and in CI output.

Why keep this separate from the review node?
  The review node produces scores. This module analyses them. Separating
  production code from evaluation utilities means the evaluation logic
  can be changed without touching the graph.
"""

from dataclasses import dataclass
from typing import Optional

from orchestrator.state import ReviewScores


# ---------------------------------------------------------------------------
# Baseline expectations
# ---------------------------------------------------------------------------
# These are the minimum acceptable scores per dimension for the credit
# risk dataset with the current prompts. If a run produces scores below
# these, it is a regression worth investigating.
#
# How to update:
#   Run the full graph on the credit risk dataset.
#   If the scores are consistently higher than these baselines, raise them.
#   Never lower a baseline — that's admitting regression.

@dataclass
class RubricBaseline:
    """Minimum acceptable scores for each review dimension."""
    problem_clarity: float = 0.7
    data_quality: float = 0.7
    model_appropriateness: float = 0.7
    dashboard_completeness: float = 0.6
    limitations_acknowledged: float = 0.6
    composite: float = 0.65


# The default baseline for the credit risk project
CREDIT_RISK_BASELINE = RubricBaseline()


# ---------------------------------------------------------------------------
# Baseline check
# ---------------------------------------------------------------------------

@dataclass
class BaselineCheckResult:
    """Result of checking scores against the baseline."""
    passed: bool
    failures: list[str]
    summary: str


def check_against_baseline(
    scores: ReviewScores,
    baseline: RubricBaseline = CREDIT_RISK_BASELINE,
) -> BaselineCheckResult:
    """
    Check whether ReviewScores meet the minimum baseline expectations.

    Returns a BaselineCheckResult with pass/fail status and details
    on any dimensions that fell below the baseline.

    Use this in CI to catch prompt regressions automatically.
    """
    failures = []

    checks = [
        ("problem_clarity",
         scores.problem_clarity.score,
         baseline.problem_clarity),
        ("data_quality",
         scores.data_quality.score,
         baseline.data_quality),
        ("model_appropriateness",
         scores.model_appropriateness.score,
         baseline.model_appropriateness),
        ("dashboard_completeness",
         scores.dashboard_completeness.score,
         baseline.dashboard_completeness),
        ("limitations_acknowledged",
         scores.limitations_acknowledged.score,
         baseline.limitations_acknowledged),
        ("composite",
         scores.composite_score,
         baseline.composite),
    ]

    for name, actual, expected in checks:
        if actual < expected:
            failures.append(
                f"{name}: {actual:.2f} < baseline {expected:.2f} "
                f"(gap: {expected - actual:.2f})"
            )

    passed = len(failures) == 0
    summary = (
        f"Baseline check {'PASSED' if passed else 'FAILED'}: "
        f"{len(checks) - len(failures)}/{len(checks)} dimensions met."
    )

    return BaselineCheckResult(
        passed=passed,
        failures=failures,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

@dataclass
class ScoreComparison:
    """Comparison between two ReviewScores runs."""
    dimension: str
    score_a: float
    score_b: float
    delta: float        # score_b - score_a (positive = improvement)
    improved: bool
    regressed: bool


@dataclass
class ComparisonReport:
    """Full side-by-side comparison of two ReviewScores."""
    label_a: str
    label_b: str
    comparisons: list[ScoreComparison]
    composite_delta: float
    overall_improved: bool
    summary: str


def compare_review_scores(
    scores_a: ReviewScores,
    scores_b: ReviewScores,
    label_a: str = "Version A",
    label_b: str = "Version B",
    threshold: float = 0.05,
) -> ComparisonReport:
    """
    Compare two ReviewScores objects side by side.

    Args:
        scores_a: Baseline scores (e.g. old prompt version)
        scores_b: New scores (e.g. updated prompt version)
        label_a: Human-readable label for scores_a
        label_b: Human-readable label for scores_b
        threshold: Minimum delta to count as improved or regressed

    Returns:
        ComparisonReport with per-dimension deltas and overall verdict.

    Typical use:
        old = run_graph_with_prompt_v1()
        new = run_graph_with_prompt_v2()
        report = compare_review_scores(old, new, "Prompt v1", "Prompt v2")
        print(score_report(new))
    """
    dimension_pairs = [
        ("problem_clarity",
         scores_a.problem_clarity.score,
         scores_b.problem_clarity.score),
        ("data_quality",
         scores_a.data_quality.score,
         scores_b.data_quality.score),
        ("model_appropriateness",
         scores_a.model_appropriateness.score,
         scores_b.model_appropriateness.score),
        ("dashboard_completeness",
         scores_a.dashboard_completeness.score,
         scores_b.dashboard_completeness.score),
        ("limitations_acknowledged",
         scores_a.limitations_acknowledged.score,
         scores_b.limitations_acknowledged.score),
    ]

    comparisons = []
    for name, a, b in dimension_pairs:
        delta = b - a
        comparisons.append(ScoreComparison(
            dimension=name,
            score_a=round(a, 3),
            score_b=round(b, 3),
            delta=round(delta, 3),
            improved=delta > threshold,
            regressed=delta < -threshold,
        ))

    composite_delta = scores_b.composite_score - scores_a.composite_score
    improved_count = sum(1 for c in comparisons if c.improved)
    regressed_count = sum(1 for c in comparisons if c.regressed)
    overall_improved = composite_delta > threshold

    summary = (
        f"{label_b} vs {label_a}: "
        f"composite {scores_a.composite_score:.2f} -> "
        f"{scores_b.composite_score:.2f} "
        f"({'+'if composite_delta >= 0 else ''}{composite_delta:.2f}). "
        f"{improved_count} improved, {regressed_count} regressed."
    )

    return ComparisonReport(
        label_a=label_a,
        label_b=label_b,
        comparisons=comparisons,
        composite_delta=round(composite_delta, 3),
        overall_improved=overall_improved,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Score report
# ---------------------------------------------------------------------------

def score_report(scores: ReviewScores) -> str:
    """
    Generate a human-readable text report of ReviewScores.

    Used in terminal output and the final delivery summary.
    """
    lines = [
        "RUBRIC REVIEW SCORES",
        "=" * 40,
        f"  Problem clarity      : {scores.problem_clarity.score:.2f}",
        f"  Data quality         : {scores.data_quality.score:.2f}",
        f"  Model appropriateness: {scores.model_appropriateness.score:.2f}",
        f"  Dashboard completeness: {scores.dashboard_completeness.score:.2f}",
        f"  Limitations ack.     : {scores.limitations_acknowledged.score:.2f}",
        "  " + "-" * 30,
        f"  Composite score      : {scores.composite_score:.2f}",
        f"  Retry recommended    : {scores.retry_recommended}",
        "=" * 40,
        f"  Summary: {scores.overall_summary}",
    ]

    all_flags = (
        scores.problem_clarity.flags
        + scores.data_quality.flags
        + scores.model_appropriateness.flags
        + scores.dashboard_completeness.flags
        + scores.limitations_acknowledged.flags
    )
    if all_flags:
        lines.append("  Flags:")
        for flag in all_flags:
            lines.append(f"    ! {flag}")

    return "\n".join(lines)
