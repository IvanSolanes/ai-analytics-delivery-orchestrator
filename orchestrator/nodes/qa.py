"""
nodes/qa.py
-----------
Node 8: QA Checklist + Final Report Assembly

Two types of checks:
  1. Deterministic assertions — binary pass/fail, no LLM needed
  2. LLM hallucination flag scan — reads review text, flags claims
     that are demonstrably wrong or contradicted by verified facts

Key design principle for the flag scan:
  The scanner receives a rich context of verified facts. It is explicitly
  instructed NOT to flag claims that are simply not mentioned in the
  context — only claims that are demonstrably wrong or invented numbers.
  This prevents false positives where true statements are flagged just
  because the scanner's context was too narrow.
"""

import json
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from rich.console import Console

from orchestrator.config import get_llm, settings
from orchestrator.state import (
    AnalyticsState,
    AssertionResult,
    FinalReport,
    QAResult,
)

console = Console()


# ---------------------------------------------------------------------------
# Deterministic assertions
# ---------------------------------------------------------------------------

def _run_assertions(state: AnalyticsState) -> list[AssertionResult]:
    """Run 8 deterministic consistency checks across all state fields."""
    scoped = state["scoped_problem"]
    profile = state["data_profile"]
    etl = state["etl_artifacts"]
    model = state["model_results"]
    assertions = []

    # 1. Target column consistency
    target_match = scoped.target_column == etl.target_column
    assertions.append(AssertionResult(
        name="target_column_match",
        passed=target_match,
        detail=(
            f"Scoped target '{scoped.target_column}' "
            f"{'==' if target_match else '!='} "
            f"ETL target '{etl.target_column}'"
        ),
    ))

    # 2. Metric consistency
    metric_match = scoped.success_metric in model.primary_metric
    assertions.append(AssertionResult(
        name="metric_consistency",
        passed=metric_match,
        detail=(
            f"Scoped metric '{scoped.success_metric}' "
            f"{'found in' if metric_match else 'not found in'} "
            f"model primary_metric '{model.primary_metric}'"
        ),
    ))

    # 3. Task type consistency
    task_match = scoped.task_type == model.task_type
    assertions.append(AssertionResult(
        name="task_type_match",
        passed=task_match,
        detail=(
            f"Scoped task '{scoped.task_type.value}' "
            f"{'==' if task_match else '!='} "
            f"model task '{model.task_type.value}'"
        ),
    ))

    # 4. Test score plausibility
    if "classification" in scoped.task_type.value:
        score_ok = model.test_score > 0.5
        detail = (
            f"Test score {model.test_score:.4f} "
            f"{'>' if score_ok else '<='} 0.5 (random baseline)"
        )
    else:
        score_ok = model.test_score > 0.0
        detail = f"Test score {model.test_score:.4f} {'>' if score_ok else '<='} 0.0"
    assertions.append(AssertionResult(
        name="test_score_plausible", passed=score_ok, detail=detail
    ))

    # 5. Limitations present
    has_limitations = len(scoped.limitations) > 0
    assertions.append(AssertionResult(
        name="limitations_present",
        passed=has_limitations,
        detail=f"Found {len(scoped.limitations)} limitation(s) in scoped problem",
    ))

    # 6. Quality warnings present
    has_warnings = len(profile.quality_warnings) > 0
    assertions.append(AssertionResult(
        name="quality_warnings_present",
        passed=has_warnings,
        detail=f"Found {len(profile.quality_warnings)} quality warning(s)",
    ))

    # 7. Feature importances present
    has_importances = len(model.feature_importances) > 0
    assertions.append(AssertionResult(
        name="feature_importances_present",
        passed=has_importances,
        detail=f"Found {len(model.feature_importances)} feature importance(s)",
    ))

    # 8. CV scores count
    from orchestrator.nodes.model import N_CV_FOLDS
    cv_count_ok = len(model.cv_scores) == N_CV_FOLDS
    assertions.append(AssertionResult(
        name="cv_scores_complete",
        passed=cv_count_ok,
        detail=(
            f"Found {len(model.cv_scores)} CV scores "
            f"(expected {N_CV_FOLDS})"
        ),
    ))

    return assertions


# ---------------------------------------------------------------------------
# LLM hallucination flag scan
# ---------------------------------------------------------------------------
# Why the prompt says "do NOT flag claims simply not mentioned":
#   The scanner only receives a subset of all verified facts. If a claim
#   is true (e.g. the target column name is correct) but that specific
#   fact wasn't in the scanner's context, a naive scanner would flag it
#   as a hallucination — a false positive. We explicitly instruct the
#   scanner to only flag things that are DEMONSTRABLY WRONG, not merely
#   absent from its context.

_FLAG_SCAN_PROMPT = """You are a QA analyst checking an analytics review for
hallucinated or fabricated claims.

The review text is:
{review_text}

These are the VERIFIED FACTS about the analysis:
Target column: {target_column}
Task type: {task_type}
Algorithm: {algorithm}
CV mean score ({primary_metric}): {cv_mean}
Test score: {test_score}
Feature importance chart present: {has_feature_importance}
Top features: {top_features}
Feature columns (sample): {feature_columns}
Quality warnings: {quality_warnings}
Top correlations with target: {top_correlations}
Target distribution: {target_distribution}
Training notes: {training_notes}

YOUR TASK:
Identify claims in the review text that are DEMONSTRABLY WRONG or FABRICATED.

IMPORTANT RULES:
- Do NOT flag a claim just because it is not explicitly listed in the facts above.
  The facts above are a summary — the full analysis has more detail.
- Only flag claims that directly CONTRADICT the verified facts.
  For example: flagging "the test score is 0.95" when verified score is 0.80.
- Do NOT flag: correct column names, correct algorithm names, correct scores,
  or general statements about model quality that are consistent with the facts.
- If the review text accurately describes the analysis, return an empty flags list.

Return a JSON object with a single key "flags" containing a list of strings.
Each string must quote the specific wrong claim and explain why it is wrong.
If no fabricated claims are found, return {{"flags": []}}.
Output ONLY the JSON object — no explanation, no markdown.
"""


def _scan_for_hallucinations(state: AnalyticsState) -> list[str]:
    """
    Use the LLM to scan review text for fabricated or contradicted claims.
    Returns a list of flag strings. Empty list means clean.
    """
    review_scores = state.get("review_scores")
    if not review_scores:
        return []

    scoped = state["scoped_problem"]
    profile = state["data_profile"]
    etl = state["etl_artifacts"]
    model = state["model_results"]

    # Build the full review text from all justifications
    review_text_parts = [
        f"Overall: {review_scores.overall_summary}",
        f"Problem clarity: {review_scores.problem_clarity.justification}",
        f"Data quality: {review_scores.data_quality.justification}",
        f"Model: {review_scores.model_appropriateness.justification}",
        f"Dashboard: {review_scores.dashboard_completeness.justification}",
        f"Limitations: {review_scores.limitations_acknowledged.justification}",
    ]
    review_text = "\n".join(review_text_parts)

    top_corr_str = ", ".join(
        f"{c['feature']}={c['correlation']:.3f}"
        for c in profile.top_correlations[:3]
    ) or "none computed"

    top_features_str = ", ".join(
        item["feature"] for item in model.feature_importances[:5]
    ) or "none"

    # Sample of feature columns — give scanner enough context
    feature_sample = ", ".join(etl.feature_columns[:5])
    if len(etl.feature_columns) > 5:
        feature_sample += f" (+ {len(etl.feature_columns) - 5} more)"

    training_notes_str = "; ".join(model.training_notes) or "none"

    prompt = PromptTemplate(
        input_variables=[
            "review_text", "target_column", "task_type", "algorithm",
            "primary_metric", "cv_mean", "test_score",
            "has_feature_importance", "top_features", "feature_columns",
            "quality_warnings", "top_correlations", "target_distribution",
            "training_notes",
        ],
        template=_FLAG_SCAN_PROMPT,
    )

    llm = get_llm(temperature=0)

    try:
        chain = prompt | llm
        response = chain.invoke({
            "review_text": review_text,
            "target_column": scoped.target_column,
            "task_type": scoped.task_type.value,
            "algorithm": model.algorithm,
            "primary_metric": model.primary_metric,
            "cv_mean": f"{model.cv_mean:.4f}",
            "test_score": f"{model.test_score:.4f}",
            "has_feature_importance": "yes" if model.feature_importances else "no",
            "top_features": top_features_str,
            "feature_columns": feature_sample,
            "quality_warnings": str(profile.quality_warnings),
            "top_correlations": top_corr_str,
            "target_distribution": str(profile.target_distribution),
            "training_notes": training_notes_str,
        })

        raw = (
            response.content
            if hasattr(response, "content")
            else str(response)
        ).strip()

        # Strip markdown fences if present
        import re
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("flags", [])

    except Exception as e:
        console.print(f"  [yellow]Flag scan error (non-fatal): {e}[/yellow]")

    return []


# ---------------------------------------------------------------------------
# Final report assembly
# ---------------------------------------------------------------------------

def _assemble_final_report(
    state: AnalyticsState,
    qa_result: QAResult,
) -> FinalReport:
    scoped = state["scoped_problem"]
    profile = state["data_profile"]
    etl = state["etl_artifacts"]
    model = state["model_results"]
    review = state.get("review_scores")

    top_features = [item["feature"] for item in model.feature_importances[:5]]

    artifacts = {
        "pipeline": etl.pipeline_path,
        "model": model.model_path,
        "dashboard": str(settings.dashboards_dir / "dashboard.py"),
        "profile_report": profile.profile_report_path or "not generated",
    }

    return FinalReport(
        problem_statement=scoped.problem_statement,
        success_metric=scoped.success_metric,
        task_type=scoped.task_type,
        model_algorithm=model.algorithm,
        test_score=model.test_score,
        primary_metric=model.primary_metric,
        top_features=top_features,
        data_quality_warnings=profile.quality_warnings,
        preprocessing_summary=etl.preprocessing_steps,
        review_composite_score=review.composite_score if review else 0.0,
        qa_passed=qa_result.overall_passed,
        limitations=scoped.limitations,
        artifacts=artifacts,
        delivery_notes=state.get("human_feedback") or "Approved.",
    )


# ---------------------------------------------------------------------------
# The node function
# ---------------------------------------------------------------------------

def qa_node(state: AnalyticsState) -> dict:
    """
    QA checklist node.

    Runs deterministic assertions, scans review text for fabricated claims,
    assembles the final report, and saves it to disk.
    """
    console.rule("[bold]Node 8: QA Checklist[/bold]")

    # Layer 1: deterministic assertions
    console.print("  Running deterministic assertions...")
    assertions = _run_assertions(state)

    passed = sum(1 for a in assertions if a.passed)
    failed = sum(1 for a in assertions if not a.passed)

    for assertion in assertions:
        status = "[green]PASS[/green]" if assertion.passed else "[red]FAIL[/red]"
        console.print(f"  {status} {assertion.name}: {assertion.detail}")

    # Layer 2: hallucination flag scan
    console.print("  Scanning review text for fabricated claims...")
    hallucination_flags = _scan_for_hallucinations(state)

    if hallucination_flags:
        for flag in hallucination_flags:
            console.print(f"  [red]Hallucination flag:[/red] {flag}")
    else:
        console.print("  [green]No hallucination flags found.[/green]")

    # Build QAResult
    overall_passed = failed == 0 and len(hallucination_flags) == 0
    qa_summary = (
        f"{passed}/{len(assertions)} assertions passed"
        + (f", {len(hallucination_flags)} hallucination flag(s)" if hallucination_flags else "")
        + (" — QA passed." if overall_passed else " — QA failed.")
    )

    qa_result = QAResult(
        assertions=assertions,
        assertions_passed=passed,
        assertions_failed=failed,
        hallucination_flags=hallucination_flags,
        overall_passed=overall_passed,
        qa_summary=qa_summary,
    )

    # Assemble and save final report
    final_report = _assemble_final_report(state, qa_result)

    report_path = settings.output_dir / "final_report.json"
    report_path.write_text(
        final_report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    console.print(f"  Final report saved to: {report_path}")

    if overall_passed:
        console.print("  [green]QA complete — delivery ready.[/green]")
    else:
        console.print("  [yellow]QA complete — issues found, review report.[/yellow]")

    return {
        "qa_result": qa_result,
        "final_report": final_report,
    }
