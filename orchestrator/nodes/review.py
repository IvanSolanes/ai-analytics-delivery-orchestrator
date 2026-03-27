"""
nodes/review.py
---------------
Node 6: Rubric Review

Responsibility:
  Read all prior node outputs, build a structured evidence package,
  call the LLM to score five rubric dimensions, and return ReviewScores.
  The composite_score drives the retry conditional edge in the graph.

Why LLM for review?
  Judgment calls — 'is this problem statement precise?', 'are these
  limitations specific?' — require language understanding. Deterministic
  code cannot evaluate prose quality.

Why structured output?
  ReviewScores is a Pydantic model. Using .with_structured_output() means
  scores are comparable across runs and stored in state as typed fields,
  not as free text that would be unverifiable.

Why is retry_recommended computed here and not in the graph?
  The node is the one with context about what the scores mean. The graph
  just reads retry_recommended as a boolean — it doesn't need to know the
  threshold. This keeps routing logic simple.

Inputs from state:
  - scoped_problem, data_profile, etl_artifacts, model_results,
    dashboard_code, retry_count

Outputs to state:
  - review_scores: ReviewScores
"""

from pathlib import Path

from langchain_core.prompts import PromptTemplate
from rich.console import Console

from orchestrator.config import get_llm, settings
from orchestrator.state import (
    AnalyticsState,
    ReviewScores,
)

console = Console()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "review.txt"

# How many lines of dashboard code to include in the review context.
# Enough to verify key sections are present, not so much it bloats the prompt.
_DASHBOARD_SNIPPET_LINES = 40


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_review_context(state: AnalyticsState) -> dict:
    """
    Build the evidence package for the review prompt.

    Every value is a plain string. The LLM receives facts from state,
    not raw Python objects.
    """
    scoped = state["scoped_problem"]
    profile = state["data_profile"]
    model = state["model_results"]
    dashboard_code = state.get("dashboard_code", "")

    # Top 5 features as readable lines
    top_features_str = "\n".join(
        f"  {item['feature']}: {item['importance']:.4f}"
        for item in model.feature_importances[:5]
    ) or "  Not available."

    # Top correlations
    top_corr_str = "\n".join(
        f"  {item['feature']}: {item['correlation']:.4f}"
        for item in profile.top_correlations[:5]
    ) or "  Not computed."

    # Quality warnings
    warnings_str = "\n".join(
        f"  - {w}" for w in profile.quality_warnings
    ) or "  None."

    # Limitations from scope
    limitations_str = "\n".join(
        f"  - {lim}" for lim in scoped.limitations
    ) or "  None listed."

    # Out of scope
    out_of_scope_str = "\n".join(
        f"  - {item}" for item in scoped.out_of_scope
    ) or "  None listed."

    # Training notes
    training_notes_str = "\n".join(
        f"  - {note}" for note in model.training_notes
    ) or "  None."

    # Dashboard snippet — first N lines only
    dashboard_lines = dashboard_code.splitlines()
    dashboard_snippet = "\n".join(
        dashboard_lines[:_DASHBOARD_SNIPPET_LINES]
    ) or "  [No dashboard code generated]"

    return {
        # Problem scope
        "target_column": scoped.target_column,
        "task_type": scoped.task_type.value,
        "success_metric": scoped.success_metric,
        "problem_statement": scoped.problem_statement,
        "out_of_scope": out_of_scope_str,
        "limitations": limitations_str,
        # Data profile
        "n_rows": str(profile.n_rows),
        "n_columns": str(profile.n_columns),
        "high_null_columns": str(profile.high_null_columns) or "None",
        "quality_warnings": warnings_str,
        "target_distribution": str(profile.target_distribution),
        "top_correlations": top_corr_str,
        # Model results
        "algorithm": model.algorithm,
        "primary_metric": model.primary_metric,
        "cv_mean": f"{model.cv_mean:.4f}",
        "cv_std": f"{model.cv_std:.4f}",
        "test_score": f"{model.test_score:.4f}",
        "training_notes": training_notes_str,
        "top_features": top_features_str,
        # Dashboard
        "dashboard_snippet": dashboard_snippet,
        # Threshold for retry decision
        "review_threshold": str(settings.review_score_threshold),
    }


# ---------------------------------------------------------------------------
# The node function
# ---------------------------------------------------------------------------

def review_node(state: AnalyticsState) -> dict:
    """
    Rubric review node.

    Builds an evidence package, calls the LLM with structured output
    parsing, computes the composite score, and sets retry_recommended.

    Returns:
        dict with key "review_scores" containing a ReviewScores instance.
    """
    console.rule("[bold]Node 6: Rubric Review[/bold]")

    retry_count = state.get("retry_count", 0)
    console.print(f"  Review pass {retry_count + 1} "
                  f"(threshold: {settings.review_score_threshold})")

    # --- Step 1: Build context ---
    context = _build_review_context(state)

    # --- Step 2: Build chain ---
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = PromptTemplate(
        input_variables=list(context.keys()),
        template=template,
    )

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(ReviewScores)
    chain = prompt | structured_llm

    # --- Step 3: Call the LLM ---
    console.print("  Calling LLM for rubric review...")
    review_scores: ReviewScores = chain.invoke(context)

    # --- Step 4: Log scores ---
    console.print(f"  Problem clarity     : {review_scores.problem_clarity.score:.2f}")
    console.print(f"  Data quality        : {review_scores.data_quality.score:.2f}")
    console.print(f"  Model appropriateness: {review_scores.model_appropriateness.score:.2f}")
    console.print(f"  Dashboard completeness: {review_scores.dashboard_completeness.score:.2f}")
    console.print(f"  Limitations         : {review_scores.limitations_acknowledged.score:.2f}")
    console.print(f"  Composite score     : {review_scores.composite_score:.2f}")

    if review_scores.retry_recommended:
        console.print(
            f"  [yellow]Retry recommended[/yellow] — "
            f"score {review_scores.composite_score:.2f} < "
            f"threshold {settings.review_score_threshold}"
        )
    else:
        console.print("  [green]Review passed.[/green]")

    # --- Step 5: Log any flags ---
    all_flags = (
        review_scores.problem_clarity.flags
        + review_scores.data_quality.flags
        + review_scores.model_appropriateness.flags
        + review_scores.dashboard_completeness.flags
        + review_scores.limitations_acknowledged.flags
    )
    for flag in all_flags:
        console.print(f"  [yellow]Flag:[/yellow] {flag}")

    return {"review_scores": review_scores}
