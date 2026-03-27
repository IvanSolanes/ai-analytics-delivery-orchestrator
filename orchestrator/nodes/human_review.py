"""
nodes/human_review.py
---------------------
Node 7: Human Review

When interrupt_before=["human_review_node"] is set at compile time,
LangGraph pauses BEFORE this node runs and saves the full state to the
checkpointer. The caller resumes by invoking the graph again with a
Command(resume=...) — at which point this node runs and reads the
feedback from the interrupt value.

This approach is more stable than calling interrupt() inside the node
and avoids internal LangGraph bugs with Command(resume=None).
"""

from langgraph.types import interrupt
from rich.console import Console

from orchestrator.state import AnalyticsState

console = Console()


def human_review_node(state: AnalyticsState) -> dict:
    """
    Human review node.

    LangGraph pauses before this node (interrupt_before at compile time).
    When resumed, this node receives the human's decision via interrupt()
    and writes it to state.

    feedback = ""        -> approved, continue to QA
    feedback = "string"  -> revision notes, loop back to scoping
    """
    console.rule("[bold]Node 7: Human Review[/bold]")

    review_scores = state.get("review_scores")
    model_results = state.get("model_results")

    # Build summary for display
    summary_lines = [
        "",
        "=" * 60,
        "  DELIVERY SUMMARY — HUMAN REVIEW REQUIRED",
        "=" * 60,
    ]
    if model_results:
        summary_lines += [
            f"  Algorithm  : {model_results.algorithm}",
            f"  Test score : {model_results.test_score:.4f} ({model_results.primary_metric})",
            f"  CV mean    : {model_results.cv_mean:.4f} ± {model_results.cv_std:.4f}",
        ]
    if review_scores:
        summary_lines += [
            "",
            f"  Review composite score: {review_scores.composite_score:.2f}",
            f"  Summary: {review_scores.overall_summary}",
        ]
        all_flags = (
            review_scores.problem_clarity.flags
            + review_scores.data_quality.flags
            + review_scores.model_appropriateness.flags
            + review_scores.dashboard_completeness.flags
            + review_scores.limitations_acknowledged.flags
        )
        if all_flags:
            summary_lines += ["", "  Flags:"] + [f"    - {f}" for f in all_flags]

    summary_lines += ["", "=" * 60, ""]
    for line in summary_lines:
        console.print(line)

    # Get the human's decision — interrupt() returns whatever was passed
    # to Command(resume=...) when the graph was resumed
    feedback = interrupt({"summary": summary_lines})

    # Empty string or None means approved
    if not feedback:
        console.print("  [green]Approved. Continuing to QA.[/green]")
        return {"human_feedback": None}
    else:
        console.print(f"  [yellow]Revision requested:[/yellow] {feedback}")
        return {"human_feedback": str(feedback)}
