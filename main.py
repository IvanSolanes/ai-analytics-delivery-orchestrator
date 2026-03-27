"""
main.py
-------
Entry point for the AI Analytics Delivery Orchestrator.

Usage:
    python main.py
    python main.py --dataset data/credit_risk_sample.csv
"""

import sys
import uuid
import argparse
from pathlib import Path

from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from orchestrator.graph import graph
from orchestrator.state import AnalyticsState

console = Console()

DEFAULT_BRIEF = (
    "The credit team needs to score incoming loan applications. "
    "Build a model that flags high-risk borrowers based on financial "
    "history and credit utilisation patterns, so analysts can prioritise "
    "manual review of borderline cases."
)
DEFAULT_DATASET = "data/credit_risk_sample.csv"


def _display_final_report(state: dict) -> None:
    final_report = state.get("final_report")
    qa_result = state.get("qa_result")
    if not final_report:
        console.print("[red]No final report found in state.[/red]")
        return

    console.print()
    console.print(Panel.fit(
        "[bold]DELIVERY COMPLETE[/bold]",
        border_style="green" if (qa_result and qa_result.overall_passed) else "yellow",
    ))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Algorithm", final_report.model_algorithm)
    table.add_row(f"Test {final_report.primary_metric}", f"{final_report.test_score:.4f}")
    table.add_row("Review score", f"{final_report.review_composite_score:.2f}")
    table.add_row("QA passed", "[green]Yes[/green]" if final_report.qa_passed else "[red]No[/red]")
    console.print(table)

    console.print("\n[bold]Top features (SHAP):[/bold]")
    for i, feat in enumerate(final_report.top_features[:5], 1):
        console.print(f"  {i}. {feat}")

    console.print("\n[bold]Limitations:[/bold]")
    for lim in final_report.limitations:
        console.print(f"  • {lim}")

    console.print("\n[bold]Artifacts:[/bold]")
    for name, path in final_report.artifacts.items():
        console.print(f"  {name}: {path}")

    if qa_result:
        console.print(f"\n[bold]QA:[/bold] {qa_result.qa_summary}")


def _show_review_summary(current_state: dict) -> None:
    """Display delivery summary at the human review checkpoint."""
    review_scores = current_state.get("review_scores")
    model_results = current_state.get("model_results")

    console.print()
    console.print("=" * 60)
    console.print("  HUMAN REVIEW REQUIRED")
    console.print("=" * 60)

    if model_results:
        console.print(f"  Algorithm  : {model_results.algorithm}")
        console.print(f"  Test score : {model_results.test_score:.4f} ({model_results.primary_metric})")
        console.print(f"  CV mean    : {model_results.cv_mean:.4f} ± {model_results.cv_std:.4f}")

    if review_scores:
        console.print(f"\n  Review composite score : {review_scores.composite_score:.2f}")
        console.print(f"  Summary: {review_scores.overall_summary}")

        all_flags = (
            review_scores.problem_clarity.flags
            + review_scores.data_quality.flags
            + review_scores.model_appropriateness.flags
            + review_scores.dashboard_completeness.flags
            + review_scores.limitations_acknowledged.flags
        )
        if all_flags:
            console.print("\n  Review flags:")
            for flag in all_flags:
                console.print(f"    - {flag}")

    console.print("=" * 60)
    console.print()
    console.print("  Options:")
    console.print("    Press [Enter]  → approve and continue to QA")
    console.print("    Type notes     → request revisions")
    console.print()


def run(dataset_path: str, brief: str) -> None:
    if not Path(dataset_path).exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        console.print("Run: python data/generate_sample.py")
        sys.exit(1)

    thread_id = f"run-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AnalyticsState = {
        "business_brief": brief,
        "dataset_path": dataset_path,
        "retry_count": 0,
    }

    console.print(Panel.fit(
        f"[bold]AI Analytics Delivery Orchestrator[/bold]\n"
        f"Dataset : {dataset_path}\n"
        f"Thread  : {thread_id}",
        border_style="blue",
    ))

    # --- First invocation: runs until interrupt_before=["human_review_node"] ---
    graph.invoke(initial_state, config=config)

    # --- Check if the graph is paused waiting for human input ---
    # With interrupt_before, LangGraph stops before the node and waits.
    # We detect this by checking graph.get_state(config).next — if it
    # contains "human_review_node", the graph is paused and waiting.
    while True:
        graph_state = graph.get_state(config)

        # .next is a tuple of node names that will run next
        # If "human_review_node" is next, the graph is paused for review
        if "human_review_node" in (graph_state.next or ()):
            _show_review_summary(graph_state.values)

            user_input = input("Your decision: ").strip()
            feedback = user_input  # empty string = approved

            # Resume — feedback is passed to interrupt() inside the node
            graph.invoke(Command(resume=feedback), config=config)

        else:
            # Graph has finished — no more pending nodes
            break

    # --- Display the final delivery report ---
    final_state = graph.get_state(config).values
    _display_final_report(final_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Analytics Delivery Orchestrator")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--brief", default=DEFAULT_BRIEF)
    args = parser.parse_args()
    run(dataset_path=args.dataset, brief=args.brief)
