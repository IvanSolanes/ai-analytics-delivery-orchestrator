"""
graph.py
--------
LangGraph state machine assembly.

Key design decision:
  We use interrupt_before=["human_review_node"] at compile time rather
  than calling interrupt() inside the node. This is the more stable
  LangGraph pattern — it pauses the graph before the node runs, saves
  state to the checkpointer, and resumes cleanly when the caller calls
  graph.invoke(Command(resume=value), config=config).

  The interrupt() call inside human_review_node then receives the
  resume value directly and processes it.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from orchestrator.config import settings
from orchestrator.nodes.dashboard import dashboard_node
from orchestrator.nodes.etl import etl_node
from orchestrator.nodes.human_review import human_review_node
from orchestrator.nodes.model import model_node
from orchestrator.nodes.profiling import profiling_node
from orchestrator.nodes.qa import qa_node
from orchestrator.nodes.review import review_node
from orchestrator.nodes.scoping import scoping_node
from orchestrator.state import AnalyticsState

console = Console()


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_review(state: AnalyticsState) -> str:
    review_scores = state.get("review_scores")
    retry_count = state.get("retry_count", 0)
    if (
        review_scores is not None
        and review_scores.retry_recommended
        and retry_count < settings.max_retries
    ):
        console.print(
            f"  [yellow]Routing: retry dashboard "
            f"(attempt {retry_count + 1}/{settings.max_retries})[/yellow]"
        )
        return "retry_dashboard"
    return "proceed_to_human_review"


def route_after_human_review(state: AnalyticsState) -> str:
    feedback = state.get("human_feedback")
    if feedback is None:
        console.print("  Routing: approved -> QA")
        return "approved"
    console.print("  Routing: revision requested -> scoping")
    return "revision_requested"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def increment_retry_count(state: AnalyticsState) -> dict:
    current = state.get("retry_count", 0)
    console.print(f"  Retry count: {current} -> {current + 1}")
    return {"retry_count": current + 1}


def inject_human_feedback(state: AnalyticsState) -> dict:
    feedback = state.get("human_feedback", "")
    original_brief = state.get("business_brief", "")
    updated_brief = (
        f"{original_brief}\n\n"
        f"[REVISION NOTES FROM HUMAN REVIEW]: {feedback}"
    )
    console.print(f"  Injecting feedback into brief ({len(feedback)} chars)")
    return {
        "business_brief": updated_brief,
        "human_feedback": None,
        "retry_count": 0,
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the analytics orchestrator graph."""
    workflow = StateGraph(AnalyticsState)

    workflow.add_node("scoping_node", scoping_node)
    workflow.add_node("profiling_node", profiling_node)
    workflow.add_node("etl_node", etl_node)
    workflow.add_node("model_node", model_node)
    workflow.add_node("dashboard_node", dashboard_node)
    workflow.add_node("review_node", review_node)
    workflow.add_node("increment_retry", increment_retry_count)
    workflow.add_node("human_review_node", human_review_node)
    workflow.add_node("inject_feedback", inject_human_feedback)
    workflow.add_node("qa_node", qa_node)

    workflow.add_edge(START, "scoping_node")
    workflow.add_edge("scoping_node", "profiling_node")
    workflow.add_edge("profiling_node", "etl_node")
    workflow.add_edge("etl_node", "model_node")
    workflow.add_edge("model_node", "dashboard_node")
    workflow.add_edge("dashboard_node", "review_node")

    workflow.add_conditional_edges(
        "review_node",
        route_after_review,
        {
            "retry_dashboard": "increment_retry",
            "proceed_to_human_review": "human_review_node",
        },
    )
    workflow.add_edge("increment_retry", "dashboard_node")

    workflow.add_conditional_edges(
        "human_review_node",
        route_after_human_review,
        {
            "approved": "qa_node",
            "revision_requested": "inject_feedback",
        },
    )
    workflow.add_edge("inject_feedback", "scoping_node")
    workflow.add_edge("qa_node", END)

    checkpointer = MemorySaver()

    # interrupt_before pauses the graph BEFORE the node runs —
    # more stable than interrupt() inside the node for some LangGraph versions
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review_node"],
    )
    return graph


graph = build_graph()
