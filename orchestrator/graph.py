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
from orchestrator.feedback import parse_human_feedback
from orchestrator.nodes.dashboard import dashboard_node
from orchestrator.nodes.etl import etl_node
from orchestrator.nodes.human_review import human_review_node
from orchestrator.nodes.model import model_node
from orchestrator.nodes.profiling import profiling_node
from orchestrator.nodes.qa import qa_node
from orchestrator.nodes.review import review_node
from orchestrator.nodes.scoping import scoping_node
from orchestrator.state import AnalyticsState, ResumeFrom

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
    console.print("  Routing: revision requested -> inject feedback")
    return "revision_requested"


def route_after_feedback_injection(state: AnalyticsState) -> str:
    resume_from = state.get("resume_from", ResumeFrom.SCOPING)
    if isinstance(resume_from, ResumeFrom):
        route = resume_from.value
    else:
        route = str(resume_from)
    console.print(f"  Routing after feedback injection -> {route}")
    return route


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def increment_retry_count(state: AnalyticsState) -> dict:
    current = state.get("retry_count", 0)
    console.print(f"  Retry count: {current} -> {current + 1}")
    return {"retry_count": current + 1}


def inject_human_feedback(state: AnalyticsState) -> dict:
    feedback = state.get("human_feedback") or ""
    if not feedback.strip():
        return {
            "human_feedback": None,
            "retry_count": 0,
            "resume_from": ResumeFrom.SCOPING,
        }

    action = parse_human_feedback(feedback, state)
    console.print(f"  Parsed feedback action: {action.rationale}")

    existing_dashboard_requests = list(state.get("dashboard_requests", []))
    merged_dashboard_requests = existing_dashboard_requests.copy()
    for request in action.dashboard_requests:
        if request not in merged_dashboard_requests:
            merged_dashboard_requests.append(request)

    updates: dict = {
        "human_feedback": None,
        "retry_count": 0,
        "feedback_action": action,
        "resume_from": action.resume_from,
        "dashboard_requests": merged_dashboard_requests,
    }

    scoped_problem = state.get("scoped_problem")
    if scoped_problem is not None:
        scoped_problem = scoped_problem.model_copy(deep=True)

        if action.set_model_recommendation is not None:
            scoped_problem.model_recommendation = action.set_model_recommendation

        if action.set_success_metric is not None:
            scoped_problem.success_metric = action.set_success_metric

        if action.set_feature_selection_strategy is not None:
            scoped_problem.feature_selection_strategy = action.set_feature_selection_strategy

        if action.set_feature_selection_k is not None:
            scoped_problem.feature_selection_k = action.set_feature_selection_k

        if action.set_feature_selection_threshold is not None:
            scoped_problem.feature_selection_threshold = action.set_feature_selection_threshold

        if action.add_features_to_exclude:
            scoped_problem.features_to_exclude = sorted(
                set(scoped_problem.features_to_exclude)
                | set(action.add_features_to_exclude)
            )

        updates["scoped_problem"] = scoped_problem

    if action.update_business_brief:
        original_brief = state.get("business_brief", "")
        updated_brief = (
            f"{original_brief}\n\n"
            f"[REVISION NOTES FROM HUMAN REVIEW]: {action.update_business_brief}"
        )
        updates["business_brief"] = updated_brief

    return updates


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
    workflow.add_conditional_edges(
        "inject_feedback",
        route_after_feedback_injection,
        {
            ResumeFrom.SCOPING.value: "scoping_node",
            ResumeFrom.ETL.value: "etl_node",
            ResumeFrom.MODEL.value: "model_node",
            ResumeFrom.DASHBOARD.value: "dashboard_node",
        },
    )
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
