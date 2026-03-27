"""
nodes/dashboard.py
------------------
Node 5: Dashboard Generation

Responsibility:
  Take all prior node outputs, build a structured context, call the LLM,
  validate the generated Streamlit code, and write it to disk.

Why LLM here?
  Writing Streamlit boilerplate from a structured context is exactly where
  an LLM adds value — it handles layout, formatting, and code structure from
  a specification, freeing us from templating logic.

Why NOT LLM for the context?
  The numbers fed to the LLM — scores, feature names, warnings, reviewer
  instructions — come directly from state. The LLM never invents values.
  It only writes the code that displays them.

Validation layers:
  1. ast.parse() — rejects syntactically invalid Python immediately
  2. Content checks — verifies key sections are present in the output
  3. Optional request checks — verifies some reviewer-requested sections exist

Inputs from state:
  - scoped_problem: ScopedProblem
  - data_profile: DataProfile
  - etl_artifacts: ETLArtifacts
  - model_results: ModelResults
  - dashboard_requests: list[str] (optional reviewer requests)

Outputs to state:
  - dashboard_code: str (raw Streamlit Python code)
"""

from __future__ import annotations

import ast
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from rich.console import Console

from orchestrator.config import get_llm, settings
from orchestrator.state import AnalyticsState

console = Console()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "dashboard.txt"

_REQUIRED_SECTIONS = [
    "import streamlit",
    "st.set_page_config",
    ".metric(",
    "plotly",
]


def _format_dashboard_requests(requests: list[str]) -> str:
    if not requests:
        return "No additional dashboard review requests."
    return "\n".join(f" - {request}" for request in requests)


def _build_context(state: AnalyticsState) -> dict:
    """
    Build a flat string-keyed context dict for the prompt template.

    Every value is a plain string — no nested objects, no Pydantic models.
    This keeps the prompt readable and avoids serialisation surprises.
    """
    scoped = state["scoped_problem"]
    profile = state["data_profile"]
    etl = state["etl_artifacts"]
    model = state["model_results"]
    dashboard_requests = list(state.get("dashboard_requests", []))

    importances_str = "\n".join(
        f" - {item['feature']}: {item['importance']:.4f}"
        for item in model.feature_importances[:10]
    ) or " No feature importances available."

    warnings_str = "\n".join(
        f" - {warning}" for warning in profile.quality_warnings
    ) or " No warnings."

    limitations_str = "\n".join(
        f" - {limitation}" for limitation in scoped.limitations
    ) or " No limitations specified."

    profile_summary_path = profile.profile_report_path or "None"
    n_train = int(etl.n_rows_after_cleaning * 0.8)

    return {
        "problem_statement": scoped.problem_statement,
        "target_column": scoped.target_column,
        "task_type": scoped.task_type.value,
        "success_metric": scoped.success_metric,
        "algorithm": model.algorithm,
        "cv_mean": f"{model.cv_mean:.4f}",
        "cv_std": f"{model.cv_std:.4f}",
        "test_score": f"{model.test_score:.4f}",
        "primary_metric": model.primary_metric,
        "n_train_rows": str(n_train),
        "n_features": str(len(etl.feature_columns)),
        "feature_importances": importances_str,
        "quality_warnings": warnings_str,
        "limitations": limitations_str,
        "pipeline_path": etl.pipeline_path,
        "model_path": model.model_path,
        "processed_data_path": etl.processed_data_path,
        "profile_summary_path": profile_summary_path,
        "dashboard_requests": _format_dashboard_requests(dashboard_requests),
    }


def _validate_dashboard_code(code: str, dashboard_requests: list[str] | None = None) -> list[str]:
    """
    Validate generated Streamlit code.

    Returns a list of validation errors. Empty list means valid.

    Checks:
      1. ast.parse() — syntax validity
      2. Required section presence — content completeness
      3. Optional reviewer-request checks for a few concrete asks
    """
    errors = []

    try:
        ast.parse(code)
    except SyntaxError as exc:
        errors.append(f"SyntaxError: {exc}")
        return errors

    code_lower = code.lower()
    for section in _REQUIRED_SECTIONS:
        if section.lower() not in code_lower:
            errors.append(f"Missing required section: '{section}'")

    requests = [request.lower() for request in (dashboard_requests or [])]
    wants_confusion_matrix = any("confusion matrix" in request for request in requests)
    if wants_confusion_matrix:
        cm_tokens = [
            "confusion matrix",
            "confusion_matrix",
            "confusionmatrixdisplay",
            "px.imshow",
            "heatmap",
        ]
        if not any(token in code_lower for token in cm_tokens):
            errors.append(
                "Reviewer requested a confusion matrix, but no confusion-matrix section was detected."
            )

    return errors


def dashboard_node(state: AnalyticsState) -> dict:
    """
    Dashboard generation node.

    Builds a context from prior state, calls the LLM to generate Streamlit code,
    validates it, and writes it to disk.

    Returns:
        dict with key "dashboard_code" containing the generated Python code.

    Raises:
        ValueError: if the generated code fails validation after generation.
    """
    console.rule("[bold]Node 5: Dashboard Generation[/bold]")

    context = _build_context(state)
    console.print(f"  Context built: {len(context)} fields")
    console.print(
        f"  Algorithm: {context['algorithm']} | "
        f"Score: {context['test_score']} | "
        f"Features: {context['n_features']}"
    )
    if state.get("dashboard_requests"):
        console.print(
            f"  Reviewer dashboard requests: {len(state['dashboard_requests'])}"
        )

    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = PromptTemplate(
        input_variables=list(context.keys()),
        template=template,
    )

    llm = get_llm(temperature=0.1)
    chain = prompt | llm

    console.print("  Calling LLM for dashboard generation...")
    response = chain.invoke(context)

    raw_code = response.content if hasattr(response, "content") else str(response)
    raw_code = raw_code.strip()
    if raw_code.startswith("```"):
        lines = raw_code.split("\n")
        raw_code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    console.print(f"  Generated {len(raw_code.splitlines())} lines of code")

    errors = _validate_dashboard_code(
        raw_code,
        dashboard_requests=list(state.get("dashboard_requests", [])),
    )
    if errors:
        error_summary = "; ".join(errors)
        console.print(f"  [red]Validation failed:[/red] {error_summary}")
        raise ValueError(
            f"Generated dashboard code failed validation: {error_summary}"
        )

    console.print("  [green]Validation passed.[/green]")

    dashboard_path = settings.dashboards_dir / "dashboard.py"
    dashboard_path.write_text(raw_code, encoding="utf-8")
    console.print(f"  Dashboard saved to: {dashboard_path}")
    console.print("  [green]Dashboard generation complete.[/green]")

    return {"dashboard_code": raw_code}
