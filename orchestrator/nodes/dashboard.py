"""
nodes/dashboard.py
------------------
Node 5: Dashboard Generation

Responsibility:
  Take all prior node outputs, build a structured context, call the LLM,
  validate the generated Streamlit code, and write it to disk.

Why LLM here?
  Writing Streamlit boilerplate from a structured context is exactly where
  an LLM adds value — it handles layout, formatting, and code structure
  from a specification, freeing us from templating logic.

Why NOT LLM for the context?
  The numbers fed to the LLM — scores, feature names, warnings — come
  directly from state. The LLM never invents values. It only writes the
  code that displays them.

Validation layers:
  1. ast.parse() — rejects syntactically invalid Python immediately
  2. Content checks — verifies key sections are present in the output
  3. If validation fails, we raise a clear error that the review node
     can catch and trigger a retry

Inputs from state:
  - scoped_problem: ScopedProblem
  - data_profile: DataProfile
  - etl_artifacts: ETLArtifacts
  - model_results: ModelResults

Outputs to state:
  - dashboard_code: str  (raw Streamlit Python code)
"""

import ast
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from rich.console import Console

from orchestrator.config import get_llm, settings
from orchestrator.state import AnalyticsState

console = Console()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "dashboard.txt"

# Sections that must appear in valid generated dashboard code
_REQUIRED_SECTIONS = [
    "import streamlit",
    "st.set_page_config",
    ".metric(",
    "plotly",
]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------
# Why build context explicitly instead of passing the full state?
# 1. The LLM only needs what it needs — no noise
# 2. Every value is stringified and formatted here — the prompt stays clean
# 3. This function is independently testable
# ---------------------------------------------------------------------------

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

    # Format feature importances as readable lines
    importances_str = "\n".join(
        f"  {item['feature']}: {item['importance']:.4f}"
        for item in model.feature_importances[:10]
    ) or "  No feature importances available."

    # Format quality warnings
    warnings_str = "\n".join(
        f"  - {w}" for w in profile.quality_warnings
    ) or "  No warnings."

    # Format limitations
    limitations_str = "\n".join(
        f"  - {lim}" for lim in scoped.limitations
    ) or "  No limitations specified."

    # Profile summary path for the dashboard to load
    profile_summary_path = profile.profile_report_path or "None"

    # Number of training rows from the splits
    n_train = int(len(etl.feature_columns) and
                  model.cv_mean > 0 and
                  etl.n_rows_after_cleaning * 0.8) or etl.n_rows_after_cleaning

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
        "n_train_rows": str(etl.n_rows_after_cleaning),
        "n_features": str(len(etl.feature_columns)),
        "feature_importances": importances_str,
        "quality_warnings": warnings_str,
        "limitations": limitations_str,
        "pipeline_path": etl.pipeline_path,
        "model_path": model.model_path,
        "profile_summary_path": profile_summary_path,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_dashboard_code(code: str) -> list[str]:
    """
    Validate generated Streamlit code.

    Returns a list of validation errors. Empty list means valid.

    Two checks:
    1. ast.parse() — syntax validity
    2. Required section presence — content completeness
    """
    errors = []

    # Check 1: valid Python syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"SyntaxError: {e}")
        return errors  # no point checking sections if syntax is broken

    # Check 2: required sections present
    code_lower = code.lower()
    for section in _REQUIRED_SECTIONS:
        if section.lower() not in code_lower:
            errors.append(f"Missing required section: '{section}'")

    return errors


# ---------------------------------------------------------------------------
# The node function
# ---------------------------------------------------------------------------

def dashboard_node(state: AnalyticsState) -> dict:
    """
    Dashboard generation node.

    Builds a context from prior state, calls the LLM to generate
    Streamlit code, validates it, and writes it to disk.

    Returns:
        dict with key "dashboard_code" containing the generated Python code.

    Raises:
        ValueError: if the generated code fails validation after generation.
    """
    console.rule("[bold]Node 5: Dashboard Generation[/bold]")

    # --- Step 1: Build the prompt context ---
    context = _build_context(state)
    console.print(f"  Context built: {len(context)} fields")
    console.print(f"  Algorithm: {context['algorithm']} | "
                  f"Score: {context['test_score']} | "
                  f"Features: {context['n_features']}")

    # --- Step 2: Load prompt and build chain ---
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = PromptTemplate(
        input_variables=list(context.keys()),
        template=template,
    )

    # temperature=0.1: slight creativity for layout variety,
    # but not so high that it invents values or breaks syntax
    llm = get_llm(temperature=0.1)

    # For dashboard generation we want raw text output, not structured JSON.
    # We use the LLM directly without .with_structured_output() — the
    # output is Python code, not a Pydantic schema.
    chain = prompt | llm

    # --- Step 3: Call the LLM ---
    console.print("  Calling LLM for dashboard generation...")
    response = chain.invoke(context)

    # Extract content from the AIMessage response object
    raw_code = (
        response.content
        if hasattr(response, "content")
        else str(response)
    )

    # Strip any accidental markdown code fences the LLM may add
    # despite the prompt instruction — defensive cleaning
    raw_code = raw_code.strip()
    if raw_code.startswith("```"):
        lines = raw_code.split("\n")
        # Remove first line (```python or ```) and last line (```)
        raw_code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    console.print(f"  Generated {len(raw_code.splitlines())} lines of code")

    # --- Step 4: Validate ---
    errors = _validate_dashboard_code(raw_code)
    if errors:
        error_summary = "; ".join(errors)
        console.print(f"  [red]Validation failed:[/red] {error_summary}")
        raise ValueError(
            f"Generated dashboard code failed validation: {error_summary}"
        )

    console.print("  [green]Validation passed.[/green]")

    # --- Step 5: Write to disk ---
    dashboard_path = settings.dashboards_dir / "dashboard.py"
    dashboard_path.write_text(raw_code, encoding="utf-8")
    console.print(f"  Dashboard saved to: {dashboard_path}")
    console.print("  [green]Dashboard generation complete.[/green]")

    return {"dashboard_code": raw_code}
    