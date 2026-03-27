"""
nodes/scoping.py
----------------
Node 1: Problem Scoping

Responsibility:
  Read the business brief and dataset column names, call the LLM,
  and return a validated ScopedProblem written into state.

Why LLM here?
  Interpreting a free-text business brief — identifying the target variable,
  inferring the task type, anticipating limitations — requires language
  understanding. This is exactly where the LLM adds value.

Why NOT LLM for anything else in this node?
  Column names come from pandas (deterministic). The node does not ask the
  LLM to describe the data or compute statistics — that is the profiling
  node's job.

Inputs from state:
  - business_brief: str
  - dataset_path: str

Outputs to state:
  - scoped_problem: ScopedProblem
"""

from pathlib import Path

import pandas as pd
from langchain_core.prompts import PromptTemplate
from rich.console import Console

# get_llm() returns the correct provider (openai or huggingface) based on
# the LLM_PROVIDER environment variable. No node imports ChatOpenAI directly.
from orchestrator.config import get_llm
from orchestrator.state import AnalyticsState, ScopedProblem

console = Console()

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "scoping.txt"


def _load_prompt() -> PromptTemplate:
    """Load the scoping prompt template from disk."""
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    return PromptTemplate(
        input_variables=["business_brief", "column_names"],
        template=template,
    )


# ---------------------------------------------------------------------------
# Column extraction — deterministic, never from LLM
# ---------------------------------------------------------------------------

def _extract_column_names(dataset_path: str) -> list[str]:
    """
    Read only the header row of the CSV and return column names.
    nrows=0 loads zero data rows — just the schema. Fast and memory-safe.
    """
    df = pd.read_csv(dataset_path, nrows=0)
    return df.columns.tolist()


# ---------------------------------------------------------------------------
# The node function
# ---------------------------------------------------------------------------

def scoping_node(state: AnalyticsState) -> dict:
    """
    Problem scoping node.

    Reads the business brief and dataset column names, calls the LLM
    with structured output parsing, and returns a validated ScopedProblem.

    Returns:
        dict with key "scoped_problem" containing a ScopedProblem instance.

    Raises:
        ValueError: if the LLM returns a target_column that does not exist
                    in the dataset, or if the API key is missing.
        FileNotFoundError: if dataset_path does not exist.
    """
    console.rule("[bold]Node 1: Problem Scoping[/bold]")

    # --- Step 1: Extract column names (deterministic) ---
    dataset_path = state["dataset_path"]
    console.print(f"  Reading column names from: {dataset_path}")

    column_names = _extract_column_names(dataset_path)
    console.print(f"  Found {len(column_names)} columns: {column_names}")

    # --- Step 2: Build the LLM chain ---
    # get_llm() reads LLM_PROVIDER from .env and returns the right client.
    # .with_structured_output(ScopedProblem) instructs the model to return
    # JSON matching our Pydantic schema exactly.
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(ScopedProblem)
    prompt = _load_prompt()
    chain = prompt | structured_llm

    # --- Step 3: Call the LLM ---
    console.print("  Calling LLM for problem scoping...")

    scoped_problem: ScopedProblem = chain.invoke({
        "business_brief": state["business_brief"],
        "column_names": ", ".join(column_names),
    })

    # --- Step 4: Validate target column exists (deterministic guard) ---
    # .with_structured_output() validates schema shape.
    # We additionally check that the column value actually exists in the CSV.
    if scoped_problem.target_column not in column_names:
        raise ValueError(
            f"LLM returned target_column='{scoped_problem.target_column}' "
            f"but that column does not exist in the dataset. "
            f"Available columns: {column_names}"
        )

    # --- Step 5: Log result ---
    console.print(f"  [green]Target column:[/green] {scoped_problem.target_column}")
    console.print(f"  [green]Task type:[/green]     {scoped_problem.task_type.value}")
    console.print(f"  [green]Metric:[/green]        {scoped_problem.success_metric}")
    console.print(f"  [green]Limitations:[/green]   {len(scoped_problem.limitations)} identified")
    console.print("  [green]Scoping complete.[/green]")

    return {"scoped_problem": scoped_problem}
