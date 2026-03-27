# AI Analytics Delivery Orchestrator

A LangGraph state machine that turns a business brief and a CSV dataset into a reproducible analytics delivery.

## What it produces

Given a plain-English brief and a tabular dataset, the system delivers:

- **Scoped problem definition** — target variable, success metric, constraints (LLM, Pydantic-validated)
- **Data profile** — null rates, distributions, correlations (deterministic, ydata-profiling)
- **ETL pipeline** — sklearn Pipeline serialised to disk (deterministic, reproducible)
- **Baseline model** — trained and cross-validated, with metrics stored in state (deterministic)
- **Streamlit dashboard** — generated app with key metrics and charts (LLM codegen)
- **Rubric review** — every output scored 0–1 across 5 dimensions (LLM, structured output)
- **QA checklist** — deterministic assertions + hallucination flag scan (pytest + LLM)
- **Final delivery summary** — full report with limitations acknowledged

## Design principles

- **LLM for judgment, deterministic code for facts** — data statistics never come from an LLM
- **Every LLM output is Pydantic-validated** — no raw strings passed between nodes
- **Human-in-the-loop is a real interrupt** — not a simulated approval
- **Rubric-based evaluation** — quantifiable scores, not vibes
- **Checkpointing** — LangGraph MemorySaver resumes from the last completed node on failure

## Setup

```bash
cp .env.example .env        # add your OPENAI_API_KEY
pip install -e ".[dev]"     # install all dependencies
pytest                      # run the test suite
python main.py              # run the full graph
```

## Architecture

```
START → problem_scoping → data_profiling → etl_pipeline →
        baseline_model → dashboard_generation →
        rubric_review →(low score retry)→ dashboard_generation
               ↓
        human_review →(revise)→ problem_scoping
               ↓
        qa_checklist → delivery_package → END
```

## Stack

| Layer | Tool | Reason |
|---|---|---|
| Orchestration | LangGraph | State machine with checkpointing and human interrupt |
| LLM calls | LangChain + OpenAI | Structured output parsing via Pydantic |
| Data profiling | ydata-profiling | Deterministic statistics |
| ML | scikit-learn | Reproducible baseline model and pipeline |
| Validation | Pydantic v2 | Schema enforcement at every node boundary |
| Testing | pytest | Golden test cases and consistency assertions |
| Output | Streamlit | Lightweight generated dashboard |
