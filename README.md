# AI Analytics Delivery Orchestrator

A LangGraph state machine that turns a business brief and a CSV dataset
into a reproducible analytics delivery.

## Problem

**Domain:** Credit risk — financial services  
**Dataset:** Give Me Some Credit (Kaggle)  
**Task:** Binary classification — predict whether a borrower will experience
financial distress (90+ days past due) within two years  
**Metric:** ROC-AUC  

**Business brief:**  
"The credit team needs to score incoming loan applications. Build a model
that flags high-risk borrowers based on financial history and utilisation
patterns, so analysts can prioritise manual review of borderline cases."

## What it produces

Given a plain-English brief and the dataset CSV, the system delivers:

- Scoped problem definition — target variable, success metric, constraints (LLM, Pydantic-validated)
- Data profile — null rates, distributions, correlations (deterministic, pandas)
- ETL pipeline — sklearn Pipeline serialised to disk (deterministic, reproducible)
- Baseline model — trained and cross-validated, with metrics stored in state (deterministic)
- Streamlit dashboard — generated app with key metrics and charts (LLM codegen)
- Rubric review — every output scored 0–1 across 5 dimensions (LLM, structured output)
- QA checklist — deterministic assertions + hallucination flag scan (pytest + LLM)
- Final delivery summary — full report with acknowledged limitations

## Design principles

- LLM for judgment, deterministic code for facts — data statistics never come from an LLM
- Every LLM output is Pydantic-validated — no raw strings passed between nodes
- Human-in-the-loop is a real interrupt — not a simulated approval
- Rubric-based evaluation — quantifiable scores, not vibes
- Checkpointing — LangGraph MemorySaver resumes from the last completed node on failure

## Setup
```bash
cp .env.example .env        # add your OPENAI_API_KEY
pip install -e ".[dev]"     # install all dependencies
pytest                      # run the test suite
python main.py              # run the full graph
```

## Dataset

Download from Kaggle:
https://www.kaggle.com/c/GiveMeSomeCredit/data

Place `cs-training.csv` in the `data/` folder and rename it `credit_risk_sample.csv`.

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
| Data profiling | pandas | Deterministic statistics — null rates, distributions, correlations |
| ML | scikit-learn | Reproducible baseline model and pipeline |
| Validation | Pydantic v2 | Schema enforcement at every node boundary |
| Testing | pytest | Golden test cases and consistency assertions |
| Output | Streamlit | Lightweight generated dashboard |