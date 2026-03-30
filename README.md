# AI Analytics Delivery Orchestrator

A LangGraph-based orchestration system that turns a business analytics request into a reproducible mini-delivery: scoped problem, validated data pipeline, baseline model, dashboard app, review report, and QA checklist.

This project is designed to show how AI can support a realistic analytics workflow end to end, not just generate isolated code or one-off notebook output. It combines LLM judgment where interpretation is useful with deterministic Python and machine learning code where correctness, traceability, and reproducibility matter most.

---

## Overview

Many analytics projects fail not because the team cannot train a model, but because the overall workflow is weak:

* the business problem is vague
* scope is undocumented
* preprocessing choices are not reproducible
* outputs are hard to review
* stakeholder feedback is disconnected from execution
* iteration means rerunning everything manually

This project addresses that gap by treating analytics delivery as an orchestration problem.

A user provides:

* a business brief
* a tabular dataset

The system produces:

* a structured problem definition
* a deterministic data profile
* an ETL / preprocessing pipeline
* a baseline machine learning model
* a generated Streamlit dashboard
* a rubric-based review
* a human review checkpoint
* a QA checklist
* a final delivery summary

---

## Core capabilities

### 1) Structured problem scoping

The system translates a plain-English business brief into a validated analytics problem definition, including:

* target variable
* task type
* success metric
* business constraints
* exclusions
* assumptions
* limitations
* model recommendation
* feature selection strategy

### 2) Deterministic data profiling

The profiling step computes concrete diagnostics from the dataset, such as:

* schema and column types
* missingness
* distributions
* summary statistics
* correlation-oriented signals

### 3) Reproducible ETL / preprocessing

The ETL node builds a deterministic preprocessing pipeline using standard Python / sklearn tooling.

This includes:

* duplicate handling
* feature exclusion
* numeric / categorical preprocessing
* serialized pipeline artifacts
* feature selection when requested

### 4) Baseline model training

The modeling step trains a baseline ML model using deterministic code and stores performance outputs for downstream use.

### 5) Dashboard generation

The system generates a lightweight Streamlit dashboard that presents the key outputs for stakeholder review.

### 6) Rubric-based review

Generated outputs are reviewed against a structured rubric rather than vague free-form comments alone.

### 7) Human-in-the-loop review

The workflow pauses for explicit human feedback before packaging final delivery artifacts.

### 8) Actionable review feedback

Human feedback can modify workflow state and reroute execution to the right downstream stage instead of always restarting from scratch.

Examples:

* “try fewer features” → rerun from ETL
* “drop customer_id” → rerun from ETL
* “use random forest” → rerun from modeling
* “optimize for F1” → rerun from modeling
* “add confusion matrix” → rerun from dashboard only
* “the target is wrong” → rerun from scoping

This is one of the main design strengths of the project.

---

## End-to-end workflow

```text
Business brief + CSV
        ↓
Problem scoping
        ↓
Data profiling
        ↓
ETL / preprocessing
        ↓
Baseline modeling
        ↓
Dashboard generation
        ↓
Rubric review
        ↓
Human review
        ↓
QA checklist
        ↓
Final delivery package
```

---

## Example use case

The current example use case is a **credit risk** analytics problem based on the **Give Me Some Credit** dataset.

Example business brief:

> The credit team needs to score incoming loan applications. Build a model that flags high-risk borrowers based on financial history and utilisation patterns, so analysts can prioritise manual review of borderline cases.

Typical framing:

* **domain:** credit risk / financial services
* **task:** binary classification
* **metric:** ROC-AUC
* **goal:** help analysts prioritize manual review for riskier applicants

---

## Design principles

### LLM for judgment, deterministic code for facts

The LLM helps where interpretation or synthesis is needed. It does not invent dataset statistics or evaluation metrics.

### Typed state between nodes

Node outputs are validated through structured schemas rather than loosely passing strings through the graph.

### Real interrupt for human review

Human-in-the-loop is implemented as an actual checkpoint and decision point.

### Selective recomputation

The system tries to rerun only the stages affected by feedback.

### Reviewability and QA

The goal is not just to produce output, but to make that output inspectable and easier to trust.

---

## Architecture

### High-level node responsibilities

| Node           | Responsibility                                               |
| -------------- | ------------------------------------------------------------ |
| `scoping`      | Convert a business brief into a structured analytics problem |
| `profiling`    | Compute deterministic data diagnostics                       |
| `etl`          | Build preprocessing pipeline and feature set                 |
| `model`        | Train and evaluate a baseline model                          |
| `dashboard`    | Generate a lightweight stakeholder-facing app                |
| `review`       | Score artifacts against a rubric                             |
| `human_review` | Pause for human approval or revision                         |
| `qa`           | Validate artifacts and flag issues                           |
| `delivery`     | Produce final packaged summary                               |

### Orchestration behavior

The system is modeled as a LangGraph workflow with checkpointing and stateful transitions. This allows:

* explicit node boundaries
* typed intermediate artifacts
* retries
* interrupts
* partial reruns after feedback

---

## Human-in-the-loop behavior

A major goal of this project is to make review comments operational.

### Supported feedback categories

#### Scoping feedback

Use when the business framing is wrong.

Examples:

* “The target is wrong. Predict churn, not default.”
* “This should be regression, not classification.”
* “Use recall as the primary business metric.”
* “Rewrite the problem for a fraud team audience.”

#### ETL / feature feedback

Use when data prep or the feature set should change.

Examples:

* “Try fewer features.”
* “Use correlation filtering.”
* “Drop `customer_id`.”
* “Exclude likely leakage columns.”
* “Use a simpler feature set for interpretability.”

#### Model feedback

Use when the model or evaluation objective should change.

Examples:

* “Use random forest.”
* “Try logistic regression.”
* “Optimize for F1.”
* “Use ROC-AUC as the main score.”

#### Dashboard / presentation feedback

Use when the analysis is acceptable but the presentation should improve.

Examples:

* “Add a confusion matrix.”
* “Show feature importance.”
* “Add an executive summary.”
* “Make the dashboard more business-friendly.”
* “Surface model limitations clearly.”

### Best practice for reviewers

Concrete feedback works best.

Good examples:

* “Try fewer features for interpretability.”
* “Drop `customer_id` and `application_id`.”
* “Use random forest and optimize for ROC-AUC.”
* “Add a confusion matrix and a business summary.”
* “The target is wrong; re-scope the problem.”

Weak examples:

* “Improve this.”
* “Make it better.”
* “Redo it.”

---

## Tech stack

| Layer           | Tools                                               |
| --------------- | --------------------------------------------------- |
| Orchestration   | LangGraph                                           |
| LLM integration | LangChain + OpenAI                                  |
| Validation      | Pydantic                                            |
| Data processing | pandas                                              |
| Profiling       | ydata-profiling / deterministic profiling utilities |
| Modeling        | scikit-learn                                        |
| Serialization   | joblib                                              |
| Dashboard       | Streamlit                                           |
| Testing         | pytest                                              |

---

## Repository structure

```text
ai-analytics-delivery-orchestrator/
├── orchestrator/
│   ├── nodes/
│   ├── prompts/
│   ├── graph.py
│   ├── state.py
│   └── ...
├── tests/
├── data/
├── main.py
├── pyproject.toml
└── README.md
```

---

## Setup

### 1) Clone the repository

```bash
git clone https://github.com/IvanSolanes/ai-analytics-delivery-orchestrator.git
cd ai-analytics-delivery-orchestrator
```

### 2) Create and activate a virtual environment

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3) Configure environment variables

Create a `.env` file from the example and add your API key:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 4) Install dependencies

```bash
pip install -e ".[dev]"
```

---

## Dataset

This repo uses the Kaggle **Give Me Some Credit** dataset for the example workflow.

### Expected local file setup

Download the dataset and place the CSV in:

```text
data/credit_risk_sample.csv
```

If you are using the Kaggle training file directly, rename:

```text
cs-training.csv
```

to:

```text
credit_risk_sample.csv
```

---

## Running the project

### Run tests

```bash
pytest
```

### Run the full workflow

```bash
python main.py
```

---

## What the system produces

Depending on configuration and run path, the workflow is designed to generate artifacts such as:

* **scoped problem definition**
* **data profile**
* **serialized ETL pipeline**
* **baseline model outputs**
* **dashboard app code**
* **review report**
* **QA checklist**
* **final delivery summary**

This makes the system easier to demonstrate than a notebook-only workflow because the outputs are packaged as a small but coherent delivery.

---

## Example demo story

A good live demo flow is:

1. Start with a short business brief and CSV
2. Show the scoped problem output
3. Show profiling and ETL artifacts
4. Show baseline model metrics
5. Show generated dashboard
6. Pause at human review
7. Enter feedback such as:

   * “try fewer features”
   * “use random forest”
   * “add confusion matrix”
8. Show that the system reroutes to the correct stage
9. Show the updated delivery package

This demonstrates orchestration, typed state, and actionable HITL behavior in a very visible way.

---

## Testing strategy

The best way to test this kind of project is in layers:

### Unit tests

For helpers and deterministic utility functions.

### Node-level tests

For ETL, modeling, feedback parsing, and state mutation logic.

### Graph / routing tests

For verifying reroute behavior after review feedback.

### Golden-path tests

For representative end-to-end scenarios.

### Regression tests

For features like:

* feature selection
* selective recomputation
* dashboard-only reruns
* state compatibility across updates

---

## Key engineering choices

### Why not let the LLM do everything?

Because analytics delivery requires reproducibility. Dataset statistics, transformations, and model metrics should come from deterministic code.

### Why typed state?

Because orchestration becomes brittle when intermediate node outputs are unstructured.

### Why baseline models first?

Because the goal is a robust, reviewable system flow, not leaderboard optimization.

### Why human review in the middle?

Because many analytics workflows need judgment after seeing the first delivery draft, not only at the end.

### Why selective reruns?

Because in real workflows, a reviewer often wants one part changed, not a full restart.

---

## Limitations

This is intentionally a focused orchestration demo, not a full production platform.

Current limitations may include:

* baseline-model orientation rather than broad model search
* dashboard quality depending on prompt quality and available context
* reviewer feedback parser working best with concrete instructions
* limited domain coverage outside tabular analytics workflows
* lightweight artifact management compared with production-grade ML platforms

---

## Quick summary

**AI Analytics Delivery Orchestrator** is a LangGraph-based system for turning a business analytics request into a structured, reproducible, reviewable mini-delivery with deterministic ML components, generated stakeholder artifacts, and actionable human feedback.
