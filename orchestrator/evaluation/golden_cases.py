"""
evaluation/golden_cases.py
--------------------------
Golden test cases for the scoping node.

What is a golden test case?
  A fixed input with a known-good expected output. If the node's logic or
  prompt changes in a way that breaks the contract, the golden test catches it.

Why mock the LLM?
  Golden tests must be:
    - Fast (no API latency)
    - Free (no token cost)
    - Deterministic (same result every run, no network dependency)

  The LLM is mocked to return a pre-defined ScopedProblem. We are testing
  that the NODE handles that output correctly — validates the target column,
  writes to state, logs results — not that the LLM produces a specific answer.

  LLM output quality is evaluated separately in the rubric review node.
"""

from orchestrator.state import ScopedProblem, TaskType

# ---------------------------------------------------------------------------
# Golden case 1 — Credit Risk (our primary dataset)
# ---------------------------------------------------------------------------
# This is the expected ScopedProblem for the Give Me Some Credit dataset.
# Used to verify the node correctly handles the credit risk brief.

CREDIT_RISK_COLUMNS = [
    "Unnamed: 0",
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

CREDIT_RISK_BRIEF = (
    "The credit team needs to score incoming loan applications. "
    "Build a model that flags high-risk borrowers based on financial "
    "history and credit utilisation patterns, so analysts can prioritise "
    "manual review of borderline cases."
)

CREDIT_RISK_EXPECTED_SCOPE = ScopedProblem(
    target_column="SeriousDlqin2yrs",
    task_type=TaskType.BINARY_CLASSIFICATION,
    success_metric="roc_auc",
    problem_statement=(
        "Predict whether a borrower will experience financial distress "
        "(90+ days past due) within two years, using financial history "
        "and credit utilisation features."
    ),
    features_to_exclude=["Unnamed: 0"],
    known_constraints=[
        "No external data sources available.",
        "Model must be interpretable for credit analysts.",
    ],
    out_of_scope=[
        "Loan approval policy decisions.",
        "Real-time scoring pipeline.",
    ],
    limitations=[
        "Baseline model only — no hyperparameter tuning.",
        "MonthlyIncome and NumberOfDependents have significant missing values.",
        "Severe class imbalance (~93.3% non-default) will affect recall.",
        "Dataset may not reflect current economic conditions.",
    ],
)