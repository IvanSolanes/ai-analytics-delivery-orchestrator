"""
test_nodes.py
-------------
Tests for individual graph nodes.

Testing philosophy for nodes:
  - Mock get_llm() — we test node logic, not the LLM provider
  - Use real file I/O for CSV column extraction (deterministic)
  - Every test is independent — no shared mutable state
  - Golden cases verify the contract between nodes

What we test here:
  Scoping node:
    1. Column extraction from CSV works correctly
    2. Node writes scoped_problem to state
    3. Node raises ValueError if LLM returns a non-existent target column
    4. Golden case: credit risk target column and metric are correct
    5. Golden case: limitations are non-empty
    6. Node only writes to its own state field

  Provider switching:
    7. get_llm raises ValueError for unsupported provider
    8. get_llm raises ValueError when OpenAI key is missing
    9. get_llm raises ValueError when HuggingFace key is missing
"""

import csv
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.evaluation.golden_cases import (
    CREDIT_RISK_BRIEF,
    CREDIT_RISK_COLUMNS,
    CREDIT_RISK_EXPECTED_SCOPE,
)
from orchestrator.nodes.scoping import _extract_column_names, scoping_node
from orchestrator.state import ScopedProblem, TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_csv(columns: list[str], n_rows: int = 3) -> str:
    """
    Create a temporary CSV with the given columns and dummy data.
    Returns the file path — caller is responsible for cleanup.
    """
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    writer = csv.writer(f)
    writer.writerow(columns)
    for _ in range(n_rows):
        writer.writerow(["0"] * len(columns))
    f.close()
    return f.name


def _make_scoping_state(path: str) -> dict:
    """Return a minimal valid state dict for the scoping node."""
    return {
        "business_brief": CREDIT_RISK_BRIEF,
        "dataset_path": path,
        "retry_count": 0,
    }


# ---------------------------------------------------------------------------
# Tests: _extract_column_names
# ---------------------------------------------------------------------------

def test_extract_column_names_returns_correct_columns():
    """Column names from CSV header must match exactly."""
    columns = ["age", "income", "defaulted"]
    path = _make_temp_csv(columns)
    try:
        assert _extract_column_names(path) == columns
    finally:
        os.unlink(path)


def test_extract_column_names_handles_credit_risk_columns():
    """All Give Me Some Credit columns are extracted correctly."""
    path = _make_temp_csv(CREDIT_RISK_COLUMNS)
    try:
        result = _extract_column_names(path)
        assert result == CREDIT_RISK_COLUMNS
        assert "SeriousDlqin2yrs" in result
        assert "MonthlyIncome" in result
    finally:
        os.unlink(path)


def test_extract_column_names_raises_on_missing_file():
    """FileNotFoundError is raised for a non-existent path."""
    with pytest.raises(Exception):
        _extract_column_names("/nonexistent/path/file.csv")


# ---------------------------------------------------------------------------
# Tests: scoping_node (get_llm mocked)
#
# Why mock get_llm and not ChatOpenAI directly?
# The node now calls get_llm() — it does not know or care which provider
# is active. By mocking get_llm we test the node logic independently of
# which provider is configured in .env. This is the correct abstraction.
# ---------------------------------------------------------------------------

@patch("orchestrator.nodes.scoping.get_llm")
def test_scoping_node_writes_scoped_problem_to_state(mock_get_llm):
    """scoping_node must return a dict containing a valid ScopedProblem."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = CREDIT_RISK_EXPECTED_SCOPE
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    mock_get_llm.return_value = mock_llm

    path = _make_temp_csv(CREDIT_RISK_COLUMNS)
    try:
        with patch("orchestrator.nodes.scoping.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = lambda self, other: mock_chain
            result = scoping_node(_make_scoping_state(path))

        assert "scoped_problem" in result
        assert isinstance(result["scoped_problem"], ScopedProblem)
    finally:
        os.unlink(path)


@patch("orchestrator.nodes.scoping.get_llm")
def test_scoping_node_validates_target_column_exists(mock_get_llm):
    """
    If the LLM returns a target_column not in the CSV,
    scoping_node must raise ValueError before writing to state.
    """
    bad_scope = ScopedProblem(
        target_column="nonexistent_column",
        task_type=TaskType.BINARY_CLASSIFICATION,
        success_metric="roc_auc",
        problem_statement="Test.",
        limitations=["Test limitation."],
    )
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = bad_scope
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    mock_get_llm.return_value = mock_llm

    path = _make_temp_csv(CREDIT_RISK_COLUMNS)
    try:
        with patch("orchestrator.nodes.scoping.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = lambda self, other: mock_chain
            with pytest.raises(ValueError, match="does not exist in the dataset"):
                scoping_node(_make_scoping_state(path))
    finally:
        os.unlink(path)


@patch("orchestrator.nodes.scoping.get_llm")
def test_scoping_node_golden_case_target_column(mock_get_llm):
    """Golden case: target must be SeriousDlqin2yrs, metric must be roc_auc."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = CREDIT_RISK_EXPECTED_SCOPE
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    mock_get_llm.return_value = mock_llm

    path = _make_temp_csv(CREDIT_RISK_COLUMNS)
    try:
        with patch("orchestrator.nodes.scoping.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = lambda self, other: mock_chain
            result = scoping_node(_make_scoping_state(path))

        assert result["scoped_problem"].target_column == "SeriousDlqin2yrs"
        assert result["scoped_problem"].task_type == TaskType.BINARY_CLASSIFICATION
        assert result["scoped_problem"].success_metric == "roc_auc"
    finally:
        os.unlink(path)


@patch("orchestrator.nodes.scoping.get_llm")
def test_scoping_node_golden_case_limitations_not_empty(mock_get_llm):
    """Golden case: limitations must be a non-empty list."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = CREDIT_RISK_EXPECTED_SCOPE
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    mock_get_llm.return_value = mock_llm

    path = _make_temp_csv(CREDIT_RISK_COLUMNS)
    try:
        with patch("orchestrator.nodes.scoping.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = lambda self, other: mock_chain
            result = scoping_node(_make_scoping_state(path))

        assert len(result["scoped_problem"].limitations) > 0
    finally:
        os.unlink(path)


@patch("orchestrator.nodes.scoping.get_llm")
def test_scoping_node_only_writes_scoped_problem(mock_get_llm):
    """Node must return only its own state field — nothing else."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = CREDIT_RISK_EXPECTED_SCOPE
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    mock_get_llm.return_value = mock_llm

    path = _make_temp_csv(CREDIT_RISK_COLUMNS)
    try:
        with patch("orchestrator.nodes.scoping.PromptTemplate") as mock_pt:
            mock_pt.return_value.__or__ = lambda self, other: mock_chain
            result = scoping_node(_make_scoping_state(path))

        assert list(result.keys()) == ["scoped_problem"]
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: provider switching via get_llm()
# These tests verify config.py behaviour — isolated from node logic.
# ---------------------------------------------------------------------------

def test_get_llm_raises_for_unsupported_provider():
    """get_llm raises ValueError for an unrecognised provider name."""
    from orchestrator.config import Settings, get_llm
    with patch("orchestrator.config.settings") as mock_settings:
        mock_settings.llm_provider = "anthropic"
        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            get_llm()


def test_get_llm_raises_when_openai_key_missing():
    """get_llm raises ValueError when OpenAI is selected but key is empty."""
    from orchestrator.config import get_llm
    with patch("orchestrator.config.settings") as mock_settings:
        mock_settings.llm_provider = "openai"
        mock_settings.openai_api_key = ""
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            get_llm()


def test_get_llm_raises_when_huggingface_key_missing():
    """get_llm raises ValueError when HuggingFace is selected but key is empty."""
    from orchestrator.config import get_llm
    with patch("orchestrator.config.settings") as mock_settings:
        mock_settings.llm_provider = "huggingface"
        mock_settings.huggingface_api_key = ""
        with pytest.raises(ValueError, match="HUGGINGFACE_API_KEY is not set"):
            get_llm()
