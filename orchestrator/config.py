"""
config.py
---------
Central settings and LLM provider factory.

Two responsibilities:
  1. Settings — load all configuration from environment variables once at startup
  2. get_llm() — return the correct LLM instance based on LLM_PROVIDER

Why a factory function instead of a module-level LLM instance?
  - Every node calls get_llm() — if the provider changes in .env, every node
    picks it up automatically without code changes
  - Easy to mock in tests: patch 'orchestrator.config.get_llm'
  - Deferred instantiation — the LLM client is only created when a node
    actually needs it, not at import time

Provider trade-offs (documented here so every node author knows):
  openai:
    + Reliable structured output via .with_structured_output()
    + Large context window (128k tokens for gpt-4o)
    + Strong instruction following
    - Requires paid API key

  huggingface:
    + Free tier available via HuggingFace Inference API
    + Shows provider-agnostic design in the portfolio
    - Structured output (.with_structured_output()) reliability varies by model
    - Mistral-7B-Instruct is recommended — better instruction following than most
    - Free tier has rate limits and queuing delays
    - Smaller context windows (typically 8k–32k tokens)

Recommendation: use openai for demos and evaluation. Use huggingface to show
the provider-switching capability and for cost-free development runs.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- LLM Provider ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()

    # --- OpenAI ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # --- HuggingFace ---
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    # Mistral-7B-Instruct: best free option for instruction following + JSON output
    huggingface_model: str = os.getenv(
        "HUGGINGFACE_MODEL",
        "mistralai/Mistral-7B-Instruct-v0.3"
    )

    # --- Runtime ---
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "./outputs"))
    review_score_threshold: float = float(
        os.getenv("REVIEW_SCORE_THRESHOLD", "0.6")
    )
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))

    # --- Derived paths ---
    @property
    def profiles_dir(self) -> Path:
        p = self.output_dir / "profiles"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_dir(self) -> Path:
        p = self.output_dir / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def dashboards_dir(self) -> Path:
        p = self.output_dir / "dashboards"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()


def get_llm(temperature: float = 0):
    """
    LLM provider factory.

    Returns the correct LangChain LLM instance based on the LLM_PROVIDER
    environment variable. All nodes call this function — none instantiate
    a specific LLM class directly.

    Args:
        temperature: Controls randomness. Default 0 = deterministic output.
                     Scoping and review nodes use 0.
                     Dashboard generation may benefit from a small value (0.1).

    Returns:
        A LangChain BaseChatModel instance (OpenAI or HuggingFace).

    Raises:
        ValueError: if LLM_PROVIDER is set to an unsupported value.
        ValueError: if the required API key for the chosen provider is missing.

    Usage in nodes:
        from orchestrator.config import get_llm
        llm = get_llm()
        structured_llm = llm.with_structured_output(MySchema)

    Usage in tests:
        with patch("orchestrator.config.get_llm") as mock:
            mock.return_value = MagicMock()
    """
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Add it to your .env file."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
        )

    elif settings.llm_provider == "huggingface":
        if not settings.huggingface_api_key:
            raise ValueError(
                "LLM_PROVIDER=huggingface but HUGGINGFACE_API_KEY is not set. "
                "Get a free token at huggingface.co/settings/tokens "
                "and add it to your .env file."
            )
        from langchain_huggingface import HuggingFaceEndpoint
        return HuggingFaceEndpoint(
            repo_id=settings.huggingface_model,
            huggingfacehub_api_token=settings.huggingface_api_key,
            temperature=temperature,
            max_new_tokens=1024,
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{settings.llm_provider}'. "
            f"Supported values: 'openai', 'huggingface'."
        )