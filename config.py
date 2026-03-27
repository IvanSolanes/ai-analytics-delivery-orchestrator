"""
config.py
---------
Central settings loaded once at startup from environment variables.

Why a dedicated config module?
  - Every node imports settings from one place — no scattered os.getenv() calls
  - Easy to mock in tests by overriding Settings fields
  - Makes all tuneable parameters visible in one screen
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "./outputs"))
    review_score_threshold: float = float(os.getenv("REVIEW_SCORE_THRESHOLD", "0.6"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))

    # Derived paths — created at runtime if they don't exist
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
