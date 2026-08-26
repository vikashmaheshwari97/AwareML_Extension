from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    llm_enabled: bool = os.getenv("AWAREML_LLM_ENABLED", "false").lower() == "true"
    codecarbon_enabled: bool = os.getenv("AWAREML_CODECARBON_ENABLED", "false").lower() == "true"
    country_iso: str = os.getenv("AWAREML_COUNTRY_ISO", "EST")
    study_db: Path = Path(os.getenv("AWAREML_STUDY_DB", "artifacts/awareml_studies.sqlite3"))


settings = Settings()
