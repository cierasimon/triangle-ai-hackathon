"""Settings read from the environment (and the repo-root .env, if present).

Never log or print the values here; OPENROUTER_API_KEY is a secret.
"""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, existing env vars win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    llm_provider: str
    llm_model: str
    llm_fallback_model: str
    llm_vision_model: str
    public_base_url: str
    google_cloud_project: str = ""
    google_cloud_location: str = "global"
    gemini_vision_model: str = "gemini-3.5-flash-lite"

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider == "openrouter" and bool(self.openrouter_api_key)


def get_settings() -> Settings:
    if not os.environ.get("PAPERPHYSICS_NO_DOTENV"):
        _load_dotenv(REPO_ROOT / ".env")
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        llm_provider=os.environ.get("LLM_PROVIDER", "openrouter"),
        llm_model=os.environ.get("LLM_MODEL", "nvidia/nemotron-3.5-super-vl:free"),
        llm_fallback_model=os.environ.get("LLM_FALLBACK_MODEL", "openrouter/free"),
        llm_vision_model=os.environ.get("LLM_VISION_MODEL", "nvidia/nemotron-3.5-super-vl:free"),
        public_base_url=os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/"),
        google_cloud_project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        google_cloud_location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        gemini_vision_model=os.environ.get("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite"),
    )
