import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    gemini_model: str = "gemini-2.0-flash"


def load_settings() -> Settings:
    """
    Loads configuration from environment variables.

    Expected:
      - GOOGLE_API_KEY
      - optional GEMINI_MODEL
    """
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Create a .env file (copy env.example -> .env) "
            "and set GOOGLE_API_KEY."
        )
    if api_key.lower() in {"your_key_here", "changeme", "replace_me"}:
        raise RuntimeError(
            "GOOGLE_API_KEY is still a placeholder. Open your .env and paste a real Gemini API key "
            "(from https://aistudio.google.com/app/apikey)."
        )

    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"
    return Settings(google_api_key=api_key, gemini_model=model)

