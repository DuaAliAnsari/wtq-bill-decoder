"""Runtime configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ALLOWED_MODELS = {
    "anthropic": {"claude-haiku-4-5"},
    "google": {"gemini-3.5-flash-lite", "gemini-3.1-flash-lite"},
    "openai": {"gpt-4.1-mini", "gpt-5-mini"},
}


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or not allowed by the competition rules."""


@dataclass(frozen=True)
class Config:
    provider: str
    model: str
    api_key: str
    temperature: float = 0.0
    seed: int = 7
    requests_per_minute: float = 12.0
    max_retries: int = 6
    timeout: int = 120
    cache_dir: Path = Path(".cache")

    @classmethod
    def from_env(cls, require_key: bool = True) -> "Config":
        load_dotenv()
        provider = os.getenv("MODEL_PROVIDER", "google").strip().lower()
        model = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite").strip()
        # Accept the key under API_KEY (documented name) or the provider-specific names.
        api_key = (
            os.getenv("API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip().strip('"').strip("'")

        if provider not in ALLOWED_MODELS:
            raise ConfigError(f"MODEL_PROVIDER must be one of {sorted(ALLOWED_MODELS)}, got '{provider}'.")
        if model not in ALLOWED_MODELS[provider]:
            raise ConfigError(
                f"Model '{model}' is not on the competition's allowed list for {provider}: "
                f"{sorted(ALLOWED_MODELS[provider])}. Using any other model leads to disqualification."
            )
        if require_key and (not api_key or api_key == "your-api-key-here"):
            raise ConfigError("API_KEY is not set. Copy .env.example to .env and paste your key.")

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=float(os.getenv("TEMPERATURE", "0")),
            seed=int(os.getenv("SEED", "7")),
            requests_per_minute=float(os.getenv("REQUESTS_PER_MINUTE", "12")),
            max_retries=int(os.getenv("MAX_RETRIES", "6")),
            timeout=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),
            cache_dir=Path(os.getenv("CACHE_DIR", ".cache")),
        )
