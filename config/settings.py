import os
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()


class AgentConfig(BaseModel):
    """Typed configuration for the standalone DecideIt AI Agent."""

    llm_provider: Literal["gemini", "groq", "mock"] = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").lower()  # type: ignore
    )
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or None
    )
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
    )
    groq_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY") or None
    )
    groq_model: str = Field(
        default_factory=lambda: os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
    )
    llm_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        ge=1,
        le=300,
    )
    llm_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")),
        ge=0,
        le=10,
    )
    agent_max_iterations: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_ITERATIONS", "12")),
        ge=1,
        le=50,
    )
    agent_max_tool_calls: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_TOOL_CALLS", "20")),
        ge=1,
        le=100,
    )
    allowed_docs_dir: str = Field(
        default_factory=lambda: os.getenv("ALLOWED_DOCS_DIR", ".")
    )
    max_document_size_mb: int = Field(
        default_factory=lambda: int(os.getenv("MAX_DOCUMENT_SIZE_MB", "15")),
        ge=1,
        le=100,
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_provider(cls, v: str) -> str:
        val = str(v).strip().lower()
        if val not in {"gemini", "groq", "mock"}:
            raise ValueError(f"Unsupported LLM_PROVIDER: '{val}'. Must be 'gemini', 'groq', or 'mock'.")
        return val

    def validate_provider_credentials(self) -> None:
        """Validate required API keys only when the live provider is invoked."""
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when LLM_PROVIDER is set to 'gemini'."
            )
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required when LLM_PROVIDER is set to 'groq'."
            )

    def safe_dict(self) -> dict:
        """Return configuration dictionary with secrets redacted."""
        data = self.model_dump()
        if data.get("gemini_api_key"):
            data["gemini_api_key"] = "***REDACTED***"
        if data.get("groq_api_key"):
            data["groq_api_key"] = "***REDACTED***"
        return data