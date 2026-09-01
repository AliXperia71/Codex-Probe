"""Configuration schema for CodexProbe.

The whole point of this module is requirement 4 of the task: swapping which real
backend the proxy forwards to must be a *config* change, not a code change. So
every backend-specific fact lives here, and no other module hard-codes a URL, a
model name, or a wire format.

The key modelling decision: `BackendConfig.wire_api` describes what the *real
backend* speaks. Codex itself always speaks the Responses API to us (it has no
other option since Feb 2026). So the proxy's operating mode is derived, not
configured separately:

    backend speaks "responses"  ->  passthrough (relay bytes unchanged)
    backend speaks "chat"       ->  translate   (Responses <-> Chat Completions)

That is why `ProxyConfig.mode` is a computed property rather than a field: it
cannot drift out of sync with the backend it describes.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProxyMode(str, Enum):
    """How the proxy bridges Codex to the backend. Derived from the backend's wire API."""

    PASSTHROUGH = "passthrough"
    TRANSLATE = "translate"


class BackendConfig(BaseModel):
    """The real LLM backend that CodexProbe forwards to."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(
        description="Root of the backend API, e.g. 'https://api.openai.com/v1' "
        "or 'http://localhost:11434/v1'. No trailing slash required."
    )
    wire_api: Literal["responses", "chat"] = Field(
        default="responses",
        description="What the backend speaks. 'responses' -> passthrough; 'chat' -> translate.",
    )
    api_key_env: str | None = Field(
        default=None,
        description="Name of the environment variable holding the backend API key. "
        "Preferred over `api_key` so secrets stay out of config files.",
    )
    api_key: str | None = Field(
        default=None,
        description="Literal API key. Discouraged; use `api_key_env`. Takes lower "
        "precedence than `api_key_env` when both are set.",
    )
    model: str | None = Field(
        default=None,
        description="Optional model override. When set, replaces the 'model' field of "
        "every forwarded request. Lets you point Codex at a local model without "
        "Codex knowing the name.",
    )
    timeout_s: float = Field(
        default=600.0,
        gt=0,
        description="Total request timeout. Generous by default: local models on CPU "
        "can take minutes for a long generation.",
    )
    connect_timeout_s: float = Field(
        default=10.0,
        gt=0,
        description="Connection timeout. Short, so a wrong base_url fails fast "
        "instead of hanging the Codex session.",
    )

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        v = v.rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got {v!r}")
        return v

    def resolve_api_key(self) -> str | None:
        """Return the API key to forward, or None if the backend needs no auth.

        `api_key_env` wins over `api_key`. A named-but-unset environment variable is
        an error rather than a silent None, because the resulting 401 from the backend
        would otherwise be very confusing to debug.
        """
        if self.api_key_env:
            value = os.environ.get(self.api_key_env)
            if not value:
                raise ValueError(
                    f"api_key_env={self.api_key_env!r} is set in the config but that "
                    f"environment variable is empty or undefined."
                )
            return value
        return self.api_key


class ListenConfig(BaseModel):
    """Where the proxy itself listens."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="127.0.0.1", description="Bind address. Loopback by default.")
    port: int = Field(
        default=8135,
        ge=0,
        le=65535,
        description="Bind port. 0 means 'let the OS pick a free port', which is what "
        "the test suite uses to avoid collisions.",
    )


class ProxyConfig(BaseModel):
    """Top-level CodexProbe configuration."""

    model_config = ConfigDict(extra="forbid")

    backend: BackendConfig
    listen: ListenConfig = Field(default_factory=ListenConfig)
    log_dir: Path = Field(
        default=Path("./codex_probe_logs"),
        description="Directory for per-session JSONL logs. Created if absent.",
    )
    session_id: str | None = Field(
        default=None,
        description="Names the log file. Defaults to a UTC timestamp at start().",
    )
    seed: int | None = Field(
        default=None,
        description="Injected as the 'seed' field of every forwarded request when the "
        "request does not already carry one. Backends that honour it (vLLM, some "
        "Ollama builds) then produce reproducible output.",
    )

    @property
    def mode(self) -> ProxyMode:
        """Derived from the backend's wire API. See module docstring."""
        return ProxyMode.PASSTHROUGH if self.backend.wire_api == "responses" else ProxyMode.TRANSLATE

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProxyConfig:
        """Build from a plain dict / parsed JSON, as the task's public API specifies."""
        return cls.model_validate(data)
