"""Config schema tests.

Configuration is the task's designated swap mechanism (requirement 4), so its
validation is load-bearing rather than incidental: a misconfigured backend should
fail loudly at construction, not as a confusing 401 twenty minutes into a run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from codex_probe import BackendConfig, ProxyConfig, ProxyMode


def test_minimal_config_gets_sensible_defaults():
    config = ProxyConfig.from_dict({"backend": {"base_url": "https://api.openai.com/v1"}})
    assert config.backend.wire_api == "responses"
    assert config.listen.host == "127.0.0.1"
    assert config.listen.port == 8135
    assert config.log_dir == Path("./codex_probe_logs")
    assert config.seed is None


def test_mode_is_derived_from_backend_wire_api():
    """The proxy's mode cannot drift out of sync with the backend it describes."""
    responses = ProxyConfig.from_dict({"backend": {"base_url": "http://x/v1", "wire_api": "responses"}})
    chat = ProxyConfig.from_dict({"backend": {"base_url": "http://x/v1", "wire_api": "chat"}})
    assert responses.mode is ProxyMode.PASSTHROUGH
    assert chat.mode is ProxyMode.TRANSLATE


def test_backend_is_required():
    with pytest.raises(ValidationError):
        ProxyConfig.from_dict({})


def test_base_url_must_be_http():
    with pytest.raises(ValidationError, match="must start with http"):
        BackendConfig(base_url="localhost:11434/v1")


def test_trailing_slash_is_normalised():
    """Otherwise the forwarded URL ends up with a double slash."""
    assert BackendConfig(base_url="http://localhost:11434/v1/").base_url == "http://localhost:11434/v1"


def test_unknown_wire_api_is_rejected():
    with pytest.raises(ValidationError):
        BackendConfig(base_url="http://x/v1", wire_api="grpc")


def test_typos_are_rejected_rather_than_silently_ignored():
    """`extra="forbid"` turns a misspelled key into an error instead of a default."""
    with pytest.raises(ValidationError):
        BackendConfig(base_url="http://x/v1", base_urls="oops")
    with pytest.raises(ValidationError):
        ProxyConfig.from_dict({"backend": {"base_url": "http://x/v1"}, "logdir": "./x"})


def test_port_range_is_validated():
    with pytest.raises(ValidationError):
        ProxyConfig.from_dict({"backend": {"base_url": "http://x/v1"}, "listen": {"port": 70000}})


def test_api_key_resolved_from_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-abc")
    assert BackendConfig(base_url="http://x/v1", api_key_env="MY_KEY").resolve_api_key() == "sk-abc"


def test_missing_env_var_raises_a_clear_error(monkeypatch):
    """Silently returning None here would surface as an opaque 401 from the backend."""
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    backend = BackendConfig(base_url="http://x/v1", api_key_env="ABSENT_KEY")
    with pytest.raises(ValueError, match="ABSENT_KEY"):
        backend.resolve_api_key()


def test_env_key_takes_precedence_over_literal(monkeypatch):
    monkeypatch.setenv("MY_KEY", "from-env")
    backend = BackendConfig(base_url="http://x/v1", api_key_env="MY_KEY", api_key="literal")
    assert backend.resolve_api_key() == "from-env"


def test_no_auth_configured_returns_none():
    """Local servers such as Ollama take no credential."""
    assert BackendConfig(base_url="http://localhost:11434/v1").resolve_api_key() is None


def test_config_accepts_a_proxyconfig_instance_or_a_dict():
    from codex_probe import ProxyRecorder

    as_dict = ProxyRecorder({"backend": {"base_url": "http://x/v1"}})
    as_model = ProxyRecorder(ProxyConfig.from_dict({"backend": {"base_url": "http://x/v1"}}))
    assert as_dict.config.backend.base_url == as_model.config.backend.base_url
