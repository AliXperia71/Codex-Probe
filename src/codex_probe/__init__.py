"""CodexProbe -- a recording reverse proxy for the OpenAI Codex CLI.

Sits at Codex's `model_providers.<name>.base_url` extension point, logs every LLM
call verbatim, and lets you swap the backend model with a config change.

    from codex_probe import ProxyRecorder

    recorder = ProxyRecorder({
        "backend": {
            "base_url": "https://api.openai.com/v1",
            "wire_api": "responses",
            "api_key_env": "OPENAI_API_KEY",
        },
        "log_dir": "./logs",
    })
    endpoint = recorder.start()
    ...
    calls = recorder.stop()
"""

from .config import BackendConfig, ListenConfig, ProxyConfig, ProxyMode
from .logstore import CallRecord, SessionLog, read_session
from .recorder import ProxyRecorder

__version__ = "0.1.0"

__all__ = [
    "ProxyRecorder",
    "ProxyConfig",
    "BackendConfig",
    "ListenConfig",
    "ProxyMode",
    "CallRecord",
    "SessionLog",
    "read_session",
    "__version__",
]
