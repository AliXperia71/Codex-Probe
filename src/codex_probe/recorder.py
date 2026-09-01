"""`ProxyRecorder` -- the public API from the task description.

    recorder = ProxyRecorder(config)
    endpoint = recorder.start()   # "http://127.0.0.1:8135/v1"
    calls = recorder.stop()

Design notes worth defending:

* **The server runs on a background thread.** The caller's code (a script, a
  pytest test, a notebook) stays synchronous and just gets a URL back. Running
  uvicorn in the foreground would mean the caller could never reach the line
  after `start()`.
* **The listening socket is bound before uvicorn starts.** That is what makes
  `port: 0` usable: the OS assigns a free port and `start()` can report the real
  one immediately. It also removes a race -- the socket is already accepting
  connections when `start()` returns, so a caller can launch Codex on the very
  next line without a sleep or a retry loop.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn

from .config import ProxyConfig, ProxyMode
from .logstore import SessionLog
from .server import create_app

logger = logging.getLogger("codex_probe")


class ProxyRecorder:
    """Runs the recording proxy and collects the calls that pass through it."""

    def __init__(self, config: ProxyConfig | dict[str, Any]) -> None:
        self.config = (
            config if isinstance(config, ProxyConfig) else ProxyConfig.from_dict(config)
        )
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._session: SessionLog | None = None
        self._endpoint: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self, startup_timeout_s: float = 15.0) -> str:
        """Start the proxy and return the endpoint to put in Codex's `base_url`.

        The returned URL already carries the `/v1` suffix, so Codex appending
        `/responses` lands on this server's route.
        """
        if self._server is not None:
            raise RuntimeError("recorder is already started")

        session_id = self.config.session_id or datetime.now(timezone.utc).strftime(
            "session-%Y%m%dT%H%M%SZ"
        )
        self._session = SessionLog(session_id, self.config.log_dir)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.config.listen.host, self.config.listen.port))
        except OSError as exc:
            sock.close()
            raise RuntimeError(
                f"could not bind {self.config.listen.host}:{self.config.listen.port} -- "
                f"is another recorder already running? ({exc})"
            ) from exc
        sock.listen(128)
        host, port = sock.getsockname()[:2]
        self._socket = sock

        app = create_app(self.config, self._session)
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="warning", access_log=False, lifespan="on")
        )
        self._server = server

        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [sock]},
            name=f"codex-probe-{port}",
            daemon=True,
        )
        thread.start()
        self._thread = thread

        # `server.started` flips once the startup hooks (including our httpx client)
        # have run. Waiting on it means the endpoint is fully usable on return.
        deadline = time.monotonic() + startup_timeout_s
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("proxy server thread exited during startup")
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError(f"proxy did not start within {startup_timeout_s}s")
            time.sleep(0.02)

        self._endpoint = f"http://{host}:{port}/v1"
        logger.info(
            "CodexProbe listening on %s (mode=%s, backend=%s, log=%s)",
            self._endpoint,
            self.config.mode.value,
            self.config.backend.base_url,
            self.log_path,
        )
        return self._endpoint

    def stop(self, shutdown_timeout_s: float = 15.0) -> list[dict[str, Any]]:
        """Shut the proxy down and return every recorded call, in order."""
        if self._server is not None:
            # uvicorn polls this flag, finishes in-flight requests, then runs the
            # shutdown hooks. A hard kill here would lose the final call's record.
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=shutdown_timeout_s)
            if self._thread.is_alive():
                logger.warning("proxy thread did not exit within %ss", shutdown_timeout_s)
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass

        self._server = None
        self._thread = None
        self._socket = None
        return self.calls()

    # -- accessors ---------------------------------------------------------

    def calls(self) -> list[dict[str, Any]]:
        """Recorded calls so far. Safe to call while the proxy is still running."""
        return self._session.records() if self._session is not None else []

    @property
    def endpoint(self) -> str | None:
        return self._endpoint

    @property
    def log_path(self) -> Path | None:
        return self._session.path if self._session is not None else None

    @property
    def mode(self) -> ProxyMode:
        return self.config.mode

    def codex_config_toml(self, provider_id: str = "codexprobe", model: str | None = None) -> str:
        """Emit the exact `~/.codex/config.toml` fragment for this running proxy.

        Generated rather than documented because the port can be OS-assigned, and a
        hand-copied stale port is the single most common way to get this wrong.
        """
        if self._endpoint is None:
            raise RuntimeError("call start() before generating the Codex config")
        model_name = model or self.config.backend.model or "gpt-5.1"
        return (
            f'model = "{model_name}"\n'
            f'model_provider = "{provider_id}"\n'
            f"\n"
            f"[model_providers.{provider_id}]\n"
            f'name = "CodexProbe"\n'
            f'base_url = "{self._endpoint}"\n'
            f'wire_api = "responses"\n'
            f'env_key = "CODEX_PROBE_KEY"\n'
        )

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> ProxyRecorder:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
