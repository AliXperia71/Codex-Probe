"""The FastAPI application: one recording reverse-proxy endpoint.

Codex POSTs to `{base_url}/responses` -- the path is fixed in Codex's Rust source
(`InferenceEndpoint::path()` returns `"/responses"`), so that is the only route
that has to exist. `/v1/models` is proxied too because some tooling probes it.

The central mechanism is the **tee**. `_relay_stream` yields each chunk to Codex
the instant it arrives and, in the same step, hands a copy to a collector.
Consequences worth being able to defend:

* Codex sees bytes with no added latency, so the CLI keeps rendering live and
  Codex's `stream_idle_timeout_ms` (300 s default) never fires because of us.
* Reassembly happens after the stream closes, off the critical path.
* A bug in the parser can corrupt a log entry but cannot corrupt the relay. That
  is what makes requirement 1 structurally true rather than merely tested.

The alternative -- read the whole upstream response, parse it, then forward it --
would be far simpler and is what most naive proxies do. It also destroys the
interactive feel of the CLI and turns a 60-second generation into 60 seconds of
apparent hang.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .config import ProxyConfig, ProxyMode
from .logstore import CallRecord, SessionLog, utc_now_iso
from .wire.chat import ChatReassembler
from .wire.responses import (
    ResponsesReassembler,
    extract_function_calls,
    extract_output_text,
)
from .wire.sse import SSEParser
from .wire.translate import (
    ChatToResponsesTranslator,
    chat_completion_to_responses,
    responses_request_to_chat,
)

logger = logging.getLogger("codex_probe")

# Headers that describe a single hop and must not be forwarded (RFC 7230 s6.1).
# `host` is dropped so httpx sets it from the upstream URL; `content-length` is
# dropped because the body may be re-serialised at a different length.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

# Stripped from the *response* because httpx has already decoded the body for us
# (see `_relay_stream`), so advertising the original encoding or length would be a lie.
_STRIPPED_RESPONSE_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


def _decompress(raw: bytes, encoding: str | None) -> bytes:
    """Decode a compressed request body for logging.

    Codex can zstd-compress request bodies (`Compression::Zstd` in
    `codex-rs/codex-api`). Only the *logged copy* is decompressed; passthrough
    still forwards the original bytes.
    """
    if not raw or not encoding:
        return raw
    encoding = encoding.lower().strip()
    try:
        if encoding == "zstd":
            import zstandard

            return zstandard.ZstdDecompressor().decompress(raw, max_output_size=0)
        if encoding == "gzip":
            import gzip

            return gzip.decompress(raw)
        if encoding == "deflate":
            import zlib

            return zlib.decompress(raw)
    except Exception as exc:  # noqa: BLE001 - logging must never break the proxy
        logger.warning("could not decompress %s request body: %s", encoding, exc)
    return raw


def _parse_json(raw: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


class _StreamCollector:
    """Accumulates a copy of the relayed stream and reassembles it afterwards."""

    def __init__(self, mode: ProxyMode) -> None:
        self.mode = mode
        self.parser = SSEParser()
        self.raw_events: list[str] = []
        self.responses = ResponsesReassembler() if mode is ProxyMode.PASSTHROUGH else None
        self.chat = ChatReassembler() if mode is ProxyMode.TRANSLATE else None

    def feed(self, chunk: bytes) -> None:
        for event in self.parser.feed(chunk):
            self._absorb(event)

    def close(self) -> None:
        for event in self.parser.flush():
            self._absorb(event)

    def _absorb(self, event) -> None:
        self.raw_events.append(event.raw)
        if self.responses is not None:
            self.responses.handle(event)
        elif self.chat is not None:
            self.chat.handle(event)


def create_app(config: ProxyConfig, session: SessionLog) -> FastAPI:
    """Build the proxy application for one recorder session."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One client for the whole session: connection pooling matters because
        # Codex issues many sequential calls to the same host, and a fresh TLS
        # handshake per call would add real latency to every turn.
        app.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.backend.timeout_s, connect=config.backend.connect_timeout_s
            ),
            follow_redirects=True,
        )
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="CodexProbe", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.session = session

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": config.mode.value,
            "backend": config.backend.base_url,
            "calls": len(session),
        }

    @app.get("/v1/models")
    @app.get("/models")
    async def models() -> Response:
        """Proxy the model list. Not used by Codex, handy for sanity checks."""
        client: httpx.AsyncClient = app.state.client
        try:
            upstream = await client.get(
                f"{config.backend.base_url}/models", headers=_auth_headers(config)
            )
        except httpx.HTTPError as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=502)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    @app.post("/v1/responses")
    @app.post("/responses")
    async def responses(request: Request) -> Response:
        return await _handle_call(app, config, session, request)

    return app


def _auth_headers(config: ProxyConfig) -> dict[str, str]:
    key = config.backend.resolve_api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _forward_headers(request: Request, config: ProxyConfig, override_body: bool) -> dict[str, str]:
    """Build the upstream header set.

    Codex's own session headers (`x-client-request-id`, `session_id`, ...) are
    forwarded untouched: they are part of the request Codex chose to make, and
    requirement 1 says the backend must see what Codex sent.
    """
    # Keys are normalised to lower case throughout. HTTP header names are
    # case-insensitive, but a plain dict is not: mixing "Authorization" and
    # "authorization" here silently sends the credential twice, and the dummy
    # token Codex supplied can win.
    headers = {
        name.lower(): value
        for name, value in request.headers.items()
        if name.lower() not in _HOP_BY_HOP
    }

    # The proxy holds the real credential; Codex is typically configured with a
    # dummy key, because the backend it believes it is talking to is us.
    key = config.backend.resolve_api_key()
    if key:
        headers["authorization"] = f"Bearer {key}"
    else:
        # A local server with no auth rejects a stray dummy bearer token.
        headers.pop("authorization", None)

    if override_body:
        # The body was re-serialised, so any encoding claim no longer holds.
        headers.pop("content-encoding", None)
        headers["content-type"] = "application/json"
    return headers


def _prepare_body(
    config: ProxyConfig, decoded: dict[str, Any] | None, raw: bytes
) -> tuple[bytes, dict[str, Any] | None, bool]:
    """Decide what body to send upstream.

    Returns `(bytes_to_send, parsed_body_sent, was_rewritten)`.

    When no model override and no seed are configured, passthrough forwards the
    **original bytes**, byte for byte. That is a stronger guarantee than
    re-serialising an equivalent JSON object: key order, whitespace and unicode
    escaping all reach the backend exactly as Codex wrote them.
    """
    backend = config.backend

    if config.mode is ProxyMode.TRANSLATE:
        if decoded is None:
            return raw, None, False
        chat_body = responses_request_to_chat(decoded)
        if backend.model:
            chat_body["model"] = backend.model
        if config.seed is not None and "seed" not in chat_body:
            chat_body["seed"] = config.seed
        return json.dumps(chat_body).encode("utf-8"), chat_body, True

    needs_rewrite = backend.model is not None or config.seed is not None
    if not needs_rewrite or decoded is None:
        return raw, decoded, False

    body = dict(decoded)
    if backend.model:
        body["model"] = backend.model
    if config.seed is not None and "seed" not in body:
        body["seed"] = config.seed
    return json.dumps(body).encode("utf-8"), body, True


async def _handle_call(
    app: FastAPI, config: ProxyConfig, session: SessionLog, request: Request
) -> Response:
    client: httpx.AsyncClient = app.state.client
    call_index = session.next_index()
    started_at = time.perf_counter()

    raw = await request.body()
    decoded_raw = _decompress(raw, request.headers.get("content-encoding"))
    decoded = _parse_json(decoded_raw)

    body_bytes, backend_body, rewritten = _prepare_body(config, decoded, raw)

    suffix = "/chat/completions" if config.mode is ProxyMode.TRANSLATE else "/responses"
    target_url = f"{config.backend.base_url}{suffix}"

    record = CallRecord(
        call_index=call_index,
        session_id=session.session_id,
        timestamp_start=utc_now_iso(),
        mode=config.mode.value,
        backend_url=target_url,
        model=(decoded or {}).get("model"),
        request=decoded,
        backend_request=backend_body if config.mode is ProxyMode.TRANSLATE or rewritten else decoded,
        request_headers={k: v for k, v in request.headers.items() if k.lower() != "authorization"},
    )

    headers = _forward_headers(request, config, override_body=rewritten)

    upstream_request = client.build_request(
        "POST", target_url, content=body_bytes, headers=headers
    )

    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        # Connection refused, DNS failure, TLS error, timeout on connect.
        record.status = "error"
        record.error = f"{type(exc).__name__}: {exc}"
        record.timestamp_end = utc_now_iso()
        record.latency_ms = (time.perf_counter() - started_at) * 1000
        session.append(record)
        logger.error("call %d: upstream request failed: %s", call_index, exc)
        return JSONResponse(
            {"error": {"message": f"CodexProbe could not reach {target_url}: {exc}",
                       "type": "upstream_unreachable"}},
            status_code=502,
        )

    content_type = upstream.headers.get("content-type", "")
    is_stream = "text/event-stream" in content_type.lower()

    if not is_stream:
        return await _handle_unary(upstream, record, session, started_at, config)

    return _handle_stream(upstream, record, session, started_at, config)


def _record_output(record: CallRecord) -> None:
    """Fill the convenience fields (`output_text`, `function_calls`) from `response`.

    They duplicate information already in `response`, but having them at the top
    level is what makes the log greppable with `jq` without walking the item array.
    """
    output = (record.response or {}).get("output")
    if not isinstance(output, list):
        return
    record.output_text = extract_output_text(output)
    record.function_calls = extract_function_calls(output)


async def _handle_unary(
    upstream: httpx.Response,
    record: CallRecord,
    session: SessionLog,
    started_at: float,
    config: ProxyConfig,
) -> Response:
    """Non-streaming path: error bodies, and mock servers used by the tests."""
    try:
        body = await upstream.aread()
    finally:
        await upstream.aclose()

    record.http_status = upstream.status_code
    record.latency_ms = (time.perf_counter() - started_at) * 1000
    record.timestamp_end = utc_now_iso()

    parsed = _parse_json(body)
    if upstream.status_code >= 400:
        record.status = "failed"
        # `error` is a string field, so a structured backend error is serialised
        # rather than assigned raw.
        backend_error = (parsed or {}).get("error") if parsed else None
        if backend_error is None:
            record.error = body.decode("utf-8", "replace")
        elif isinstance(backend_error, str):
            record.error = backend_error
        else:
            record.error = json.dumps(backend_error)
    else:
        record.status = "completed"
        record.response = parsed
        if config.mode is ProxyMode.TRANSLATE and parsed is not None:
            # A backend that ignored `stream: true` answered with a single
            # `chat.completion`. Codex cannot parse that where it expects a
            # `response`, so convert before relaying.
            record.backend_response = parsed
            record.response = chat_completion_to_responses(parsed)
            body = json.dumps(record.response).encode("utf-8")

    _record_output(record)
    session.append(record)

    headers = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() not in _STRIPPED_RESPONSE_HEADERS
    }
    return Response(
        content=body,
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type"),
    )


def _handle_stream(
    upstream: httpx.Response,
    record: CallRecord,
    session: SessionLog,
    started_at: float,
    config: ProxyConfig,
) -> StreamingResponse:
    collector = _StreamCollector(config.mode)

    headers = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() not in _STRIPPED_RESPONSE_HEADERS
    }

    if config.mode is ProxyMode.PASSTHROUGH:
        body_iter = _relay_stream(upstream, collector, record, session, started_at)
    else:
        body_iter = _translate_stream(upstream, collector, record, session, started_at)

    return StreamingResponse(
        body_iter,
        status_code=upstream.status_code,
        headers=headers,
        media_type="text/event-stream",
    )


def _finalize(
    record: CallRecord,
    session: SessionLog,
    started_at: float,
    collector: _StreamCollector,
    error: BaseException | None,
) -> None:
    """Reassemble, then write the record exactly once.

    Called from a `finally`, so a client disconnect or an upstream crash still
    produces a log entry -- flagged `incomplete`, with whatever arrived before the
    break.

    Deliberately synchronous even though its callers are async: it performs no
    I/O awaits, and a coroutine awaited inside a `finally` during task
    cancellation can be cancelled again before it completes. Keeping it sync
    guarantees the record is written on the disconnect path, which is precisely
    the path where the log is most valuable.
    """
    collector.close()
    record.raw_events = collector.raw_events
    record.latency_ms = (time.perf_counter() - started_at) * 1000
    record.timestamp_end = utc_now_iso()

    if collector.responses is not None:
        result = collector.responses.result()
        record.response = result["response"]
        record.output_text = result["output_text"]
        record.status = result["status"]
        if result["error"] is not None:
            record.error = json.dumps(result["error"]) if not isinstance(result["error"], str) else result["error"]
        record.function_calls = extract_function_calls((record.response or {}).get("output"))
    elif collector.chat is not None:
        record.backend_response = collector.chat.result()
        record.output_text = collector.chat.content
        # Read the tool calls back out of the *translated* Responses object rather
        # than the Chat one, so `function_calls` has the same shape in both modes.
        # The Chat form nests the name under `function`, which would make a
        # consumer of the log branch on `mode` to read a field.
        record.function_calls = extract_function_calls((record.response or {}).get("output"))
        record.status = "incomplete" if collector.chat.finish_reason == "length" else "completed"

    if error is not None:
        record.status = "incomplete" if record.status == "completed" else record.status
        record.error = f"{type(error).__name__}: {error}"

    session.append(record)


async def _relay_stream(
    upstream: httpx.Response,
    collector: _StreamCollector,
    record: CallRecord,
    session: SessionLog,
    started_at: float,
) -> AsyncIterator[bytes]:
    """Passthrough: relay bytes to Codex while teeing a copy to the collector."""
    record.http_status = upstream.status_code
    error: BaseException | None = None
    try:
        # `aiter_bytes` yields content-decoded bytes; the matching
        # Content-Encoding header was stripped in `_handle_stream`.
        async for chunk in upstream.aiter_bytes():
            collector.feed(chunk)
            yield chunk
    except BaseException as exc:  # noqa: BLE001 - includes client-disconnect cancellation
        error = exc
        raise
    finally:
        try:
            await upstream.aclose()
        except Exception:  # noqa: BLE001 - a failed close must not lose the record
            pass
        _finalize(record, session, started_at, collector, error)


async def _translate_stream(
    upstream: httpx.Response,
    collector: _StreamCollector,
    record: CallRecord,
    session: SessionLog,
    started_at: float,
) -> AsyncIterator[bytes]:
    """Translate: consume a Chat stream, emit a Responses stream."""
    record.http_status = upstream.status_code
    translator = ChatToResponsesTranslator(model=record.model)
    error: BaseException | None = None
    try:
        for frame in translator.start():
            yield frame

        async for chunk in upstream.aiter_bytes():
            for event in collector.parser.feed(chunk):
                collector.raw_events.append(event.raw)
                if collector.chat is not None:
                    collector.chat.handle(event)
                for frame in translator.consume(event):
                    yield frame

        for event in collector.parser.flush():
            collector.raw_events.append(event.raw)
            if collector.chat is not None:
                collector.chat.handle(event)
            for frame in translator.consume(event):
                yield frame

        # The completed items. Without these Codex renders the text and then
        # forgets the turn -- see wire/translate.py.
        for frame in translator.finish():
            yield frame
    except BaseException as exc:  # noqa: BLE001
        error = exc
        raise
    finally:
        try:
            await upstream.aclose()
        except Exception:  # noqa: BLE001
            pass
        record.response = translator.as_responses_object()
        _finalize(record, session, started_at, collector, error)
