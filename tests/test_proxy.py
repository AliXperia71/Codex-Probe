"""End-to-end proxy tests against the mock backend.

These are the tests the task's deliverables name explicitly: passthrough
correctness (request/response round-trip unchanged), streaming reassembly
correctness, and log schema validation.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from codex_probe import ProxyRecorder, read_session

from conftest import (
    MockBackend,
    chat_text_stream,
    chat_tool_call_stream,
    responses_text_stream,
    responses_tool_call_stream,
)

SAMPLE_REQUEST = {
    "model": "gpt-5.5",
    "instructions": "You are a coding agent.",
    "input": [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "add a docstring"}]}
    ],
    "tools": [{"type": "function", "name": "shell",
               "parameters": {"type": "object", "properties": {}}}],
    "stream": True,
}


async def post(endpoint: str, body: dict, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30) as client:
        return await client.post(f"{endpoint}/responses", json=body, **kwargs)


# ---------------------------------------------------------------------------
# Passthrough fidelity
# ---------------------------------------------------------------------------


async def test_request_is_forwarded_byte_identically(backend: MockBackend, probe_config):
    """Requirement 1: the backend must see exactly what Codex sent.

    Asserting on raw bytes rather than parsed JSON is deliberate -- it catches key
    reordering and re-encoding that a dict comparison would silently accept.
    """
    backend.chunks = responses_text_stream()
    raw = json.dumps(SAMPLE_REQUEST).encode()

    with ProxyRecorder(probe_config) as recorder:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(
                f"{recorder.endpoint}/responses",
                content=raw,
                headers={"content-type": "application/json"},
            )

    assert backend.last_request.raw_body == raw
    assert backend.last_request.path == "/v1/responses"


async def test_response_is_relayed_byte_identically(backend: MockBackend, probe_config):
    backend.chunks = responses_text_stream("Hello world")
    expected = b"".join(backend.chunks)

    with ProxyRecorder(probe_config) as recorder:
        response = await post(recorder.endpoint, SAMPLE_REQUEST)

    assert response.status_code == 200
    assert response.content == expected


async def test_codex_session_headers_are_forwarded(backend: MockBackend, probe_config):
    backend.chunks = responses_text_stream()
    with ProxyRecorder(probe_config) as recorder:
        await post(recorder.endpoint, SAMPLE_REQUEST,
                   headers={"x-client-request-id": "thread-42", "session_id": "sess-9"})

    forwarded = backend.last_request.headers
    assert forwarded["x-client-request-id"] == "thread-42"
    assert forwarded["session_id"] == "sess-9"


async def test_hop_by_hop_headers_are_stripped(backend: MockBackend, probe_config):
    """Per RFC 7230, hop directives describe one connection and must not be relayed.

    The client's `connection: close` applies to the client->proxy hop only. The
    proxy opens its own upstream connection, which carries its own directives, so
    the assertion is that the client's *value* does not leak -- not that the header
    is absent upstream.
    """
    backend.chunks = responses_text_stream()
    with ProxyRecorder(probe_config) as recorder:
        await post(recorder.endpoint, SAMPLE_REQUEST,
                   headers={"connection": "close", "te": "trailers"})

    assert backend.last_request.headers.get("connection", "").lower() != "close"
    assert "te" not in backend.last_request.headers


async def test_streaming_is_incremental_not_buffered(backend: MockBackend, probe_config):
    """The property the tee exists to guarantee.

    A buffer-then-forward proxy passes every other test in this file. Only timing
    distinguishes it: here the first chunk must reach the client long before the
    last one has been sent.
    """
    backend.chunks = responses_text_stream("streamed")
    backend.chunk_delay_s = 0.15  # 4 chunks -> ~0.6s total

    with ProxyRecorder(probe_config) as recorder:
        async with httpx.AsyncClient(timeout=30) as client:
            start = time.perf_counter()
            first_at = None
            async with client.stream(
                "POST", f"{recorder.endpoint}/responses", json=SAMPLE_REQUEST
            ) as response:
                async for _ in response.aiter_bytes():
                    if first_at is None:
                        first_at = time.perf_counter() - start
            total = time.perf_counter() - start

    assert first_at is not None
    assert total > 0.4, "mock did not actually delay; the test would be vacuous"
    assert first_at < total * 0.6, (
        f"first chunk arrived at {first_at:.3f}s of {total:.3f}s total -- "
        "the proxy appears to be buffering the whole response"
    )


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


async def test_call_is_recorded_with_full_request_and_response(backend, probe_config):
    """Requirement 3: unabridged capture of both directions."""
    backend.chunks = responses_text_stream("Hello world")

    with ProxyRecorder(probe_config) as recorder:
        await post(recorder.endpoint, SAMPLE_REQUEST)
        calls = recorder.calls()

    assert len(calls) == 1
    call = calls[0]

    # The entire request, not a summary.
    assert call["request"]["instructions"] == "You are a coding agent."
    assert call["request"]["input"] == SAMPLE_REQUEST["input"]
    assert call["request"]["tools"] == SAMPLE_REQUEST["tools"]

    # The entire response, streaming reassembled.
    assert call["status"] == "completed"
    assert call["output_text"] == "Hello world"
    assert call["response"]["usage"]["total_tokens"] == 15
    assert call["http_status"] == 200
    assert call["latency_ms"] > 0
    assert call["mode"] == "passthrough"
    assert call["backend_url"].endswith("/v1/responses")
    assert len(call["raw_events"]) == 4


async def test_tool_calls_are_extracted(backend, probe_config):
    backend.chunks = responses_tool_call_stream()
    with ProxyRecorder(probe_config) as recorder:
        await post(recorder.endpoint, SAMPLE_REQUEST)
        calls = recorder.calls()

    assert [c["name"] for c in calls[0]["function_calls"]] == ["shell", "read_file"]


async def test_calls_are_recorded_in_order(backend, probe_config):
    backend.chunks = responses_text_stream()
    with ProxyRecorder(probe_config) as recorder:
        for _ in range(5):
            await post(recorder.endpoint, SAMPLE_REQUEST)
        calls = recorder.stop()

    assert [c["call_index"] for c in calls] == [0, 1, 2, 3, 4]


async def test_log_is_written_as_jsonl_one_file_per_session(backend, probe_config):
    """Requirement 6: a reviewer must be able to replay a session turn by turn."""
    backend.chunks = responses_text_stream("logged")
    recorder = ProxyRecorder(probe_config)
    recorder.start()
    await post(recorder.endpoint, SAMPLE_REQUEST)
    await post(recorder.endpoint, SAMPLE_REQUEST)
    calls = recorder.stop()

    assert recorder.log_path.name == "test-session.jsonl"
    on_disk = read_session(recorder.log_path)
    assert len(on_disk) == 2 == len(calls)
    assert [r["call_index"] for r in on_disk] == [0, 1]
    # Every line must be independently parseable.
    for line in recorder.log_path.read_text().splitlines():
        assert json.loads(line)["session_id"] == "test-session"


async def test_large_payloads_are_not_truncated(backend, probe_config):
    """Requirement 3 forbids eliding anything, at any size."""
    backend.chunks = responses_text_stream("x")
    big = {**SAMPLE_REQUEST, "input": [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "y" * 500_000}]}
    ]}

    with ProxyRecorder(probe_config) as recorder:
        await post(recorder.endpoint, big)
        calls = recorder.calls()

    assert len(calls[0]["request"]["input"][0]["content"][0]["text"]) == 500_000


async def test_concurrent_calls_do_not_interleave_in_the_log(backend, probe_config):
    backend.chunks = responses_text_stream("concurrent")
    backend.chunk_delay_s = 0.02

    with ProxyRecorder(probe_config) as recorder:
        await asyncio.gather(*(post(recorder.endpoint, SAMPLE_REQUEST) for _ in range(8)))
        calls = recorder.stop()

    assert len(calls) == 8
    assert sorted(c["call_index"] for c in calls) == list(range(8))
    for call in calls:
        assert call["output_text"] == "concurrent"
        assert call["status"] == "completed"


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


async def test_backend_error_status_is_propagated(backend, probe_config):
    backend.status_code = 429
    backend.json_body = {"error": {"message": "rate limited", "code": "rate_limit_exceeded"}}

    with ProxyRecorder(probe_config) as recorder:
        response = await post(recorder.endpoint, SAMPLE_REQUEST)
        calls = recorder.calls()

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limit_exceeded"
    assert calls[0]["status"] == "failed"
    assert calls[0]["http_status"] == 429


async def test_unreachable_backend_fails_fast_with_502(tmp_path):
    """A wrong base_url must produce a clear error, not a hang."""
    config = {
        # Port 1 is reserved and refuses connections immediately.
        "backend": {"base_url": "http://127.0.0.1:1/v1", "wire_api": "responses",
                    "connect_timeout_s": 2.0},
        "listen": {"host": "127.0.0.1", "port": 0},
        "log_dir": str(tmp_path / "logs"),
        "session_id": "unreachable",
    }
    with ProxyRecorder(config) as recorder:
        started = time.perf_counter()
        response = await post(recorder.endpoint, SAMPLE_REQUEST)
        elapsed = time.perf_counter() - started
        calls = recorder.calls()

    assert response.status_code == 502
    assert elapsed < 10
    assert "could not reach" in response.json()["error"]["message"]
    assert calls[0]["status"] == "error"
    assert calls[0]["error"]


async def test_backend_crash_mid_stream_leaves_a_partial_record(backend, probe_config):
    """A dead backend must still produce a usable log of what arrived."""
    backend.chunks = responses_text_stream("partial")
    backend.drop_after = 2  # created + text delta, then die

    with ProxyRecorder(probe_config) as recorder:
        try:
            await post(recorder.endpoint, SAMPLE_REQUEST)
        except httpx.HTTPError:
            pass  # the client sees a broken stream, which is correct
        calls = recorder.calls()

    assert len(calls) == 1
    assert calls[0]["status"] == "incomplete"
    assert calls[0]["output_text"] == "partial"  # recovered from the deltas


async def test_client_disconnect_mid_stream_still_records(backend, probe_config):
    """Ctrl-C in Codex must not lose the call that was in flight."""
    backend.chunks = responses_text_stream("abandoned")
    backend.chunk_delay_s = 0.1

    with ProxyRecorder(probe_config) as recorder:
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "POST", f"{recorder.endpoint}/responses", json=SAMPLE_REQUEST
            ) as response:
                async for _ in response.aiter_bytes():
                    break  # walk away after the first chunk
        await asyncio.sleep(0.5)  # let the server-side finally run
        calls = recorder.calls()

    assert len(calls) == 1
    assert calls[0]["status"] in ("incomplete", "completed")


# ---------------------------------------------------------------------------
# Config-driven behaviour
# ---------------------------------------------------------------------------


async def test_model_override_rewrites_the_request(backend, probe_config):
    """Lets you point Codex at a local model without Codex knowing its name."""
    backend.chunks = responses_text_stream()
    probe_config["backend"]["model"] = "qwen2.5-coder:7b"

    with ProxyRecorder(probe_config) as recorder:
        await post(recorder.endpoint, SAMPLE_REQUEST)

    assert backend.last_request.json()["model"] == "qwen2.5-coder:7b"


async def test_seed_is_injected_when_absent(backend, probe_config):
    backend.chunks = responses_text_stream()
    probe_config["seed"] = 1234

    with ProxyRecorder(probe_config) as recorder:
        await post(recorder.endpoint, SAMPLE_REQUEST)

    assert backend.last_request.json()["seed"] == 1234


async def test_seed_does_not_override_an_explicit_one(backend, probe_config):
    backend.chunks = responses_text_stream()
    probe_config["seed"] = 1234

    with ProxyRecorder(probe_config) as recorder:
        await post(recorder.endpoint, {**SAMPLE_REQUEST, "seed": 999})

    assert backend.last_request.json()["seed"] == 999


async def test_zstd_compressed_request_body_is_logged_decompressed(backend, probe_config):
    """Codex can zstd its request bodies; the log must still be readable."""
    import zstandard

    backend.chunks = responses_text_stream()
    raw = json.dumps(SAMPLE_REQUEST).encode()
    compressed = zstandard.ZstdCompressor().compress(raw)

    with ProxyRecorder(probe_config) as recorder:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(
                f"{recorder.endpoint}/responses",
                content=compressed,
                headers={"content-type": "application/json", "content-encoding": "zstd"},
            )
        calls = recorder.calls()

    assert calls[0]["request"]["instructions"] == "You are a coding agent."
    # Passthrough still forwarded the original compressed bytes.
    assert backend.last_request.raw_body == compressed


async def test_api_key_is_substituted_from_env(backend, probe_config, monkeypatch):
    """Codex sends a dummy token; the proxy swaps in the real credential."""
    monkeypatch.setenv("TEST_BACKEND_KEY", "sk-real-secret")
    backend.chunks = responses_text_stream()
    probe_config["backend"]["api_key_env"] = "TEST_BACKEND_KEY"

    with ProxyRecorder(probe_config) as recorder:
        await post(recorder.endpoint, SAMPLE_REQUEST,
                   headers={"authorization": "Bearer dummy"})
        calls = recorder.calls()

    assert backend.last_request.headers["authorization"] == "Bearer sk-real-secret"
    # And the real secret must never be written to the log.
    assert "authorization" not in {k.lower() for k in calls[0]["request_headers"]}


# ---------------------------------------------------------------------------
# Translate mode
# ---------------------------------------------------------------------------


async def test_translate_mode_converts_both_directions(backend, probe_config):
    """The whole point: Codex speaks Responses to a Chat-only backend."""
    backend.chunks = chat_text_stream("Hello from Qwen")
    probe_config["backend"]["wire_api"] = "chat"

    with ProxyRecorder(probe_config) as recorder:
        assert recorder.mode.value == "translate"
        response = await post(recorder.endpoint, SAMPLE_REQUEST)
        calls = recorder.calls()

    # Request direction: the backend received Chat Completions.
    assert backend.last_request.path == "/v1/chat/completions"
    sent = backend.last_request.json()
    assert sent["messages"][0] == {"role": "system", "content": "You are a coding agent."}
    assert sent["tools"][0]["function"]["name"] == "shell"

    # Response direction: Codex received Responses-API events.
    body = response.text
    assert "response.created" in body
    assert "response.output_item.done" in body
    assert "response.completed" in body

    assert calls[0]["mode"] == "translate"
    assert calls[0]["output_text"] == "Hello from Qwen"
    assert calls[0]["backend_response"]["choices"][0]["message"]["content"] == "Hello from Qwen"


async def test_translate_mode_relays_tool_calls(backend, probe_config):
    backend.chunks = chat_tool_call_stream()
    probe_config["backend"]["wire_api"] = "chat"

    with ProxyRecorder(probe_config) as recorder:
        response = await post(recorder.endpoint, SAMPLE_REQUEST)
        calls = recorder.calls()

    assert "response.output_item.done" in response.text
    call = calls[0]["response"]["output"][0]
    assert call["type"] == "function_call"
    assert call["name"] == "shell"
    assert json.loads(call["arguments"]) == {"command": ["ls"]}


async def test_switching_backend_is_config_only(backend, probe_config, tmp_path):
    """Requirement 4, demonstrated: two backends, one unchanged code path."""
    second = MockBackend()
    second.start()
    try:
        backend.chunks = responses_text_stream("from backend one")
        second.chunks = responses_text_stream("from backend two")

        with ProxyRecorder(probe_config) as recorder:
            await post(recorder.endpoint, SAMPLE_REQUEST)
            first_calls = recorder.calls()

        swapped = {**probe_config,
                   "backend": {**probe_config["backend"], "base_url": second.base_url},
                   "session_id": "swapped"}
        with ProxyRecorder(swapped) as recorder:
            await post(recorder.endpoint, SAMPLE_REQUEST)
            second_calls = recorder.calls()
    finally:
        second.stop()

    assert first_calls[0]["output_text"] == "from backend one"
    assert second_calls[0]["output_text"] == "from backend two"
    # `backend_url` is the field that proves which server served each call.
    assert first_calls[0]["backend_url"] != second_calls[0]["backend_url"]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_healthz_reports_state(backend, probe_config):
    with ProxyRecorder(probe_config) as recorder:
        async with httpx.AsyncClient(timeout=10) as client:
            health = (await client.get(f"{recorder.endpoint.rsplit('/v1', 1)[0]}/healthz")).json()

    assert health["status"] == "ok"
    assert health["mode"] == "passthrough"


async def test_endpoint_uses_os_assigned_port(backend, probe_config):
    with ProxyRecorder(probe_config) as recorder:
        assert recorder.endpoint.startswith("http://127.0.0.1:")
        assert recorder.endpoint.endswith("/v1")
        assert int(recorder.endpoint.split(":")[2].split("/")[0]) > 0


async def test_double_start_is_rejected(backend, probe_config):
    recorder = ProxyRecorder(probe_config)
    recorder.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            recorder.start()
    finally:
        recorder.stop()


async def test_codex_config_toml_matches_the_live_endpoint(backend, probe_config):
    with ProxyRecorder(probe_config) as recorder:
        toml = recorder.codex_config_toml()
        assert f'base_url = "{recorder.endpoint}"' in toml
        assert 'wire_api = "responses"' in toml
        assert 'model_provider = "codexprobe"' in toml


async def test_translate_mode_converts_a_non_streaming_backend_reply(backend, probe_config):
    """A Chat backend that ignores `stream: true` must still reach Codex correctly.

    Without conversion Codex receives a `chat.completion` where it expects a
    `response` object and cannot parse the turn.
    """
    probe_config["backend"]["wire_api"] = "chat"
    backend.json_body = {
        "id": "chatcmpl-99",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [{"index": 0, "finish_reason": "tool_calls",
                     "message": {"role": "assistant", "content": "on it",
                                 "tool_calls": [{"id": "call_q", "type": "function",
                                                 "function": {"name": "shell",
                                                              "arguments": '{"command":["ls"]}'}}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }

    with ProxyRecorder(probe_config) as recorder:
        response = await post(recorder.endpoint, SAMPLE_REQUEST)
        calls = recorder.calls()

    body = response.json()
    assert body["object"] == "response"
    assert [item["type"] for item in body["output"]] == ["message", "function_call"]
    assert body["output"][1]["call_id"] == "call_q"
    assert body["usage"] == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}

    assert calls[0]["output_text"] == "on it"
    assert calls[0]["backend_response"]["object"] == "chat.completion"


async def test_function_calls_have_the_same_shape_in_both_modes(backend, probe_config, tmp_path):
    """A log consumer must not have to branch on `mode` to read a tool call's name.

    Chat nests the name under `function`; Responses puts it at the top level. The
    log always uses the Responses shape.
    """
    backend.chunks = responses_tool_call_stream()
    with ProxyRecorder(probe_config) as recorder:
        await post(recorder.endpoint, SAMPLE_REQUEST)
        passthrough = recorder.calls()[0]["function_calls"]

    chat_backend = MockBackend()
    chat_backend.start()
    try:
        chat_backend.chunks = chat_tool_call_stream()
        translate_config = {
            **probe_config,
            "backend": {"base_url": chat_backend.base_url, "wire_api": "chat"},
            "session_id": "translate-shape",
        }
        with ProxyRecorder(translate_config) as recorder:
            await post(recorder.endpoint, SAMPLE_REQUEST)
            translated = recorder.calls()[0]["function_calls"]
    finally:
        chat_backend.stop()

    for entry in (*passthrough, *translated):
        assert entry["type"] == "function_call"
        assert isinstance(entry["name"], str) and entry["name"]
        assert "call_id" in entry and "arguments" in entry

    assert passthrough[0]["name"] == "shell"
    assert translated[0]["name"] == "shell"
