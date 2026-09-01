"""Shared fixtures: a scriptable mock OpenAI-compatible backend.

The whole suite runs offline against this, so it needs no API key, no network and
no Ollama. That matters for a deliverable someone else has to be able to run.

The mock records exactly what it received (raw body bytes and headers) so tests
can assert byte-level passthrough fidelity, and it can emit a scripted list of
SSE chunks with a configurable inter-chunk delay so tests can prove the proxy
relays incrementally rather than buffering.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import pytest
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse


@dataclass
class CapturedRequest:
    path: str
    headers: dict[str, str]
    raw_body: bytes

    def json(self) -> Any:
        return json.loads(self.raw_body)


@dataclass
class MockBackend:
    """A scriptable stand-in for OpenAI / Ollama / vLLM."""

    chunks: list[bytes] = field(default_factory=list)
    """SSE byte chunks to emit, in order. Chunk boundaries are deliberate: tests
    use them to split frames in awkward places."""

    status_code: int = 200
    json_body: Any | None = None
    """When set, respond with JSON instead of SSE (used for error-path tests)."""

    chunk_delay_s: float = 0.0
    drop_after: int | None = None
    """Emit this many chunks then abort the stream, simulating a backend crash."""

    requests: list[CapturedRequest] = field(default_factory=list)
    _server: uvicorn.Server | None = None
    _thread: threading.Thread | None = None
    _socket: socket.socket | None = None
    base_url: str = ""

    def start(self) -> str:
        app = FastAPI()

        async def handler(request: Request) -> Response:
            body = await request.body()
            self.requests.append(
                CapturedRequest(
                    path=request.url.path,
                    headers=dict(request.headers),
                    raw_body=body,
                )
            )
            if self.json_body is not None or self.status_code >= 400:
                return JSONResponse(
                    self.json_body if self.json_body is not None else {"error": {"message": "boom"}},
                    status_code=self.status_code,
                )

            async def stream() -> Any:
                for index, chunk in enumerate(self.chunks):
                    if self.drop_after is not None and index >= self.drop_after:
                        raise RuntimeError("simulated backend crash")
                    if self.chunk_delay_s:
                        await asyncio.sleep(self.chunk_delay_s)
                    yield chunk

            return StreamingResponse(
                stream(), status_code=self.status_code, media_type="text/event-stream"
            )

        app.post("/v1/responses")(handler)
        app.post("/v1/chat/completions")(handler)

        @app.get("/v1/models")
        async def models() -> Any:
            return {"object": "list", "data": [{"id": "mock-model"}]}

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(64)
        port = sock.getsockname()[1]
        self._socket = sock

        server = uvicorn.Server(
            uvicorn.Config(app, log_level="error", access_log=False, lifespan="on")
        )
        self._server = server
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        self._thread = thread

        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise TimeoutError("mock backend did not start")
            time.sleep(0.02)

        self.base_url = f"http://127.0.0.1:{port}/v1"
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass

    @property
    def last_request(self) -> CapturedRequest:
        assert self.requests, "mock backend received no requests"
        return self.requests[-1]


@pytest.fixture
def backend() -> Iterator[MockBackend]:
    mock = MockBackend()
    mock.start()
    yield mock
    mock.stop()


# ---------------------------------------------------------------------------
# SSE fixtures shaped like real traffic
# ---------------------------------------------------------------------------


def sse(event_type: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def responses_text_stream(text: str = "Hello from the backend.") -> list[bytes]:
    """A minimal but complete Responses-API stream producing one text message."""
    message = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    return [
        sse("response.created", {"type": "response.created",
                                 "response": {"id": "resp_1", "status": "in_progress", "output": []}}),
        sse("response.output_text.delta", {"type": "response.output_text.delta",
                                           "output_index": 0, "delta": text}),
        sse("response.output_item.done", {"type": "response.output_item.done",
                                          "output_index": 0, "item": message}),
        sse("response.completed", {"type": "response.completed",
                                   "response": {"id": "resp_1", "status": "completed",
                                                "output": [message],
                                                "usage": {"input_tokens": 10,
                                                          "output_tokens": 5,
                                                          "total_tokens": 15}}}),
    ]


def responses_tool_call_stream() -> list[bytes]:
    """A Responses-API stream producing two parallel function calls."""
    call_a = {"type": "function_call", "id": "fc_a", "call_id": "call_a",
              "name": "shell", "arguments": '{"command":["ls"]}', "status": "completed"}
    call_b = {"type": "function_call", "id": "fc_b", "call_id": "call_b",
              "name": "read_file", "arguments": '{"path":"a.py"}', "status": "completed"}
    return [
        sse("response.created", {"type": "response.created",
                                 "response": {"id": "resp_2", "status": "in_progress", "output": []}}),
        sse("response.output_item.done", {"type": "response.output_item.done",
                                          "output_index": 0, "item": call_a}),
        sse("response.output_item.done", {"type": "response.output_item.done",
                                          "output_index": 1, "item": call_b}),
        sse("response.completed", {"type": "response.completed",
                                   "response": {"id": "resp_2", "status": "completed",
                                                "output": [call_a, call_b]}}),
    ]


def chat_chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
    return f"data: {json.dumps({'id': 'chatcmpl-1', 'object': 'chat.completion.chunk', 'model': 'mock-model', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n".encode()


def chat_text_stream(text: str = "Hi there") -> list[bytes]:
    """A Chat Completions stream with the text split across several chunks."""
    chunks = [chat_chunk({"role": "assistant", "content": ""})]
    for piece in [text[i : i + 3] for i in range(0, len(text), 3)]:
        chunks.append(chat_chunk({"content": piece}))
    chunks.append(chat_chunk({}, finish="stop"))
    chunks.append(b"data: [DONE]\n\n")
    return chunks


def chat_tool_call_stream() -> list[bytes]:
    """A Chat stream whose tool-call arguments are split across many chunks.

    This is the shape that breaks naive translators: the JSON is only valid once
    the per-index fragments are joined.
    """
    return [
        chat_chunk({"role": "assistant", "content": None}),
        chat_chunk({"tool_calls": [{"index": 0, "id": "call_x", "type": "function",
                                    "function": {"name": "shell", "arguments": ""}}]}),
        chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"comm'}}]}),
        chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'and":["l'}}]}),
        chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": 's"]}'}}]}),
        chat_chunk({}, finish="tool_calls"),
        b"data: [DONE]\n\n",
    ]


@pytest.fixture
def probe_config(backend: MockBackend, tmp_path: Any) -> dict[str, Any]:
    """A passthrough config pointed at the mock backend, logging to tmp_path."""
    return {
        "backend": {"base_url": backend.base_url, "wire_api": "responses"},
        "listen": {"host": "127.0.0.1", "port": 0},
        "log_dir": str(tmp_path / "logs"),
        "session_id": "test-session",
    }
