"""Stream reassembly tests for both wire formats."""

from __future__ import annotations

import json

from codex_probe.wire.chat import ChatReassembler
from codex_probe.wire.responses import ResponsesReassembler, extract_function_calls
from codex_probe.wire.sse import SSEParser

from conftest import (
    chat_chunk,
    chat_text_stream,
    chat_tool_call_stream,
    responses_text_stream,
    responses_tool_call_stream,
    sse,
)


def events_from(chunks: list[bytes]) -> list:
    parser = SSEParser()
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.flush())
    return events


def reassemble_responses(chunks: list[bytes]) -> dict:
    reassembler = ResponsesReassembler()
    for event in events_from(chunks):
        reassembler.handle(event)
    return reassembler.result()


def reassemble_chat(chunks: list[bytes]) -> ChatReassembler:
    reassembler = ChatReassembler()
    for event in events_from(chunks):
        reassembler.handle(event)
    return reassembler


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------


def test_responses_text_reassembly():
    result = reassemble_responses(responses_text_stream("Hello world"))
    assert result["status"] == "completed"
    assert result["output_text"] == "Hello world"
    assert result["response"]["usage"]["total_tokens"] == 15


def test_responses_parallel_tool_calls_keep_their_order():
    result = reassemble_responses(responses_tool_call_stream())
    calls = extract_function_calls(result["response"]["output"])
    assert [c["name"] for c in calls] == ["shell", "read_file"]
    assert json.loads(calls[0]["arguments"]) == {"command": ["ls"]}


def test_responses_completed_event_is_authoritative():
    """When `response.completed` disagrees with the items, it wins."""
    stale = {"type": "message", "id": "msg_1", "role": "assistant",
             "content": [{"type": "output_text", "text": "stale"}]}
    fresh = {"type": "message", "id": "msg_1", "role": "assistant",
             "content": [{"type": "output_text", "text": "authoritative"}]}
    chunks = [
        sse("response.output_item.done",
            {"type": "response.output_item.done", "output_index": 0, "item": stale}),
        sse("response.completed",
            {"type": "response.completed",
             "response": {"id": "r", "status": "completed", "output": [fresh]}}),
    ]
    assert reassemble_responses(chunks)["output_text"] == "authoritative"


def test_responses_truncated_stream_falls_back_to_items():
    """A stream that dies before `response.completed` still yields what arrived."""
    chunks = responses_text_stream("partial answer")[:-1]  # drop response.completed
    result = reassemble_responses(chunks)
    assert result["status"] == "incomplete"
    assert result["output_text"] == "partial answer"


def test_responses_text_deltas_survive_without_any_item():
    """Worst case: only deltas arrived before the backend died."""
    chunks = [
        sse("response.created", {"type": "response.created", "response": {"id": "r"}}),
        sse("response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "half a sen"}),
    ]
    result = reassemble_responses(chunks)
    assert result["status"] == "incomplete"
    assert result["output_text"] == "half a sen"


def test_responses_failed_event_records_error():
    chunks = [
        sse("response.failed",
            {"type": "response.failed",
             "response": {"id": "r", "status": "failed",
                          "error": {"code": "rate_limit_exceeded", "message": "slow down"}}}),
    ]
    result = reassemble_responses(chunks)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "rate_limit_exceeded"


def test_responses_ignores_unparseable_frames():
    chunks = [b"data: garbage\n\n", *responses_text_stream("fine")]
    assert reassemble_responses(chunks)["output_text"] == "fine"


def test_extract_function_calls_on_junk_input():
    assert extract_function_calls(None) == []
    assert extract_function_calls("not a list") == []
    assert extract_function_calls([{"type": "message"}, "junk", 42]) == []


# ---------------------------------------------------------------------------
# Chat Completions API
# ---------------------------------------------------------------------------


def test_chat_text_reassembly_across_chunks():
    reassembler = reassemble_chat(chat_text_stream("Hi there, friend"))
    assert reassembler.content == "Hi there, friend"
    assert reassembler.finish_reason == "stop"


def test_chat_tool_call_arguments_reassemble_into_valid_json():
    """The core Chat-format hazard: arguments arrive as fragments."""
    reassembler = reassemble_chat(chat_tool_call_stream())
    calls = reassembler.tool_calls()
    assert len(calls) == 1
    assert calls[0]["id"] == "call_x"
    assert calls[0]["function"]["name"] == "shell"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": ["ls"]}


def test_chat_parallel_tool_calls_with_interleaved_indices():
    """Fragments for two calls interleave; per-index buckets must keep them apart.

    Concatenating in arrival order would splice the two argument strings together
    and produce invalid JSON.
    """
    chunks = [
        chat_chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                                    "function": {"name": "shell", "arguments": '{"cmd":'}}]}),
        chat_chunk({"tool_calls": [{"index": 1, "id": "call_b", "type": "function",
                                    "function": {"name": "read", "arguments": '{"path":'}}]}),
        chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"ls"}'}}]}),
        chat_chunk({"tool_calls": [{"index": 1, "function": {"arguments": '"a.py"}'}}]}),
        chat_chunk({}, finish="tool_calls"),
    ]
    calls = reassemble_chat(chunks).tool_calls()
    assert [c["id"] for c in calls] == ["call_a", "call_b"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"cmd": "ls"}
    assert json.loads(calls[1]["function"]["arguments"]) == {"path": "a.py"}


def test_chat_synthesises_call_id_when_backend_omits_it():
    """Many OSS servers never send an id; Codex needs one to match the result back."""
    chunks = [
        chat_chunk({"tool_calls": [{"index": 0,
                                    "function": {"name": "shell", "arguments": "{}"}}]}),
        chat_chunk({}, finish="tool_calls"),
    ]
    calls = reassemble_chat(chunks).tool_calls()
    assert calls[0]["id"] == "call_0"


def test_chat_captures_reasoning_content_separately():
    """Qwen/DeepSeek emit reasoning_content; it must not leak into message text."""
    chunks = [
        chat_chunk({"reasoning_content": "thinking..."}),
        chat_chunk({"content": "answer"}),
        chat_chunk({}, finish="stop"),
    ]
    reassembler = reassemble_chat(chunks)
    assert reassembler.content == "answer"
    assert reassembler.reasoning == "thinking..."


def test_chat_usage_from_terminal_chunk():
    chunks = [
        chat_chunk({"content": "x"}),
        chat_chunk({}, finish="stop"),
        f'data: {json.dumps({"id": "c", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}})}\n\n'.encode(),
    ]
    assert reassemble_chat(chunks).usage["total_tokens"] == 9


def test_chat_finish_reason_length_is_preserved():
    chunks = [chat_chunk({"content": "truncated"}), chat_chunk({}, finish="length")]
    assert reassemble_chat(chunks).finish_reason == "length"


def test_chat_empty_and_malformed_chunks_are_survivable():
    chunks = [b"data: \n\n", b"data: {bad\n\n", chat_chunk({"content": "ok"}),
              chat_chunk({}, finish="stop")]
    assert reassemble_chat(chunks).content == "ok"
