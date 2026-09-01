"""Responses <-> Chat Completions translation tests.

The response-direction tests are the important ones. Codex builds conversation
state from `response.output_item.done` and `response.completed`; a translator
that emits only text deltas renders on screen and then loses the turn. These
tests pin that contract down.
"""

from __future__ import annotations

import json

from codex_probe.wire.sse import SSEParser
from codex_probe.wire.translate import (
    ChatToResponsesTranslator,
    responses_request_to_chat,
)

from conftest import chat_chunk, chat_text_stream, chat_tool_call_stream

# ---------------------------------------------------------------------------
# Direction 1: Responses request -> Chat request
# ---------------------------------------------------------------------------


def test_instructions_become_a_leading_system_message():
    chat = responses_request_to_chat(
        {"instructions": "You are a coding agent.", "input": "hi", "model": "m"}
    )
    assert chat["messages"][0] == {"role": "system", "content": "You are a coding agent."}
    assert chat["messages"][1] == {"role": "user", "content": "hi"}
    assert chat["model"] == "m"
    assert chat["stream"] is True


def test_bare_string_input_becomes_one_user_message():
    chat = responses_request_to_chat({"input": "just text"})
    assert chat["messages"] == [{"role": "user", "content": "just text"}]


def test_message_items_unwrap_content_parts():
    chat = responses_request_to_chat(
        {
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "part one "},
                             {"type": "input_text", "text": "part two"}]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "reply"}]},
            ]
        }
    )
    assert chat["messages"] == [
        {"role": "user", "content": "part one part two"},
        {"role": "assistant", "content": "reply"},
    ]


def test_function_call_and_output_round_trip():
    chat = responses_request_to_chat(
        {
            "input": [
                {"type": "function_call", "call_id": "call_1", "name": "shell",
                 "arguments": '{"command":["ls"]}'},
                {"type": "function_call_output", "call_id": "call_1", "output": "a.py\n"},
            ]
        }
    )
    assistant, tool = chat["messages"]
    assert assistant["role"] == "assistant"
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "shell"
    assert tool == {"role": "tool", "tool_call_id": "call_1", "content": "a.py\n"}


def test_parallel_function_calls_merge_into_one_assistant_message():
    """Responses lists parallel calls separately; Chat wants them in one message."""
    chat = responses_request_to_chat(
        {
            "input": [
                {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
                {"type": "function_call", "call_id": "b", "name": "g", "arguments": "{}"},
            ]
        }
    )
    assert len(chat["messages"]) == 1
    assert [c["id"] for c in chat["messages"][0]["tool_calls"]] == ["a", "b"]


def test_non_string_tool_output_is_stringified():
    chat = responses_request_to_chat(
        {"input": [{"type": "function_call_output", "call_id": "c",
                    "output": {"stdout": "ok", "exit_code": 0}}]}
    )
    assert json.loads(chat["messages"][0]["content"]) == {"stdout": "ok", "exit_code": 0}


def test_reasoning_items_are_dropped():
    chat = responses_request_to_chat(
        {"input": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "hm"}]},
                   {"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": "go"}]}]}
    )
    assert chat["messages"] == [{"role": "user", "content": "go"}]


def test_tool_schemas_flatten_to_nested_form():
    params = {"type": "object", "properties": {"path": {"type": "string"}}}
    chat = responses_request_to_chat(
        {"input": "x",
         "tools": [{"type": "function", "name": "read_file",
                    "description": "Read a file", "parameters": params}]}
    )
    assert chat["tools"] == [
        {"type": "function",
         "function": {"name": "read_file", "description": "Read a file", "parameters": params}}
    ]


def test_builtin_server_side_tools_are_dropped():
    """web_search has no Chat equivalent; forwarding it would break strict servers."""
    chat = responses_request_to_chat(
        {"input": "x", "tools": [{"type": "web_search"},
                                 {"type": "function", "name": "f", "parameters": {}}]}
    )
    assert len(chat["tools"]) == 1
    assert chat["tools"][0]["function"]["name"] == "f"


def test_tool_choice_conversion():
    chat = responses_request_to_chat(
        {"input": "x", "tools": [{"type": "function", "name": "f", "parameters": {}}],
         "tool_choice": {"type": "function", "name": "f"}}
    )
    assert chat["tool_choice"] == {"type": "function", "function": {"name": "f"}}


def test_max_output_tokens_is_renamed():
    chat = responses_request_to_chat({"input": "x", "max_output_tokens": 512})
    assert chat["max_tokens"] == 512
    assert "max_output_tokens" not in chat


def test_responses_only_fields_are_dropped():
    chat = responses_request_to_chat(
        {"input": "x", "store": True, "previous_response_id": "resp_1",
         "include": ["reasoning.encrypted_content"], "reasoning": {"effort": "high"},
         "prompt_cache_key": "k"}
    )
    for dropped in ("store", "previous_response_id", "include", "reasoning", "prompt_cache_key"):
        assert dropped not in chat


def test_image_content_becomes_chat_image_url():
    chat = responses_request_to_chat(
        {"input": [{"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": "what is this?"},
                                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}]}
    )
    content = chat["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_empty_and_unknown_input_does_not_crash():
    assert responses_request_to_chat({"input": []})["messages"] == []
    assert responses_request_to_chat({})["messages"] == []
    chat = responses_request_to_chat({"input": [{"type": "some_future_item"}, "junk", 42]})
    assert chat["messages"] == []


# ---------------------------------------------------------------------------
# Direction 2: Chat SSE -> Responses SSE
# ---------------------------------------------------------------------------


def translate(chunks: list[bytes]) -> list:
    """Drive a translator over a Chat stream and parse the frames it emits."""
    translator = ChatToResponsesTranslator(model="mock-model")
    out: list[bytes] = list(translator.start())

    parser = SSEParser()
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.flush())
    for event in events:
        out.extend(translator.consume(event))
    out.extend(translator.finish())

    emitted_parser = SSEParser()
    emitted = []
    for frame in out:
        emitted.extend(emitted_parser.feed(frame))
    emitted.extend(emitted_parser.flush())
    return emitted


def test_text_stream_emits_the_required_event_sequence():
    events = translate(chat_text_stream("Hi there"))
    kinds = [e.kind() for e in events]

    assert kinds[0] == "response.created"
    assert kinds[-1] == "response.completed"
    # The load-bearing event: without it Codex renders the text then forgets it.
    assert "response.output_item.done" in kinds
    # And the cosmetic ones that keep the CLI responsive.
    assert kinds.count("response.output_text.delta") > 1


def test_text_deltas_concatenate_to_the_full_message():
    events = translate(chat_text_stream("Hi there, friend"))
    deltas = "".join(
        e.json()["delta"] for e in events if e.kind() == "response.output_text.delta"
    )
    assert deltas == "Hi there, friend"

    completed = [e for e in events if e.kind() == "response.completed"][0]
    message = completed.json()["response"]["output"][0]
    assert message["content"][0]["text"] == "Hi there, friend"


def test_tool_call_becomes_a_completed_function_call_item():
    events = translate(chat_tool_call_stream())

    done = [e for e in events if e.kind() == "response.output_item.done"]
    assert len(done) == 1
    item = done[0].json()["item"]
    assert item["type"] == "function_call"
    assert item["name"] == "shell"
    assert item["call_id"] == "call_x"
    assert json.loads(item["arguments"]) == {"command": ["ls"]}
    assert item["status"] == "completed"


def test_completed_response_carries_every_output_item():
    chunks = [
        chat_chunk({"content": "Running that now."}),
        chat_chunk({"tool_calls": [{"index": 0, "id": "call_z", "type": "function",
                                    "function": {"name": "shell", "arguments": "{}"}}]}),
        chat_chunk({}, finish="tool_calls"),
    ]
    events = translate(chunks)
    output = [e for e in events if e.kind() == "response.completed"][0].json()["response"]["output"]
    assert [item["type"] for item in output] == ["message", "function_call"]


def test_usage_is_translated_to_responses_names():
    chunks = [
        chat_chunk({"content": "x"}),
        chat_chunk({}, finish="stop"),
        f'data: {json.dumps({"id": "c", "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}})}\n\n'.encode(),
    ]
    events = translate(chunks)
    usage = [e for e in events if e.kind() == "response.completed"][0].json()["response"]["usage"]
    assert usage == {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15}


def test_finish_reason_length_maps_to_response_incomplete():
    """Truncation must be distinguishable from a finished turn."""
    events = translate([chat_chunk({"content": "cut off"}), chat_chunk({}, finish="length")])
    assert [e.kind() for e in events][-1] == "response.incomplete"


def test_empty_stream_still_terminates_cleanly():
    """No content at all must still produce created + completed, never a hang."""
    kinds = [e.kind() for e in translate([])]
    assert kinds == ["response.created", "response.completed"]


def test_sequence_numbers_are_monotonic():
    events = translate(chat_text_stream("abcdef"))
    numbers = [e.json()["sequence_number"] for e in events]
    assert numbers == sorted(numbers)
    assert numbers == list(range(len(numbers)))
