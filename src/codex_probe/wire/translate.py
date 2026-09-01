"""Translate between the Responses API and the Chat Completions API.

This module is the reason CodexProbe can talk to a backend that Codex itself can
no longer talk to. Since February 2026 Codex only speaks the Responses API
(`WireApi` in `codex-rs/model-provider-info/src/lib.rs` has exactly one variant,
and `wire_api = "chat"` is a hard error). Plenty of OpenAI-compatible servers --
llama.cpp, older vLLM builds, LM Studio, most third-party gateways -- only speak
Chat Completions. The proxy bridges that gap, effectively reintroducing at the
proxy layer the `wire_api = "chat"` option that Codex dropped.

Two directions, and they are not symmetric:

    request:  Responses  ->  Chat        (structural, one-shot)
    response: Chat SSE   ->  Responses SSE  (stateful, streaming)

The asymmetry is the interesting part. The request direction is a pure data
reshape. The response direction has to *invent* structure the Chat format does
not carry: Chat streams a flat sequence of token deltas, while Responses streams
addressed, individually-completed items. So the translator must buffer the whole
message, decide where item boundaries fall, and emit them as completed items.

The load-bearing detail, read off Codex's parser in
`codex-rs/codex-api/src/sse/responses.rs`: Codex builds its conversation state
from `response.output_item.done` and `response.completed`. It *ignores*
`response.function_call_arguments.delta` entirely, and treats
`response.output_text.delta` as cosmetic UI text. A translator that emits only
deltas therefore renders correctly on screen and then loses the turn -- the agent
forgets what it just said and never sees its own tool call. Emitting the
completed items is not optional polish; it is the whole contract.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterable

from .chat import ChatReassembler
from .sse import SSEEvent, format_sse

# ---------------------------------------------------------------------------
# Direction 1: Responses request -> Chat Completions request
# ---------------------------------------------------------------------------

# Responses-only fields with no Chat equivalent. Forwarding them would make
# strict servers reject the request outright, so they are dropped.
_DROPPED_REQUEST_FIELDS = frozenset(
    {
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "max_output_tokens",
        "store",
        "previous_response_id",
        "include",
        "reasoning",
        "text",
        "truncation",
        "prompt_cache_key",
        "service_tier",
        "client_metadata",
        "stream_options",
        "parallel_tool_calls",
        "prompt",
        "background",
    }
)


def responses_request_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses request body into a Chat Completions request body."""
    messages: list[dict[str, Any]] = []

    # `instructions` is the Responses API's dedicated system-prompt slot. Chat has
    # no such slot, so it becomes a leading system message.
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})

    messages.extend(_input_to_messages(body.get("input")))

    chat: dict[str, Any] = {
        "messages": messages,
        "stream": True,
    }

    # Carry over the fields that mean the same thing in both formats.
    for key in ("model", "temperature", "top_p", "seed", "stop", "user"):
        if key in body and body[key] is not None:
            chat[key] = body[key]

    # Renamed rather than dropped.
    if isinstance(body.get("max_output_tokens"), int):
        chat["max_tokens"] = body["max_output_tokens"]
    if isinstance(body.get("parallel_tool_calls"), bool):
        chat["parallel_tool_calls"] = body["parallel_tool_calls"]

    tools = _tools_to_chat(body.get("tools"))
    if tools:
        chat["tools"] = tools
        tool_choice = _tool_choice_to_chat(body.get("tool_choice"))
        if tool_choice is not None:
            chat["tool_choice"] = tool_choice

    # Ask for usage numbers in the terminal chunk so the log can report them.
    chat["stream_options"] = {"include_usage": True}

    # Preserve any vendor extension we do not recognise, so a custom backend's
    # own knobs survive the trip.
    for key, value in body.items():
        if key not in _DROPPED_REQUEST_FIELDS and key not in chat and key != "stream":
            chat[key] = value

    return chat


def _input_to_messages(value: Any) -> list[dict[str, Any]]:
    """Convert a Responses `input` into a Chat `messages` array."""
    # The Responses API accepts a bare string as shorthand for a single user turn.
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        return []

    messages: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        _append_item(messages, item)
    return messages


def _append_item(messages: list[dict[str, Any]], item: dict[str, Any]) -> None:
    """Fold one Responses input item into the growing Chat message list."""
    item_type = item.get("type")

    # A function call the assistant previously made.
    if item_type == "function_call":
        call = {
            "id": item.get("call_id") or item.get("id") or f"call_{len(messages)}",
            "type": "function",
            "function": {
                "name": item.get("name") or "",
                "arguments": item.get("arguments") or "{}",
            },
        }
        # Chat represents parallel tool calls as ONE assistant message carrying
        # several entries in `tool_calls`, whereas Responses lists them as separate
        # items. So consecutive function_calls merge into the previous message.
        if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
            messages[-1]["tool_calls"].append(call)
        else:
            messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        return

    # The result of running that call.
    if item_type == "function_call_output":
        messages.append(
            {
                "role": "tool",
                "tool_call_id": item.get("call_id") or "",
                "content": _stringify_output(item.get("output")),
            }
        )
        return

    # Chain-of-thought items have no Chat representation. Dropping them is correct
    # rather than lossy in any way that matters: the Chat backend never produced
    # them and cannot consume them.
    if item_type == "reasoning":
        return

    # Everything else is a message: either {type: "message", ...} or the bare
    # {role, content} shorthand the Responses API also accepts.
    role = item.get("role")
    if not isinstance(role, str):
        return
    content = _content_to_chat(item.get("content"))
    if role == "assistant" and content is None:
        return
    messages.append({"role": role, "content": content if content is not None else ""})


def _content_to_chat(content: Any) -> Any:
    """Convert Responses content parts into Chat content.

    Returns a plain string for the common text-only case, because some servers
    reject the array form; returns the array form only when an image is present.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    text_parts: list[str] = []
    rich_parts: list[dict[str, Any]] = []
    has_image = False

    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            rich_parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in ("input_text", "output_text", "text", "summary_text"):
            text = part.get("text") or ""
            text_parts.append(text)
            rich_parts.append({"type": "text", "text": text})
        elif part_type == "refusal":
            text = part.get("refusal") or ""
            text_parts.append(text)
            rich_parts.append({"type": "text", "text": text})
        elif part_type in ("input_image", "image_url"):
            url = part.get("image_url") or part.get("url")
            if isinstance(url, dict):
                url = url.get("url")
            if isinstance(url, str) and url:
                has_image = True
                rich_parts.append({"type": "image_url", "image_url": {"url": url}})

    if has_image:
        return rich_parts
    return "".join(text_parts)


def _stringify_output(output: Any) -> str:
    """Tool results must reach a Chat backend as a string."""
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    return json.dumps(output, ensure_ascii=False)


def _tools_to_chat(tools: Any) -> list[dict[str, Any]]:
    """Flatten Responses tool schemas into Chat's nested form.

    Responses: {"type": "function", "name": ..., "parameters": {...}}
    Chat:      {"type": "function", "function": {"name": ..., "parameters": {...}}}
    """
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        # Already nested (a caller mixing formats): pass through untouched.
        if isinstance(tool.get("function"), dict):
            converted.append(tool)
            continue
        if tool.get("type") != "function":
            # Built-in server-side tools (web_search, file_search, ...) have no Chat
            # equivalent. Dropping them is the only honest option.
            continue
        function: dict[str, Any] = {
            "name": tool.get("name") or "",
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        }
        if tool.get("description"):
            function["description"] = tool["description"]
        if isinstance(tool.get("strict"), bool):
            function["strict"] = tool["strict"]
        converted.append({"type": "function", "function": function})
    return converted


def _tool_choice_to_chat(choice: Any) -> Any:
    if isinstance(choice, str):
        return choice if choice in ("auto", "none", "required") else "auto"
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = choice.get("name") or (choice.get("function") or {}).get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return None


# ---------------------------------------------------------------------------
# Direction 2: Chat Completions SSE -> Responses SSE
# ---------------------------------------------------------------------------


def _usage_to_responses(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Chat and Responses count the same tokens under different names."""
    if not isinstance(usage, dict):
        return None
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def chat_completion_to_responses(completion: dict[str, Any]) -> dict[str, Any]:
    """Convert a *non-streaming* Chat completion into a Responses object.

    The proxy always asks for `stream: true`, so this is the fallback for a
    backend that ignores the flag and answers with a single JSON body. Without
    it Codex would receive a `chat.completion` where it expects a `response` and
    fail to parse the turn.
    """
    choices = completion.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else None
    message = message if isinstance(message, dict) else {}

    response_id = completion.get("id") or f"resp_{uuid.uuid4().hex}"
    output: list[dict[str, Any]] = []

    content = message.get("content")
    if isinstance(content, str) and content:
        output.append(
            {
                "type": "message",
                "id": f"msg_{response_id}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        )

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for position, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            call_id = call.get("id") or f"call_{position}"
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{call_id}",
                    "call_id": call_id,
                    "name": function.get("name") or "",
                    "arguments": function.get("arguments") or "{}",
                    "status": "completed",
                }
            )

    status = "incomplete" if choice.get("finish_reason") == "length" else "completed"
    result: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": completion.get("created") or int(time.time()),
        "status": status,
        "model": completion.get("model"),
        "output": output,
    }
    usage = _usage_to_responses(completion.get("usage"))
    if usage is not None:
        result["usage"] = usage
    return result


class ChatToResponsesTranslator:
    """Turns a Chat Completions SSE stream into a Responses SSE stream.

    Usage mirrors the shape of the stream itself::

        translator = ChatToResponsesTranslator(model="qwen2.5-coder:7b")
        yield from translator.start()          # response.created
        for event in chat_events:
            yield from translator.consume(ev)  # live text deltas
        yield from translator.finish()         # completed items + response.completed

    Text deltas are forwarded live so the CLI keeps typing as tokens arrive. The
    completed items can only be emitted at the end, because Chat gives no signal
    that an item is finished until the stream stops.
    """

    def __init__(self, model: str | None = None, response_id: str | None = None) -> None:
        self.response_id = response_id or f"resp_{uuid.uuid4().hex}"
        self._model = model
        self._created_at = int(time.time())
        self._reassembler = ChatReassembler()
        self._sequence = 0

    # -- frame helpers -----------------------------------------------------

    def _frame(self, event_type: str, payload: dict[str, Any]) -> bytes:
        payload = {"type": event_type, "sequence_number": self._sequence, **payload}
        self._sequence += 1
        return format_sse(event_type, payload)

    def _response_envelope(self, status: str, output: list[dict[str, Any]]) -> dict[str, Any]:
        envelope: dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self._created_at,
            "status": status,
            "model": self._reassembler.model or self._model,
            "output": output,
        }
        usage = _usage_to_responses(self._reassembler.usage)
        if usage is not None:
            envelope["usage"] = usage
        return envelope

    # -- stream lifecycle --------------------------------------------------

    def start(self) -> Iterable[bytes]:
        yield self._frame("response.created", {"response": self._response_envelope("in_progress", [])})

    def consume(self, event: SSEEvent) -> Iterable[bytes]:
        """Fold one Chat chunk in, emitting any live-display frames it implies."""
        before = len(self._reassembler.content)
        self._reassembler.handle(event)
        after = self._reassembler.content

        # Forward newly-arrived text so Codex can render it as it streams. This is
        # cosmetic: the authoritative copy goes out in `finish()`.
        if len(after) > before:
            yield self._frame(
                "response.output_text.delta",
                {
                    "item_id": f"msg_{self.response_id}",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": after[before:],
                },
            )

    def finish(self) -> Iterable[bytes]:
        """Emit the completed items and terminate the response.

        This is the part Codex actually reads. See the module docstring.
        """
        output: list[dict[str, Any]] = []
        index = 0

        text = self._reassembler.content
        if text:
            message_item = {
                "type": "message",
                "id": f"msg_{self.response_id}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
            output.append(message_item)
            yield self._frame(
                "response.output_item.done", {"output_index": index, "item": message_item}
            )
            index += 1

        for call in self._reassembler.tool_calls():
            function = call.get("function") or {}
            call_item = {
                "type": "function_call",
                "id": f"fc_{call.get('id')}",
                "call_id": call.get("id"),
                "name": function.get("name") or "",
                "arguments": function.get("arguments") or "{}",
                "status": "completed",
            }
            output.append(call_item)
            yield self._frame(
                "response.output_item.done", {"output_index": index, "item": call_item}
            )
            index += 1

        # `length` means the backend hit its token ceiling mid-answer. Reporting it
        # as `incomplete` lets Codex distinguish truncation from a finished turn.
        status = "incomplete" if self._reassembler.finish_reason == "length" else "completed"
        event_name = "response.incomplete" if status == "incomplete" else "response.completed"
        yield self._frame(event_name, {"response": self._response_envelope(status, output)})

    def reassembled(self) -> dict[str, Any]:
        """The Chat-shaped completion, for the log's `backend_response` field."""
        return self._reassembler.result()

    def as_responses_object(self) -> dict[str, Any]:
        """The Responses-shaped result, for the log's `response` field."""
        output: list[dict[str, Any]] = []
        text = self._reassembler.content
        if text:
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{self.response_id}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            )
        for call in self._reassembler.tool_calls():
            function = call.get("function") or {}
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{call.get('id')}",
                    "call_id": call.get("id"),
                    "name": function.get("name") or "",
                    "arguments": function.get("arguments") or "{}",
                    "status": "completed",
                }
            )
        status = "incomplete" if self._reassembler.finish_reason == "length" else "completed"
        return self._response_envelope(status, output)
