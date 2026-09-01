"""Reassemble a streamed Chat Completions call into a single completion object.

Only translate mode uses this: it is how the proxy understands what a
Chat-Completions-only backend said, before re-expressing it as Responses-API
frames for Codex.

The one genuinely awkward part of the Chat streaming format is tool calls. A
call's arguments arrive as a sequence of string fragments spread across many
chunks, and the only thing tying the fragments together is the `index` field:

    delta.tool_calls[0].function.arguments = '{"pa'
    delta.tool_calls[0].function.arguments = 'th": "a.py'
    delta.tool_calls[0].function.arguments = '"}'

The `id` and `name` typically appear only on the first fragment. With parallel
tool calls, fragments for index 0 and index 1 interleave freely. So fragments
must be accumulated into per-index buckets and only joined at the end -- naively
concatenating in arrival order produces spliced-together garbage that parses as
invalid JSON.

This is the concrete reason the Responses API exists, incidentally: it models
output as addressed items rather than as a flat delta stream, so the client never
has to do this bookkeeping.
"""

from __future__ import annotations

from typing import Any

from .sse import SSEEvent


class _ToolCallAccumulator:
    """Fragments of one tool call, keyed by its stream index."""

    def __init__(self) -> None:
        self.id: str | None = None
        self.name: str | None = None
        self.arg_parts: list[str] = []

    def add(self, fragment: dict[str, Any]) -> None:
        call_id = fragment.get("id")
        if isinstance(call_id, str) and call_id:
            self.id = call_id
        function = fragment.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                self.name = name
            args = function.get("arguments")
            if isinstance(args, str):
                self.arg_parts.append(args)

    def build(self, fallback_index: int) -> dict[str, Any]:
        return {
            # Codex needs a stable call_id to match the eventual tool result back to
            # the call. Some OSS servers never send one, so synthesise a
            # deterministic stand-in rather than emitting null.
            "id": self.id or f"call_{fallback_index}",
            "type": "function",
            "function": {
                "name": self.name or "",
                "arguments": "".join(self.arg_parts),
            },
        }


class ChatReassembler:
    """Accumulates Chat Completions SSE chunks into one completion object."""

    def __init__(self) -> None:
        self._id: str | None = None
        self._model: str | None = None
        self._created: int | None = None
        self._role = "assistant"
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._tool_calls: dict[int, _ToolCallAccumulator] = {}
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] | None = None

    def handle(self, event: SSEEvent) -> None:
        """Fold one chunk into the accumulated state. Never raises."""
        if event.is_done():
            return
        payload = event.json()
        if not isinstance(payload, dict):
            return

        if isinstance(payload.get("id"), str):
            self._id = payload["id"]
        if isinstance(payload.get("model"), str):
            self._model = payload["model"]
        if isinstance(payload.get("created"), int):
            self._created = payload["created"]
        # Present only when the request set stream_options.include_usage.
        if isinstance(payload.get("usage"), dict):
            self._usage = payload["usage"]

        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if isinstance(choice, dict):
                self._handle_choice(choice)

    def _handle_choice(self, choice: dict[str, Any]) -> None:
        finish = choice.get("finish_reason")
        if isinstance(finish, str):
            self._finish_reason = finish

        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return

        role = delta.get("role")
        if isinstance(role, str) and role:
            self._role = role

        content = delta.get("content")
        if isinstance(content, str):
            self._content_parts.append(content)

        # Emitted by reasoning models served through the Chat API (Qwen, DeepSeek).
        # Captured for the log; deliberately not surfaced to Codex as message text.
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str):
            self._reasoning_parts.append(reasoning)

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for position, fragment in enumerate(tool_calls):
                if not isinstance(fragment, dict):
                    continue
                index = fragment.get("index")
                if not isinstance(index, int):
                    index = position
                self._tool_calls.setdefault(index, _ToolCallAccumulator()).add(fragment)

    @property
    def content(self) -> str:
        return "".join(self._content_parts)

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning_parts)

    @property
    def finish_reason(self) -> str | None:
        return self._finish_reason

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def usage(self) -> dict[str, Any] | None:
        return self._usage

    def tool_calls(self) -> list[dict[str, Any]]:
        """Completed tool calls, ordered by stream index."""
        return [self._tool_calls[i].build(i) for i in sorted(self._tool_calls)]

    def message(self) -> dict[str, Any]:
        """The assembled assistant message, in Chat Completions shape."""
        message: dict[str, Any] = {"role": self._role, "content": self.content or None}
        calls = self.tool_calls()
        if calls:
            message["tool_calls"] = calls
        return message

    def result(self) -> dict[str, Any]:
        """The full non-streaming completion object, for the log."""
        return {
            "id": self._id,
            "object": "chat.completion",
            "created": self._created,
            "model": self._model,
            "choices": [
                {
                    "index": 0,
                    "message": self.message(),
                    "finish_reason": self._finish_reason,
                }
            ],
            "usage": self._usage,
        }
