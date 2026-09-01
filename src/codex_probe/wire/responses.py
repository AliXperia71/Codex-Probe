"""Reassemble a streamed Responses-API call into a single response object.

This runs on a *copy* of the byte stream, after the bytes have already been
relayed to Codex. Nothing here can affect what Codex receives -- see
`server.py` for the tee that guarantees it.

Reassembly strategy, in priority order:

1. If `response.completed` arrives, use its `response` object verbatim. It is the
   server's own authoritative final state, already containing every output item
   and the usage numbers, so trusting it beats re-deriving it from deltas.
2. If the stream ends without it (backend crashed, network dropped, user hit
   Ctrl-C), fall back to the `response.output_item.done` items collected so far
   and mark the record `incomplete`. A partial log is far more useful than none
   when you are debugging why a run died.
3. `response.failed` records the server-reported error.

The event vocabulary below was read off Codex's own parser,
`codex-rs/codex-api/src/sse/responses.rs`, so this module and Codex agree on
which events actually carry state.
"""

from __future__ import annotations

from typing import Any

from .sse import SSEEvent

# Events that carry the completed content of one output item. Codex builds its
# conversation state from these, not from the deltas.
EVENT_ITEM_DONE = "response.output_item.done"
EVENT_ITEM_ADDED = "response.output_item.added"
EVENT_CREATED = "response.created"
EVENT_COMPLETED = "response.completed"
EVENT_FAILED = "response.failed"
EVENT_INCOMPLETE = "response.incomplete"
EVENT_TEXT_DELTA = "response.output_text.delta"


class ResponsesReassembler:
    """Accumulates Responses-API SSE frames into one final response object."""

    def __init__(self) -> None:
        self._created: dict[str, Any] | None = None
        self._final: dict[str, Any] | None = None
        self._items: dict[int, dict[str, Any]] = {}
        self._item_order: list[int] = []
        self._text_parts: list[str] = []
        self._status = "incomplete"
        self._error: Any = None

    def handle(self, event: SSEEvent) -> None:
        """Fold one frame into the accumulated state. Never raises."""
        kind = event.kind()
        if kind is None:
            return
        payload = event.json()
        if not isinstance(payload, dict):
            return

        if kind == EVENT_CREATED:
            self._created = payload.get("response")

        elif kind in (EVENT_ITEM_DONE, EVENT_ITEM_ADDED):
            item = payload.get("item")
            if isinstance(item, dict):
                # `output_index` keeps parallel tool calls in their own slots. Some
                # servers omit it, so fall back to append order.
                index = payload.get("output_index")
                if not isinstance(index, int):
                    index = len(self._item_order)
                # An `.added` item is a stub; a later `.done` for the same index
                # supersedes it. Recording both orders is why we overwrite freely.
                if index not in self._items:
                    self._item_order.append(index)
                if kind == EVENT_ITEM_DONE or index not in self._items:
                    self._items[index] = item

        elif kind == EVENT_TEXT_DELTA:
            delta = payload.get("delta")
            if isinstance(delta, str):
                self._text_parts.append(delta)

        elif kind == EVENT_COMPLETED:
            self._final = payload.get("response")
            self._status = "completed"

        elif kind == EVENT_FAILED:
            self._final = payload.get("response")
            self._status = "failed"
            if isinstance(self._final, dict):
                self._error = self._final.get("error")
            if self._error is None:
                self._error = payload.get("error")

        elif kind == EVENT_INCOMPLETE:
            self._final = payload.get("response")
            self._status = "incomplete"

    @property
    def items(self) -> list[dict[str, Any]]:
        """Output items in stream order, reconstructed from `output_item.*` events."""
        return [self._items[i] for i in sorted(self._item_order) if i in self._items]

    def output_text(self) -> str:
        """Concatenated assistant text.

        Prefers the authoritative final object; falls back to the accumulated
        deltas when the stream was truncated before `response.completed`.
        """
        if isinstance(self._final, dict):
            text = extract_output_text(self._final.get("output"))
            if text:
                return text
        text = extract_output_text(self.items)
        return text if text else "".join(self._text_parts)

    def result(self) -> dict[str, Any]:
        """The reassembled response, in the shape written to the log."""
        response = self._final
        if response is None:
            # Strategy 2: synthesise a best-effort response from what we saw.
            response = dict(self._created) if isinstance(self._created, dict) else {}
            response["output"] = self.items
        return {
            "status": self._status,
            "response": response,
            "output_text": self.output_text(),
            "error": self._error,
        }


def extract_output_text(output: Any) -> str:
    """Pull assistant text out of a Responses `output` array.

    Shape: output -> [ {type: "message", content: [ {type: "output_text", text} ]} ].
    Written defensively because this also runs over half-built items from a
    truncated stream, where any level may be missing or the wrong type.
    """
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def extract_function_calls(output: Any) -> list[dict[str, Any]]:
    """Return the `function_call` items from a Responses `output` array.

    Used by the demo scripts and tests to assert that a model actually emitted tool
    calls -- the specific thing that tends to break on small open models.
    """
    if not isinstance(output, list):
        return []
    return [
        item
        for item in output
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]
