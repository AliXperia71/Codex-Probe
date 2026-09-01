"""Incremental Server-Sent Events parser.

Both the Responses API and the Chat Completions API stream over SSE, so this is
the shared foundation for reassembling either one.

"Incremental" is the whole point. Bytes arrive from the backend in arbitrary
chunks that have nothing to do with event boundaries: a single SSE frame can be
split across three TCP reads, and one read can contain five frames plus half of a
sixth. A parser that assumes "one chunk == one event" appears to work against a
fast local server and then silently drops or corrupts events against a real
network. So this parser buffers and only emits frames it has seen terminated.

Two subtleties that are easy to get wrong:

1. **UTF-8 across chunk boundaries.** A multi-byte character can straddle two
   chunks, so we decode through `codecs.getincrementaldecoder`, which holds the
   partial sequence rather than raising or emitting a replacement character.
2. **A trailing bare CR is ambiguous.** SSE accepts CRLF, LF, or CR as a line
   terminator. If a chunk ends in "\\r" we cannot yet tell whether it is a CR
   terminator or the first half of a CRLF, so we hold it back until more bytes
   arrive or the stream ends.
"""

from __future__ import annotations

import codecs
import json
from dataclasses import dataclass, field
from typing import Any

# The sentinel the Chat Completions API sends to close a stream. The Responses
# API does not use it (it sends `response.completed` instead), but tolerating it
# in both directions costs nothing.
DONE_SENTINEL = "[DONE]"


@dataclass
class SSEEvent:
    """One dispatched SSE frame."""

    event: str | None = None
    data: str = ""
    id: str | None = None
    retry: int | None = None
    raw: str = ""
    """The frame exactly as received, minus the terminating blank line.

    Kept so the log can satisfy requirement 3 (full, unabridged capture) even when
    the frame fails to parse as JSON.
    """

    def json(self) -> Any | None:
        """Parse `data` as JSON, or return None if it is absent, `[DONE]`, or malformed.

        Returning None rather than raising is deliberate: a single malformed frame
        from a flaky backend must degrade one log entry, never break the relay to
        Codex or abort reassembly of the frames around it.
        """
        if not self.data or self.is_done():
            return None
        try:
            return json.loads(self.data)
        except json.JSONDecodeError:
            return None

    def is_done(self) -> bool:
        return self.data.strip() == DONE_SENTINEL

    def kind(self) -> str | None:
        """The event type.

        OpenAI sets it both in the SSE `event:` field and as a `type` key inside the
        JSON payload. The payload is the more reliable of the two -- some
        OpenAI-compatible servers omit the `event:` line entirely -- so it wins.
        """
        payload = self.json()
        if isinstance(payload, dict):
            type_value = payload.get("type")
            if isinstance(type_value, str):
                return type_value
        return self.event


class SSEParser:
    """Feed it bytes, get back complete `SSEEvent`s.

    Not thread-safe, and not meant to be: one parser instance belongs to exactly
    one response stream.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buf = ""
        # Accumulators for the frame currently being built.
        self._event: str | None = None
        self._data: list[str] = []
        self._id: str | None = None
        self._retry: int | None = None
        self._raw_lines: list[str] = []

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        """Consume a chunk of bytes, returning every frame completed by it."""
        text = self._decoder.decode(chunk)
        if not text:
            return []
        self._buf += text
        return self._drain(final=False)

    def flush(self) -> list[SSEEvent]:
        """Close the stream and dispatch any frame not terminated by a blank line.

        Well-behaved servers end with a blank line, so this usually returns []. It
        matters when a backend dies mid-stream: the partial frame still reaches the
        log instead of vanishing.
        """
        self._buf += self._decoder.decode(b"", final=True)
        events = self._drain(final=True)
        trailing = self._dispatch()
        if trailing is not None:
            events.append(trailing)
        return events

    def _drain(self, final: bool) -> list[SSEEvent]:
        buf = self._buf

        # Hold back an ambiguous trailing CR (see module docstring, note 2).
        held = ""
        if not final and buf.endswith("\r"):
            held = "\r"
            buf = buf[:-1]

        buf = buf.replace("\r\n", "\n").replace("\r", "\n")
        lines = buf.split("\n")

        # Unless the stream is over, the final element is an unterminated line.
        incomplete = "" if final else lines.pop()

        events: list[SSEEvent] = []
        for line in lines:
            dispatched = self._consume_line(line)
            if dispatched is not None:
                events.append(dispatched)

        self._buf = incomplete + held
        return events

    def _consume_line(self, line: str) -> SSEEvent | None:
        if line == "":
            return self._dispatch()

        self._raw_lines.append(line)

        # A line beginning with ':' is a comment. Servers use these as keepalives to
        # stop idle proxies from timing out; they carry no payload.
        if line.startswith(":"):
            return None

        field_name, sep, value = line.partition(":")
        if not sep:
            # A line with no colon is a field name with an empty value.
            field_name, value = line, ""
        elif value.startswith(" "):
            # Exactly one leading space after the colon is stripped, per spec.
            value = value[1:]

        if field_name == "event":
            self._event = value
        elif field_name == "data":
            self._data.append(value)
        elif field_name == "id":
            self._id = value
        elif field_name == "retry":
            try:
                self._retry = int(value)
            except ValueError:
                pass  # Spec says ignore a non-integer retry rather than fail.
        # Unknown field names are ignored, per spec.
        return None

    def _dispatch(self) -> SSEEvent | None:
        """Emit the pending frame, if there is one worth emitting."""
        if not self._raw_lines:
            return None  # A blank line with nothing pending: no-op.

        # Comment-only frames (keepalives) carry nothing to reassemble.
        if not self._data and self._event is None:
            self._reset()
            return None

        event = SSEEvent(
            event=self._event,
            # Multiple `data:` lines in one frame are joined with newlines, per spec.
            data="\n".join(self._data),
            id=self._id,
            retry=self._retry,
            raw="\n".join(self._raw_lines),
        )
        self._reset()
        return event

    def _reset(self) -> None:
        self._event = None
        self._data = []
        self._id = None
        self._retry = None
        self._raw_lines = []


def format_sse(event_type: str, payload: Any) -> bytes:
    """Serialise one frame in the shape Codex expects.

    Used by translate mode, which synthesises Responses-API frames rather than
    relaying them. Both the `event:` line and the payload's `type` key are written,
    because Codex reads the latter and other tooling reads the former.
    """
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {body}\n\n".encode("utf-8")
