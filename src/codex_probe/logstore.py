"""Per-session JSONL logging of captured LLM calls.

Requirement 6 asks for session-organised logs so a reviewer can replay a run
turn-by-turn. JSONL is the right shape for that: one call per line, appended as
it completes, readable with `jq` and streamable without loading the whole file.
A single JSON array would have to be rewritten on every append and would be
unreadable if the process died mid-run.

Requirement 3 forbids truncation, so nothing here elides large payloads. A long
Codex session legitimately produces multi-megabyte lines, because each call
carries the entire running conversation.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CallRecord(BaseModel):
    """One LLM call: everything that went out and everything that came back."""

    model_config = ConfigDict(extra="allow")

    call_index: int = Field(description="0-based arrival order within the session.")
    session_id: str
    timestamp_start: str
    timestamp_end: str | None = None
    latency_ms: float | None = None

    mode: str = Field(description="'passthrough' or 'translate'.")
    backend_url: str = Field(
        description="The exact URL the call was forwarded to. This is the field that "
        "proves whether a call hit OpenAI or a local server."
    )
    model: str | None = None
    status: str = Field(
        default="pending",
        description="completed | incomplete | failed | error. 'error' means the proxy "
        "or transport failed; 'failed' means the backend reported a model-level error.",
    )
    http_status: int | None = None

    request: dict[str, Any] | None = Field(
        default=None, description="The body Codex sent, decompressed. Responses-API shape."
    )
    backend_request: dict[str, Any] | None = Field(
        default=None,
        description="The body actually sent upstream. Identical to `request` in "
        "passthrough mode; the translated Chat body in translate mode.",
    )
    response: dict[str, Any] | None = Field(
        default=None, description="Reassembled response, Responses-API shape."
    )
    backend_response: dict[str, Any] | None = Field(
        default=None,
        description="Reassembled response in the backend's own format. Only set in "
        "translate mode, where it differs from `response`.",
    )

    request_headers: dict[str, str] = Field(default_factory=dict)
    raw_events: list[str] = Field(
        default_factory=list,
        description="Every SSE frame as received, verbatim. The last resort when "
        "reassembly cannot make sense of an unusual backend.",
    )
    output_text: str = ""
    function_calls: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class SessionLog:
    """Collects `CallRecord`s for one recorder session and mirrors them to disk.

    Guarded by a lock because uvicorn serves requests concurrently: Codex issues
    calls sequentially, but nothing stops a user pointing two Codex processes at
    one recorder, and interleaved writes would corrupt the file.
    """

    def __init__(self, session_id: str, log_dir: Path) -> None:
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{session_id}.jsonl"
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()
        self._next_index = 0

    def next_index(self) -> int:
        """Reserve the next arrival slot.

        Taken when a request *arrives*, not when it completes, so the log preserves
        the order Codex made the calls in even if a slow call finishes out of order.
        """
        with self._lock:
            index = self._next_index
            self._next_index += 1
            return index

    def append(self, record: CallRecord) -> None:
        """Record a completed call and flush it to disk immediately.

        Flushing per call rather than at `stop()` means a session that is killed
        mid-run still leaves a usable log -- which is exactly when you most want one.
        """
        line = record.model_dump_json()
        with self._lock:
            self._records.append(record)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def records(self) -> list[dict[str, Any]]:
        """Every call so far as plain dicts, in arrival order."""
        with self._lock:
            ordered = sorted(self._records, key=lambda r: r.call_index)
            return [r.model_dump() for r in ordered]

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def read_session(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL session log back into memory.

    Malformed trailing lines are skipped rather than fatal: a log truncated by a
    hard kill should still be readable up to the break.
    """
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
