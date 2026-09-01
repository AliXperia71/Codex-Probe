"""Wire-format handling: SSE parsing, stream reassembly, and protocol translation.

Split out from the server so every format concern is unit-testable without
starting a socket. `server.py` orchestrates; this package knows the formats.
"""

from .sse import SSEEvent, SSEParser, format_sse

__all__ = ["SSEEvent", "SSEParser", "format_sse"]
