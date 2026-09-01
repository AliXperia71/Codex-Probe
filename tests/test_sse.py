"""SSE parser tests.

These target the failure modes that actually bite in production: a parser that
assumes "one chunk == one event" passes a happy-path test against a fast local
server and then drops events against a real network.
"""

from __future__ import annotations

import json

from codex_probe.wire.sse import SSEParser, format_sse


def parse_all(chunks: list[bytes]) -> list:
    parser = SSEParser()
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.flush())
    return events


def test_single_complete_event():
    events = parse_all([b'event: ping\ndata: {"a": 1}\n\n'])
    assert len(events) == 1
    assert events[0].event == "ping"
    assert events[0].json() == {"a": 1}


def test_event_split_across_chunk_boundaries():
    """One frame arriving as five fragments must still parse exactly once."""
    frame = b'event: response.created\ndata: {"type": "response.created"}\n\n'
    chunks = [frame[i : i + 7] for i in range(0, len(frame), 7)]
    assert len(chunks) > 3, "test is only meaningful if the frame is actually split"
    events = parse_all(chunks)
    assert len(events) == 1
    assert events[0].kind() == "response.created"


def test_multiple_events_in_one_chunk():
    blob = b'data: {"n": 1}\n\ndata: {"n": 2}\n\ndata: {"n": 3}\n\n'
    events = parse_all([blob])
    assert [e.json()["n"] for e in events] == [1, 2, 3]


def test_split_between_two_events():
    """A chunk boundary landing inside the blank line separating two frames."""
    blob = b'data: {"n": 1}\n\ndata: {"n": 2}\n\n'
    events = parse_all([blob[:16], blob[16:]])
    assert [e.json()["n"] for e in events] == [1, 2]


def test_multiline_data_is_joined_with_newlines():
    events = parse_all([b"data: line one\ndata: line two\n\n"])
    assert events[0].data == "line one\nline two"


def test_crlf_line_endings():
    events = parse_all([b'event: ping\r\ndata: {"a": 1}\r\n\r\n'])
    assert len(events) == 1
    assert events[0].json() == {"a": 1}


def test_bare_cr_line_endings():
    events = parse_all([b'event: ping\rdata: {"a": 1}\r\r'])
    assert len(events) == 1
    assert events[0].json() == {"a": 1}


def test_crlf_split_across_chunks():
    """The ambiguous case: a chunk ending in CR whose LF arrives next.

    Mishandling this turns one frame into two.
    """
    events = parse_all([b'data: {"a": 1}\r', b'\n\r\n'])
    assert len(events) == 1
    assert events[0].json() == {"a": 1}


def test_comment_lines_are_ignored():
    """Keepalive comments must not produce phantom events."""
    events = parse_all([b': keepalive\n\n', b'data: {"a": 1}\n\n'])
    assert len(events) == 1
    assert events[0].json() == {"a": 1}


def test_done_sentinel():
    events = parse_all([b"data: [DONE]\n\n"])
    assert len(events) == 1
    assert events[0].is_done()
    assert events[0].json() is None


def test_malformed_json_does_not_raise():
    """A bad frame degrades one log entry; it must not break the stream."""
    events = parse_all([b"data: {not json\n\n", b'data: {"ok": true}\n\n'])
    assert len(events) == 2
    assert events[0].json() is None
    assert events[0].data == "{not json"  # raw text still captured
    assert events[1].json() == {"ok": True}


def test_empty_stream():
    assert parse_all([]) == []
    assert parse_all([b""]) == []


def test_utf8_split_mid_character():
    """A multi-byte character straddling two chunks must not be corrupted."""
    payload = json.dumps({"text": "héllo → 世界"}, ensure_ascii=False).encode("utf-8")
    frame = b"data: " + payload + b"\n\n"
    midpoint = frame.index(b"\xe4")  # inside the 3-byte sequence for 世
    events = parse_all([frame[: midpoint + 1], frame[midpoint + 1 :]])
    assert len(events) == 1
    assert events[0].json()["text"] == "héllo → 世界"


def test_unterminated_frame_is_recovered_by_flush():
    """A backend that dies mid-frame still yields what it sent."""
    parser = SSEParser()
    assert parser.feed(b'data: {"a": 1}\n') == []  # no blank line yet
    recovered = parser.flush()
    assert len(recovered) == 1
    assert recovered[0].json() == {"a": 1}


def test_field_with_no_colon():
    events = parse_all([b"data\ndata: real\n\n"])
    assert events[0].data == "\nreal"


def test_only_one_leading_space_is_stripped():
    events = parse_all([b"data:  two spaces\n\n"])
    assert events[0].data == " two spaces"


def test_kind_prefers_payload_type_over_event_field():
    """Some OpenAI-compatible servers omit the `event:` line entirely."""
    events = parse_all([b'event: wrong\ndata: {"type": "response.completed"}\n\n'])
    assert events[0].kind() == "response.completed"

    events = parse_all([b'data: {"no_type": 1}\n\n'])
    assert events[0].kind() is None


def test_raw_frame_is_preserved_verbatim():
    """Requirement 3: capture must survive a payload the parser cannot understand."""
    events = parse_all([b"event: weird\ndata: <<<not json>>>\n\n"])
    assert events[0].raw == "event: weird\ndata: <<<not json>>>"


def test_format_sse_roundtrips():
    frame = format_sse("response.completed", {"type": "response.completed", "n": 1})
    events = parse_all([frame])
    assert events[0].kind() == "response.completed"
    assert events[0].json()["n"] == 1
