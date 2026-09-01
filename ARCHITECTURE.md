# CodexProbe — Architecture & Design Rationale

This document is written to be read before a code review. It covers why the
project is shaped the way it is, what each module is responsible for, where the
failure modes are, and how each part is tested.

---

## 1. The problem

Codex CLI runs a multi-turn agent loop. Each turn is one HTTP call carrying the
system/developer instructions, the entire running conversation and tool history,
and the tool schemas; back comes an assistant message and/or tool calls. From the
outside you see the agent's *actions* but never the requests and responses that
produced them. Codex has a `log_dir` and OpenTelemetry tracing, but neither dumps
the full verbatim request/response body of every LLM call.

Two goals follow: capture every call losslessly, and use the same interception
point to swap the backend model.

## 2. Why a proxy

There were three plausible approaches.

| Approach | Verdict |
|---|---|
| Patch Codex's Rust source | Rejected. Forks a fast-moving codebase, needs a Rust toolchain, and every upgrade re-does the work. |
| `mitmproxy` / system TLS interception | Rejected. Requires certificate installation, is fragile, and captures far more than the LLM calls. |
| **Sit at the `base_url` extension point** | **Chosen.** Documented and supported, no Codex changes, no certificates, and it is the same knob users already use for custom providers. |

Because Codex lets any `model_providers.<name>.base_url` be an arbitrary
OpenAI-compatible endpoint, pointing it at `http://127.0.0.1:8135/v1` puts our
process in the exact position we need — and gives us the backend swap for free,
since the proxy chooses where to forward.

## 3. What changed since the task was written

The brief says to understand "the two `wire_api` values Codex supports" and to
point Codex at a Chat-Completions local server. As of February 2026 that is no
longer possible. From `codex-rs/model-provider-info/src/lib.rs`:

```rust
pub enum WireApi {
    #[default]
    Responses,
}
```

`"chat"` now fails deserialization with a message pointing at
[discussion #7782](https://github.com/openai/codex/discussions/7782). Codex POSTs
only to `{base_url}/responses` (`InferenceEndpoint::path()`).

Two consequences:

1. The task's suggested `vllm serve` route, whose best-supported surface is Chat
   Completions, cannot be driven by Codex directly.
2. The `wire_api` distinction the acceptance criteria asks you to explain is now
   invisible from Codex's side.

CodexProbe answers both by **reintroducing `wire_api = "chat"` at the proxy
layer**. Codex always speaks Responses to us; `backend.wire_api` says what the
backend speaks; when they differ, we translate. The removed capability is
restored without forking Codex, and the wire-format distinction becomes something
the code demonstrates rather than something the README asserts.

## 4. Module map

```
src/codex_probe/
├── config.py     what to talk to        (no I/O)
├── logstore.py   what to record         (file I/O only)
├── wire/         how formats work       (pure functions, no I/O)
│   ├── sse.py
│   ├── responses.py
│   ├── chat.py
│   └── translate.py
├── server.py     HTTP orchestration     (the tee lives here)
└── recorder.py   process lifecycle      (threads, sockets)
```

The split runs along one axis: **`wire/` knows protocol formats and nothing about
HTTP or sockets.** That is what makes every format concern unit-testable without
opening a socket — 70 of the 99 tests never start a server. `server.py` knows
HTTP and orchestrates; `recorder.py` knows process lifecycle.

### `config.py`

Pydantic models. One decision worth defending: `ProxyConfig.mode` is a **derived
property**, not a field.

```python
@property
def mode(self) -> ProxyMode:
    return ProxyMode.PASSTHROUGH if self.backend.wire_api == "responses" else ProxyMode.TRANSLATE
```

If mode were a separate field it could contradict the backend it describes
(`wire_api="chat"` with `mode="passthrough"` would send Responses payloads to a
Chat endpoint). Deriving it makes that state unrepresentable.

`extra="forbid"` is set on every model so a misspelled key is an error, not a
silently-ignored default. `resolve_api_key()` raises on a named-but-unset
environment variable rather than returning `None`, because the alternative
surfaces as an opaque 401 from the backend much later.

### `wire/sse.py`

An incremental SSE parser. The reason it is incremental rather than
line-splitting a whole body: bytes arrive in chunks that have nothing to do with
event boundaries. One frame can span three TCP reads; one read can hold five
frames plus half a sixth. A parser that assumes "one chunk == one event" passes
against a fast local server and drops events against a real network.

Two subtleties:

- **UTF-8 across boundaries.** A multi-byte character can straddle two chunks, so
  decoding goes through `codecs.getincrementaldecoder`, which holds the partial
  sequence instead of emitting a replacement character.
- **A trailing bare CR is ambiguous.** SSE accepts CRLF, LF, or CR. If a chunk
  ends in `\r` we cannot yet know whether it is a CR terminator or half of a
  CRLF, so it is held back. Getting this wrong splits one frame into two.

`SSEEvent.kind()` prefers the payload's `type` key over the SSE `event:` line,
because some OpenAI-compatible servers omit the latter entirely.

### `wire/responses.py`

Folds Responses frames into one response object. Priority order:

1. `response.completed` — the server's own authoritative final state. Trust it.
2. Otherwise, the `response.output_item.done` items collected so far, marked
   `incomplete`.
3. Otherwise, the accumulated text deltas.

The fallback chain matters more than it looks: a partial log is exactly what you
want when debugging why a run died, and that is precisely the case where
`response.completed` never arrives.

### `wire/chat.py`

Folds Chat chunks into one completion object. The awkward part is tool calls:
arguments arrive as string fragments correlated only by an `index` field.

```
delta.tool_calls[0].function.arguments = '{"pa'
delta.tool_calls[0].function.arguments = 'th": "a.py'
delta.tool_calls[0].function.arguments = '"}'
```

With parallel calls, fragments for index 0 and 1 interleave freely, so fragments
go into per-index buckets and are joined only at the end. Concatenating in
arrival order produces spliced garbage that fails to parse. This is the concrete
reason the Responses API exists: it models output as addressed items so the
client never does this bookkeeping.

`id` and `name` usually appear only on the first fragment, and some OSS servers
never send an `id` at all — so a deterministic `call_N` is synthesised, because
Codex needs a stable `call_id` to match a tool result back to its call.

### `wire/translate.py`

The two directions, which are **not symmetric**:

- **Request (Responses → Chat)** is a structural reshape: `instructions` becomes
  a leading system message, `input[]` items become `messages[]`, flat tool
  schemas nest under `function`, Responses-only fields are dropped.
- **Response (Chat → Responses)** must *invent* structure the Chat format does
  not carry. Chat streams flat token deltas; Responses streams addressed,
  individually-completed items. So the translator buffers, decides where item
  boundaries fall, and emits completed items.

**The load-bearing detail.** From Codex's parser
(`codex-rs/codex-api/src/sse/responses.rs`):

| Event | Codex's treatment |
|---|---|
| `response.output_item.done` | **builds conversation state** |
| `response.completed` | **terminates; carries final output + usage** |
| `response.output_text.delta` | cosmetic UI text |
| `response.function_call_arguments.delta` | **ignored entirely** |
| unknown `*.delta` | silently ignored |

A translator emitting only deltas renders correctly on screen and then loses the
turn — the agent forgets what it just said and never sees its own tool call.
Emitting the completed items is the contract, not polish. This is the single most
important thing in the codebase and `test_translate.py` pins it down explicitly.

Also handled: `finish_reason: "length"` maps to `response.incomplete` rather than
`response.completed`, so truncation stays distinguishable from a finished turn;
and token counts are renamed (`prompt_tokens` → `input_tokens`).

### `server.py`

One route that matters: `POST /v1/responses`. The central mechanism is the tee:

```python
async for chunk in upstream.aiter_bytes():
    collector.feed(chunk)   # copy for the log
    yield chunk             # straight through to Codex
```

Three properties follow, and they are the answer to "how does the proxy handle
streaming without changing Codex's behavior":

1. **No added latency.** Chunks reach Codex as they arrive, so the CLI keeps
   rendering and Codex's `stream_idle_timeout_ms` (300 s) never fires because of
   us. The obvious alternative — read the whole response, parse, then forward —
   turns a 60-second generation into a 60-second hang.
2. **A parser bug cannot corrupt the relay.** Logging parses a *copy*.
   Requirement 1 is structurally true, not merely tested.
3. **Partial failures still log.** Finalisation runs in a `finally`, so a client
   disconnect or backend crash still writes a record flagged `incomplete`.

`_finalize` is deliberately **synchronous** despite its async callers: it does no
I/O, and a coroutine awaited inside a `finally` during task cancellation can
itself be cancelled before completing. Keeping it sync guarantees the record is
written on the disconnect path — the path where the log is most valuable.

Other decisions:

- **Byte-level passthrough.** With no `model` override and no `seed`, the
  *original request bytes* are forwarded, not re-serialised JSON. Key order,
  whitespace and unicode escaping reach the backend exactly as Codex wrote them.
  Re-serialising would produce a semantically equal but textually different body.
- **Header handling.** Hop-by-hop headers are stripped per RFC 7230 §6.1. Header
  names are normalised to lower case first — HTTP header names are
  case-insensitive but a Python dict is not, and mixing `Authorization` with
  `authorization` sends the credential twice, letting Codex's dummy token win.
  (This was a real bug, caught by `test_api_key_is_substituted_from_env`.)
- **Response headers.** `content-encoding` and `content-length` are stripped
  because `aiter_bytes()` has already decoded the body; forwarding them would
  describe the bytes incorrectly.
- **Compression.** Codex can zstd-compress request bodies. The *logged copy* is
  decompressed; the forwarded bytes are not touched.

### `recorder.py`

`ProxyRecorder` matches the task's required API exactly. Two decisions:

- **uvicorn runs on a background thread**, so the caller stays synchronous and
  just gets a URL back. Running it in the foreground would mean the caller never
  reaches the line after `start()`.
- **The listening socket is bound before uvicorn starts.** This makes `port: 0`
  usable (the OS assigns a port and `start()` reports the real one) and removes a
  race: the socket is already accepting when `start()` returns, so a caller can
  launch Codex on the very next line without a sleep or retry loop.

`stop()` sets `should_exit` and joins rather than killing the thread, so
in-flight requests finish and the last call's record is not lost.

`codex_config_toml()` generates the config fragment for the *running* instance,
because with an OS-assigned port a hand-copied stale port is the most common way
to get this wrong.

### `logstore.py`

JSONL, one file per session. A single JSON array would need rewriting on every
append and would be unreadable if the process died mid-run; JSONL appends and
survives truncation. Records are flushed per call, not at `stop()`, so a killed
session still leaves a usable log.

`next_index()` reserves the arrival slot when a request *arrives*, not when it
completes, so the log preserves the order Codex made the calls in even if a slow
call finishes out of order. Writes are lock-guarded because uvicorn serves
concurrently.

## 5. Failure modes

| Failure | Behaviour |
|---|---|
| Backend unreachable | 502 with a clear message; record `status="error"`; fails fast on a short connect timeout rather than hanging the session |
| Backend returns 4xx/5xx | Status and body relayed unchanged; record `status="failed"` |
| Backend dies mid-stream | Client sees a broken stream (correct); record `status="incomplete"` with whatever arrived |
| Client (Codex) disconnects | Upstream closed; record still written from the `finally` |
| Malformed SSE frame | `json()` returns `None`; raw text preserved in `raw_events`; surrounding frames unaffected |
| Non-UTF-8 bytes | Incremental decoder replaces rather than raising |
| Unset `api_key_env` | `ValueError` naming the variable, at resolve time |
| Port already in use | `RuntimeError` naming the address, at `start()` |
| Unknown Responses item type in translate | Skipped, not fatal — forward compatibility with new item types |

## 6. Testing strategy

99 tests, fully offline against a scriptable mock backend — no API key, no
network, no Ollama. That matters for a deliverable someone else has to run.

Layered deliberately:

- **`test_config.py` (13)** — validation, derived mode, auth resolution, typo rejection.
- **`test_sse.py` (19)** — the parser's real hazards: frames split across chunk
  boundaries, CRLF split across chunks, bare CR, UTF-8 split mid-character,
  keepalive comments, malformed JSON, unterminated frames.
- **`test_reassembly.py` (16)** — both formats, including parallel tool calls with
  interleaved indices and truncated-stream fallbacks.
- **`test_translate.py` (22)** — both directions, with explicit assertions that
  `output_item.done` and `response.completed` are emitted.
- **`test_proxy.py` (29)** — the integration layer.

Three integration tests deserve mention:

- `test_request_is_forwarded_byte_identically` asserts on **raw bytes**, not
  parsed JSON, because a dict comparison silently accepts key reordering and
  re-encoding.
- `test_streaming_is_incremental_not_buffered` is the only test that
  distinguishes the tee from a buffer-then-forward proxy — every other test
  passes either way. It delays the mock between chunks and asserts the first
  chunk reaches the client well before the last is sent.
- `test_switching_backend_is_config_only` runs two backends through one unchanged
  code path, demonstrating requirement 4 rather than asserting it.

## 7. Where to change things

| Change | Where |
|---|---|
| A different backend | `config` only — no code |
| A new wire format | New module in `wire/`, plus a branch in `_prepare_body` and `_handle_stream` |
| A new logged field | A field on `CallRecord` |
| Redaction before logging | One hook in `_finalize` — the single point where records are written |
| Replay/fixture generation | Read the JSONL back with `read_session()`; records hold complete request bodies |

## 8. Known limitations

- **Responses-API statefulness is not supported.** `previous_response_id` and
  `store=true` would require the proxy to hold server-side conversation state.
  Codex sends the full history each turn and Ollama's Responses implementation is
  itself non-stateful, so this has not been needed. Translate mode drops both
  fields.
- **Built-in server-side tools** (`web_search`, `file_search`) have no Chat
  equivalent and are dropped in translate mode.
- **Reasoning items are dropped** when translating to Chat. There is nowhere to
  put them, and the Chat backend neither produced nor can consume them.
- **No WebSocket transport.** Codex has an experimental Responses-over-WebSocket
  path (`supports_websockets`); this proxy handles HTTP/SSE only.
- **Non-streaming requests** are handled but untested against real Codex, which
  always streams.
