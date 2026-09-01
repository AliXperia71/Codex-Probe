# CodexProbe

A local recording reverse-proxy for the OpenAI **Codex CLI**. It sits at Codex's
`model_providers.<name>.base_url` extension point and does two things:

1. **Records every LLM call** Codex makes — the complete request (system/developer
   instructions, the full running message and tool history, the tool schemas) and the
   complete response (message text and/or tool calls), with streaming responses
   reassembled — into a per-session JSONL log.
2. **Swaps the backend model.** Pointing Codex at a self-hosted open model is a change
   to the `config` dict, not a change to any code, so the same recorder captures both
   the "before" (OpenAI) and "after" (local) runs.

It does this without patching Codex's Rust source. `base_url` is a supported,
documented extension point, so the proxy survives Codex upgrades and needs no Rust
toolchain.

---

## Important: the task description is out of date, and it matters

The task brief says to understand "the difference between the two `wire_api` values
Codex supports" and to point Codex at a local server speaking Chat Completions.
**That is no longer possible.** As of February 2026 Codex speaks only the Responses
API. From `codex-rs/model-provider-info/src/lib.rs` in the Codex source:

```rust
pub enum WireApi {
    /// The Responses API exposed by OpenAI at `/v1/responses`.
    #[default]
    Responses,
}
```

and deserializing the old value is now a hard error:

```rust
"chat" => Err(serde::de::Error::custom(CHAT_WIRE_API_REMOVED_ERROR)),
```

> `wire_api = "chat"` is no longer supported.
> How to fix: set `wire_api = "responses"` in your provider config.
> More info: https://github.com/openai/codex/discussions/7782

This has a direct consequence for the task's requirement 5. The brief suggests
`vllm serve Qwen/Qwen2.5-Coder-7B-Instruct`, whose OpenAI-compatible surface is
primarily Chat Completions. Codex can no longer talk to that directly.

CodexProbe handles it by **reintroducing `wire_api = "chat"` at the proxy layer**.
The proxy always speaks Responses to Codex; the `backend.wire_api` setting says what
the *backend* speaks; when they differ, the proxy translates. So the capability Codex
removed is restored without forking Codex.

See [Wire formats](#wire-formats-responses-vs-chat-completions) for what the two
formats actually differ on.

---

## Install

```bash
cd codex_probe
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Dependencies: `fastapi`, `uvicorn`, `httpx`, `pydantic`,
`zstandard`.

To use it with the real Codex CLI you also need Codex itself (Node ≥ 22):

```bash
npm install -g @openai/codex     # or: brew install --cask codex
```

---

## Usage

```python
from codex_probe import ProxyRecorder

config = {
    "backend": {
        "base_url": "https://api.openai.com/v1",
        "wire_api": "responses",          # backend speaks Responses -> passthrough
        "api_key_env": "OPENAI_API_KEY",
    },
    "listen": {"host": "127.0.0.1", "port": 8135},
    "log_dir": "./codex_probe_logs",
}

recorder = ProxyRecorder(config)
endpoint = recorder.start()       # "http://127.0.0.1:8135/v1"

# point Codex's model_providers.<name>.base_url at `endpoint`,
# then run Codex normally, e.g.:  codex exec "add a docstring to hello.py"

calls = recorder.stop()           # list[dict], one entry per LLM call, in order
```

`ProxyRecorder` is also a context manager (`with ProxyRecorder(config) as r: ...`).

### Config reference

| Key | Meaning |
|---|---|
| `backend.base_url` | Real backend root, e.g. `https://api.openai.com/v1` or `http://localhost:11434/v1` |
| `backend.wire_api` | `"responses"` → passthrough mode; `"chat"` → translate mode |
| `backend.api_key_env` | Env var holding the backend key (preferred over `api_key`) |
| `backend.model` | Optional override; replaces `model` on every forwarded request |
| `backend.timeout_s` | Total request timeout (default 600 s — local models are slow) |
| `listen.host` / `listen.port` | Where the proxy listens. Port `0` = OS picks a free one |
| `log_dir` | Directory for per-session JSONL logs |
| `session_id` | Names the log file. Defaults to a UTC timestamp |
| `seed` | Injected into requests that lack one, for reproducibility |

The proxy's mode is **derived** from `backend.wire_api` rather than configured
separately, so it cannot drift out of sync with the backend it describes.

---

## Wiring up Codex

Write `~/.codex/config.toml`. `ProxyRecorder.codex_config_toml()` generates this for
the running instance (useful when the port is OS-assigned):

```toml
model = "gpt-5.1"
model_provider = "codexprobe"

[model_providers.codexprobe]
name = "CodexProbe"
base_url = "http://127.0.0.1:8135/v1"
wire_api = "responses"
env_key = "CODEX_PROBE_KEY"
```

Codex appends `/responses` to `base_url`, so it POSTs to
`http://127.0.0.1:8135/v1/responses`.

`env_key` names a variable Codex sends as a bearer token. The proxy holds the *real*
credential (from `backend.api_key_env`) and substitutes it, so `CODEX_PROBE_KEY` can
be any non-empty dummy value:

```bash
export CODEX_PROBE_KEY=dummy
```

Note: `model_provider` and `model_providers` only take effect in the **user-level**
`~/.codex/config.toml`.

Then run Codex as normal:

```bash
codex exec "add a docstring to hello.py"
```

---

## Swapping in a local Qwen model

This is the same recorder with a different `config` — no code changes.

### Serving the model

The task suggests vLLM. **vLLM is CUDA-centric and does not run on Apple Silicon**, so
on a Mac use Ollama, which serves an OpenAI-compatible API and — since v0.13.3 —
implements `/v1/responses` natively:

```bash
ollama pull qwen2.5-coder:7b     # ~4.7 GB at Q4_K_M
ollama serve                     # listens on 127.0.0.1:11434
```

**Hardware.** `qwen2.5-coder:7b` at Q4_K_M needs roughly 6 GB of RAM/VRAM and runs
comfortably on a 16 GB Apple Silicon machine. The fp16 weights (~15 GB) do not.
On an NVIDIA box, `vllm serve Qwen/Qwen2.5-Coder-7B-Instruct` wants ~20 GB VRAM
(A100/H100, or a 24 GB 4090); use `backend.wire_api = "chat"` there, since vLLM's
Chat Completions surface is the better-tested one.

### Config: passthrough (Ollama speaks Responses)

```python
config = {
    "backend": {
        "base_url": "http://localhost:11434/v1",
        "wire_api": "responses",
        "model": "qwen2.5-coder:7b",
    },
    "log_dir": "./codex_probe_logs",
}
```

### Config: translate (backend speaks only Chat Completions)

```python
config = {
    "backend": {
        "base_url": "http://localhost:11434/v1",
        "wire_api": "chat",              # the only line that changed
        "model": "qwen2.5-coder:7b",
    },
    "log_dir": "./codex_probe_logs",
}
```

Use this against llama.cpp, LM Studio, older vLLM, or any gateway without a Responses
endpoint.

### Which local model to use

Measured on this setup (see `FINDINGS.md` for the full write-up):

| Model | Emits structured tool calls? | Completed the demo task? |
|---|---|---|
| `qwen2.5-coder:7b` | **No** — writes them as markdown JSON in the message text | No, in either mode |
| `qwen3.5:9b` | Yes | Yes in translate mode; inconsistent in passthrough |

`qwen2.5-coder:7b` is the model the task brief names, and it is the wrong choice for
an *agentic* harness: it was trained for code completion, not function calling, so
Codex sees a plain text turn, assumes the agent is done, and exits 0 having changed
nothing. Prefer `qwen3.5:9b` (or another tool-calling-tuned build) for anything that
needs to drive the Codex loop.

### Confirming the calls really hit the local model

Every log record carries `backend_url`:

```bash
jq -r '.backend_url' codex_probe_logs/session-*.jsonl | sort -u
# http://localhost:11434/v1/responses      <- local, not OpenAI
```

---

## What a captured call looks like

One JSON object per line, one file per Codex invocation
(`codex_probe_logs/<session_id>.jsonl`). Abridged:

```jsonc
{
  "call_index": 3,
  "session_id": "session-20260831T142201Z",
  "timestamp_start": "2026-08-31T14:22:07.113402+00:00",
  "latency_ms": 4182.7,
  "mode": "passthrough",
  "backend_url": "http://localhost:11434/v1/responses",
  "model": "qwen2.5-coder:7b",
  "status": "completed",
  "http_status": 200,

  "request": {
    "model": "qwen2.5-coder:7b",
    "instructions": "You are a coding agent running in the Codex CLI...",
    "input": [
      {"type": "message", "role": "user",
       "content": [{"type": "input_text", "text": "add a docstring to hello.py"}]},
      {"type": "function_call", "call_id": "call_01", "name": "shell",
       "arguments": "{\"command\":[\"cat\",\"hello.py\"]}"},
      {"type": "function_call_output", "call_id": "call_01",
       "output": "def hello():\n    return 'hi'\n"}
    ],
    "tools": [{"type": "function", "name": "shell", "parameters": {"...": "..."}}],
    "stream": true
  },

  "response": {
    "id": "resp_abc123",
    "status": "completed",
    "output": [
      {"type": "function_call", "call_id": "call_02", "name": "shell",
       "arguments": "{\"command\":[\"apply_patch\",\"...\"]}", "status": "completed"}
    ],
    "usage": {"input_tokens": 4310, "output_tokens": 96, "total_tokens": 4406}
  },

  "output_text": "",
  "function_calls": [{"type": "function_call", "call_id": "call_02", "name": "shell", "...": "..."}],
  "raw_events": ["event: response.created\ndata: {...}", "..."]
}
```

Fields worth knowing:

- `request` — what Codex sent, decompressed. Note it carries the **entire** running
  conversation: Codex is stateless between turns, so call *n* replays every prior
  message and tool result. This is why late-session log lines are large, and why the
  logs make token growth over a session directly visible.
- `backend_request` — what was actually sent upstream. Identical to `request` in
  passthrough; the translated Chat body in translate mode.
- `response` — the reassembled response in Responses shape.
- `backend_response` — the backend's own format; only set in translate mode.
- `raw_events` — every SSE frame verbatim, the fallback when reassembly meets an
  unusual backend.
- `backend_url` — proves which server served the call.

Nothing is truncated at any size (requirement 3).

---

## Wire formats: Responses vs Chat Completions

The distinction the translator exists to bridge.

**Chat Completions** models a conversation as a flat `messages` array and streams a
flat sequence of token deltas. Tool calls arrive as string fragments correlated only
by an `index` field, so the client must accumulate per-index buckets and join them at
the end:

```
delta.tool_calls[0].function.arguments = '{"pa'
delta.tool_calls[0].function.arguments = 'th": "a.py'
delta.tool_calls[0].function.arguments = '"}'
```

Concatenating in arrival order instead of per index produces invalid JSON as soon as
two tool calls run in parallel.

**Responses** models output as a list of addressed, individually-completed *items*
(`message`, `function_call`, `reasoning`), each announced and completed by its own
event. The client does no reassembly bookkeeping, and reasoning items have a
first-class representation that Chat has nowhere to put.

| | Chat Completions | Responses |
|---|---|---|
| Endpoint | `/v1/chat/completions` | `/v1/responses` |
| History | `messages[]` | `input[]` items |
| System prompt | `{"role": "system"}` message | dedicated `instructions` field |
| Tool schema | nested under `function` | flat on the tool object |
| Tool results | `{"role": "tool", "tool_call_id"}` | `function_call_output` item |
| Streaming | flat token deltas | typed per-item lifecycle events |
| Reasoning | no representation | first-class `reasoning` items |
| Terminator | `data: [DONE]` | `response.completed` |
| Token counts | `prompt_tokens` / `completion_tokens` | `input_tokens` / `output_tokens` |

### The detail that makes or breaks a translator

From Codex's own parser (`codex-rs/codex-api/src/sse/responses.rs`), Codex builds
conversation state from `response.output_item.done` and `response.completed`. It
**ignores `response.function_call_arguments.delta` entirely**, and treats
`response.output_text.delta` as cosmetic UI text.

So a translator that emits only deltas renders correctly on screen and then loses the
turn — the agent forgets what it just said and never sees its own tool call. Emitting
completed items is the contract, not a nicety.

---

## How streaming works without changing Codex's behavior

`server.py` **tees** the stream. Each chunk is yielded to Codex the instant it arrives
and, in the same step, copied to a collector:

```python
async for chunk in upstream.aiter_bytes():
    collector.feed(chunk)   # copy for the log
    yield chunk             # straight through to Codex
```

Reassembly runs after the stream closes, in a `finally` block, off the critical path.
Three properties follow:

1. **No added latency.** The CLI keeps rendering live, and Codex's
   `stream_idle_timeout_ms` (300 s default) never fires because of the proxy.
2. **A parser bug cannot corrupt the relay.** Logging parses a *copy*; passthrough
   forwards bytes. This makes requirement 1 structurally true rather than merely
   tested.
3. **Partial failures still log.** Because finalisation is in a `finally`, a client
   disconnect or a backend crash mid-stream still writes a record, flagged
   `incomplete`, with whatever arrived.

Two further passthrough details:

- When no `model` override and no `seed` are configured, the request body is forwarded
  as the **original bytes**, not re-serialised JSON. Key order, whitespace and unicode
  escaping reach the backend exactly as Codex wrote them.
- Hop-by-hop headers (`Connection`, `Transfer-Encoding`, …) are stripped per RFC 7230;
  Codex's session headers are forwarded untouched. Codex can zstd-compress request
  bodies, so the logged copy is decompressed while the forwarded bytes are not.

---

## Tests

```bash
pytest -v
```

The suite runs fully offline against a mock OpenAI-compatible server — no API key, no
network, no Ollama needed. Coverage:

- **Config (13)** — validation, env-var auth resolution, derived mode, error messages.
- **SSE parsing** — events split across chunk boundaries, multi-line `data:`, CRLF vs
  LF vs bare CR, keepalive comments, `[DONE]`, malformed JSON, UTF-8 split mid-character.
- **Responses reassembly** — text, single and parallel tool calls, usage, `failed`,
  truncated streams.
- **Chat reassembly** — argument fragments split across many chunks, interleaved
  parallel calls, `finish_reason`.
- **Translation** — both directions, including the load-bearing
  `output_item.done` / `response.completed` emission.
- **Integration** — byte-identical passthrough, incremental (non-buffered) streaming,
  error propagation, unreachable backend, mid-stream disconnects, concurrent calls,
  zstd request bodies, log schema and ordering.

`examples/quickstart.py` runs the documented walkthrough end to end.

---

## Project layout

```
src/codex_probe/
├── __init__.py      # public API
├── config.py        # Pydantic config; mode derived from backend.wire_api
├── recorder.py      # ProxyRecorder: start()/stop() lifecycle
├── server.py        # FastAPI app; the tee
├── logstore.py      # JSONL session writer + record schema
└── wire/
    ├── sse.py       # incremental SSE parser
    ├── responses.py # Responses stream reassembly
    ├── chat.py      # Chat stream reassembly
    └── translate.py # Responses <-> Chat conversion
```

The split is along one axis: `wire/` knows protocol formats and nothing about HTTP or
sockets; `server.py` knows HTTP and orchestrates; `recorder.py` knows process
lifecycle. That is what lets every format concern be unit-tested without opening a
socket.

**Where to change things.** A new backend is a `config` change. A new *wire format* is
a new module under `wire/` plus a branch in `_prepare_body` and `_handle_stream`. A
new logged field is a field on `CallRecord`.

## Further reading

- **`ARCHITECTURE.md`** — full design rationale, failure modes, and a per-module
  walkthrough. Written to be read before a code review.
- **`FINDINGS.md`** — what actually happened running Codex through the proxy against
  OpenAI and two local Qwen models, including where the open models' tool calling
  breaks and how the wire formats compared.
