# Findings

What actually happened when Codex was run through CodexProbe against three
backends. The acceptance criteria explicitly allow "the write-up honestly
documents where the open model's tool-calling behavior diverges/breaks"; this is
that write-up.

All runs used the same task against the same one-file workspace:

```python
# hello.py
def hello():
    return 'hi'
```

> Task: "Add a short docstring to the hello() function in hello.py."

---

## 1. Codex no longer supports Chat Completions

The task brief asks you to understand "the two `wire_api` values Codex supports"
and to point Codex at a local Chat-Completions server. Neither is possible any
more. From `codex-rs/model-provider-info/src/lib.rs` on `main`:

```rust
pub enum WireApi {
    /// The Responses API exposed by OpenAI at `/v1/responses`.
    #[default]
    Responses,
}
```

```rust
"chat" => Err(serde::de::Error::custom(CHAT_WIRE_API_REMOVED_ERROR)),
```

> `wire_api = "chat"` is no longer supported.
> How to fix: set `wire_api = "responses"` in your provider config.
> More info: https://github.com/openai/codex/discussions/7782

The stated rationale in that discussion is that Chat Completions "hampered our
ability to improve Codex": the Responses API carries reasoning items and typed
per-item lifecycle events that the flat Chat delta stream cannot express.

**Impact on this task.** The brief's suggested
`vllm serve Qwen/Qwen2.5-Coder-7B-Instruct` exposes Chat Completions as its
best-supported surface, so Codex cannot drive it directly. The community answer
is a translation proxy — which is exactly what CodexProbe's translate mode is.
The removed `wire_api = "chat"` is reintroduced one layer down.

Secondary finding: **vLLM is not runnable on this machine at all** (Apple M4,
arm64; vLLM is CUDA-centric). Ollama 0.32.3 was used instead. Ollama added native
`/v1/responses` in v0.13.3, non-stateful only — no `previous_response_id`.

---

## 2. Results

| Backend | Model | Mode | Calls | Codex exit | Task completed? |
|---|---|---|---:|---|---|
| OpenAI | `gpt-5.1` | passthrough | 4 | 0 | **Yes** |
| Ollama | `qwen3.5:9b` | translate | 8 | 0 | **Yes** |
| Ollama | `qwen3.5:9b` | translate (2nd run) | 4 | 0 | **Yes** |
| Ollama | `qwen3.5:9b` | passthrough | 3 | 0 | **No** — malformed patch |
| Ollama | `qwen3.5:9b` | passthrough (2nd run) | 2 | 0 | **No** — gave up after one tool call |
| Ollama | `qwen2.5-coder:7b` | passthrough | 1 | 0 | **No** — no tool call emitted |
| Ollama | `qwen2.5-coder:7b` | translate | 1 | 0 | **No** — no tool call emitted |

Two failure modes appeared, and they are qualitatively different. Both were
invisible from Codex's own output — every run above exited 0.

### 2a. `qwen2.5-coder:7b` writes tool calls as prose instead of calling tools

The model the task brief names produced exactly one call, no structured tool
call, and stopped. The recorded `output_text` shows why:

```
Let's add a short docstring to the `hello()` function in `hello.py`.

**Plan Update:**
- **step:** Open the `hello.py` file.

```json
{"name": "update_plan", "arguments": {"plan": [{"step": "Open the `hello.py`
file.", "status": "in_progress"}]}}
```

Next, I'll open the `hello.py` file and locate the `hello()` function...
```

The model *knows* which tool to call and emits well-formed JSON for it — but as
a markdown-fenced code block inside an assistant message, not as a
`function_call` item. The recorded output item types are `['message']` and
nothing else. Codex sees a plain text turn with no tool calls, concludes the
agent is done, and exits 0 having changed nothing.

**This reproduced identically in both passthrough and translate mode.** That is
the useful part: running the same model over two different wire formats isolates
the variable. The failure is in the model's tool-calling ability, not in the
protocol or the proxy. Qwen2.5-Coder was trained primarily for code completion,
not agentic function calling, and 10 tool schemas of Codex's harness is well
outside what it handles.

### 2b. `qwen3.5:9b` calls tools correctly but struggles with `apply_patch`

The newer model emits genuine structured `function_call` items —
`exec_command`, `apply_patch` — and drove the loop to completion in the
translate run, producing correct output:

```python
def hello():
    """Says hi."""
    return 'hi'
```

One passthrough run, however, gave up after three calls. The captured
`apply_patch` arguments show the model deliberating *inside the argument
string*:

```
{"cmd":"*** Begin Patch
*** Update File: hello.py@+1,+4
-def hello():
-    return 'hi'
+def hello():\n\"\"\"Say hi.\"\"\"\n-    pass  # placeholder for clarity
     return 'hi'\n
# Wait - that's wrong syntax. Let me redo this properly with proper format:

*** Begin Patch
*** Update File: hello.py@1,2
...as the old content is already there in my head... no wait I need to use
- and + correctly for existing lines.
```

Three defects in one payload: two `*** Begin Patch` headers in a single call,
malformed hunk headers (`@+1,+4`), and inconsistent escaping. The next call
produced no tool call at all.

**On whether translate genuinely beat passthrough here.** Across completed
runs the tally was **translate 2/2 completed the task, passthrough 0/2**. A
third passthrough run was making normal progress (`exec_command`,
`apply_patch`, `exec_command`, `exec_command`) when a harness timeout cut it
off, so it is inconclusive rather than a failure.

That is suggestive of a real difference, and there is a plausible mechanism:
Ollama's Chat Completions path (which translate mode drives) is years older and
far more exercised than its `/v1/responses` implementation, which landed in
v0.13.3 — matching the reports in discussion #7782 of Responses-API streaming
and tool-calling problems across LM Studio, vLLM and llama.cpp.

But it is **not conclusive**. n=2 per mode, temperature is non-default-free, and
a 9B model's success on a strict diff format is exactly the kind of marginal
capability that swings run to run. The claim this evidence supports is "translate
mode was at least as reliable as passthrough, and plausibly more so" — not that
the native endpoint is broken.

The distinction that *is* well supported is between the two models: 2a is a
reproducible, mode-independent inability to emit tool calls at all; 2b is a
capable tool-caller that sometimes fumbles a strict diff format.

### 2c. The proxy captured a failure it did not cause

The first OpenAI attempt used `gpt-5.2-codex`, which appears in `GET /v1/models`
but is not callable by this account:

```json
{"message": "Model not found gpt-5.2-codex", "type": "invalid_request_error"}
```

The log recorded six calls — Codex's full retry sequence — at 1630, 470, 249,
260, 211 and 285 ms. Useful incidentally: it confirms the proxy relays error
status and body faithfully, and makes Codex's retry-with-backoff policy directly
observable.

## 3. Things the logs make visible that are otherwise invisible

**Context growth is steep.** Codex is stateless between turns, so every call
replays the entire conversation. From `qwen35-translate.jsonl`:

| Call | History items | Input tokens | Output tokens |
|---:|---:|---:|---:|
| 0 | 3 | 7,079 | 66 |
| 1 | 5 | 7,173 | 90 |
| 3 | 9 | 7,378 | 130 |
| 5 | 13 | 7,604 | 131 |
| 7 | 17 | 7,824 | 69 |

59,541 input tokens against 788 output tokens across the session — a **76:1
ratio** to add one line to one file.

**Prompt caching is what makes this viable.** On the OpenAI run, the final call
sent 13,424 input tokens of which **13,184 were served from cache** (98%). Codex
appends to a stable prefix precisely so the cache keeps hitting.

**The tool surface dominates the prompt.** Codex sent 10 tool schemas on every
call (11 against `gpt-5.2-codex`). The first call's 7,079 input tokens carry a
3-item conversation — so the overwhelming majority is instructions plus tool
definitions, before the task is even stated. For a 7B model that is most of the
usable attention budget, and a plausible contributing factor to 2a.

**Latency.** OpenAI `gpt-5.1`: 2.3–7.7 s per call. Local Qwen on an M4:
7–157 s per call — roughly 10–20× slower, partly because the machine was
simultaneously pulling a 4.7 GB model.

---

## 4. What this says about the task's premise

The brief assumes the interesting question is "can a local open model stand in
for the OpenAI backend." The measured answer is: the *transport* substitutes
cleanly — swapping backends is genuinely a config change, the calls provably hit
localhost, and both wire formats round-trip. What does not substitute is the
model's ability to produce Codex's structured edit format under a large tool
surface.

That failure is invisible from Codex's output, which reported success. It is
plainly visible in the recorded call arguments. Which is a reasonable argument
for building the recorder in the first place.

---

## Reproducing

```bash
# OpenAI baseline
python examples/quickstart.py --backend openai

# Local Qwen, passthrough (Ollama's native /v1/responses)
python examples/quickstart.py --backend ollama --model qwen3.5:9b

# Local Qwen, translate (Responses -> Chat Completions)
python examples/quickstart.py --backend ollama-chat --model qwen3.5:9b
```

Logs land in `codex_probe_logs/<session>.jsonl`. To confirm which server served a
run:

```bash
jq -r '.backend_url' codex_probe_logs/*.jsonl | sort -u
```
