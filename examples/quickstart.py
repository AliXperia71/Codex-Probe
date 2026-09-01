#!/usr/bin/env python3
"""CodexProbe quickstart: record a Codex session, then swap the backend model.

Runs the exact walkthrough from the README:

    1. Start the recorder in front of a backend.
    2. Print the `~/.codex/config.toml` fragment that points Codex at it.
    3. Run a real Codex task through it (or wait for you to run one).
    4. Print what was captured.

Usage:

    # against OpenAI (needs OPENAI_API_KEY)
    python examples/quickstart.py --backend openai

    # against a local Qwen served by Ollama, passthrough (Ollama speaks Responses)
    python examples/quickstart.py --backend ollama

    # against the same Ollama, but forcing the Responses -> Chat translator
    python examples/quickstart.py --backend ollama-chat

    # don't drive Codex; just hold the proxy open so you can run it yourself
    python examples/quickstart.py --backend ollama --manual
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from codex_probe import ProxyRecorder

BACKENDS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "wire_api": "responses",
        "api_key_env": "OPENAI_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "wire_api": "responses",
        "model": "qwen2.5-coder:7b",
    },
    "ollama-chat": {
        "base_url": "http://localhost:11434/v1",
        "wire_api": "chat",
        "model": "qwen2.5-coder:7b",
    },
}

DEFAULT_MODELS = {
    "openai": "gpt-5.1",
    "ollama": "qwen2.5-coder:7b",
    "ollama-chat": "qwen2.5-coder:7b",
}


def build_workspace() -> Path:
    """A throwaway repo with one obvious thing to fix.

    Deliberately tiny: the point is to exercise Codex's tool-calling loop, not to
    test the model's coding ability.
    """
    workspace = Path(tempfile.mkdtemp(prefix="codexprobe-demo-"))
    (workspace / "hello.py").write_text("def hello():\n    return 'hi'\n")
    return workspace


def run_codex(task: str, endpoint: str, model: str, workspace: Path) -> int:
    """Drive `codex exec` through the proxy in a sandboxed workspace."""
    config_dir = workspace / ".codex-home"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.toml").write_text(
        f'model = "{model}"\n'
        f'model_provider = "codexprobe"\n\n'
        f"[model_providers.codexprobe]\n"
        f'name = "CodexProbe"\n'
        f'base_url = "{endpoint}"\n'
        f'wire_api = "responses"\n'
        f'env_key = "CODEX_PROBE_KEY"\n'
    )

    env = {
        **os.environ,
        # Codex requires a non-empty token; the proxy substitutes the real one.
        "CODEX_PROBE_KEY": "codexprobe-dummy",
        # Keep the demo out of the user's real ~/.codex.
        "CODEX_HOME": str(config_dir),
    }

    command = [
        "codex", "exec",
        "--skip-git-repo-check",
        "--sandbox", "workspace-write",
        "-C", str(workspace),
        task,
    ]
    print(f"$ {' '.join(command[:-1])} {task!r}\n")
    result = subprocess.run(env=env, args=command, text=True, timeout=900)
    return result.returncode


def summarize(calls: list[dict], log_path: Path) -> None:
    print("\n" + "=" * 72)
    print(f"Captured {len(calls)} LLM call(s) -> {log_path}")
    print("=" * 72)

    for call in calls:
        request = call.get("request") or {}
        history = request.get("input")
        history_len = len(history) if isinstance(history, list) else 0
        tools = request.get("tools") or []
        names = [c.get("name") for c in call.get("function_calls") or []]
        text = (call.get("output_text") or "").replace("\n", " ")

        print(
            f"\n[{call['call_index']}] {call['status']:<10} "
            f"{call.get('latency_ms', 0):7.0f} ms  -> {call['backend_url']}"
        )
        print(f"     request : {history_len} history item(s), {len(tools)} tool schema(s)")
        usage = ((call.get("response") or {}).get("usage")) or {}
        if usage:
            print(f"     tokens  : in={usage.get('input_tokens')} out={usage.get('output_tokens')}")
        if names:
            print(f"     calls   : {', '.join(n for n in names if n)}")
        if text:
            print(f"     text    : {text[:100]}{'...' if len(text) > 100 else ''}")
        if call.get("error"):
            print(f"     error   : {str(call['error'])[:160]}")

    # The acceptance criterion: prove which server actually served the calls.
    backends = sorted({c["backend_url"] for c in calls})
    print(f"\nBackends actually contacted: {backends}")

    total_in = sum(((c.get("response") or {}).get("usage") or {}).get("input_tokens", 0) for c in calls)
    if total_in:
        print(f"Total input tokens across the session: {total_in:,}")
        print("(Codex is stateless between turns, so each call replays the whole conversation.)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="ollama")
    parser.add_argument("--model", default=None, help="Override the model name.")
    parser.add_argument("--task", default="Add a short docstring to the hello() function in hello.py.")
    parser.add_argument("--log-dir", default="./codex_probe_logs")
    parser.add_argument("--manual", action="store_true",
                        help="Hold the proxy open instead of running Codex.")
    args = parser.parse_args()

    backend = dict(BACKENDS[args.backend])
    if args.model:
        backend["model"] = args.model
    model = backend.get("model") or DEFAULT_MODELS[args.backend]

    if backend.get("api_key_env") and not os.environ.get(backend["api_key_env"]):
        print(f"error: ${backend['api_key_env']} is not set", file=sys.stderr)
        return 2

    config = {
        "backend": backend,
        "listen": {"host": "127.0.0.1", "port": 0},  # 0 -> OS picks a free port
        "log_dir": args.log_dir,
        "session_id": f"{args.backend}-{os.getpid()}",
    }

    recorder = ProxyRecorder(config)
    endpoint = recorder.start()

    print(f"CodexProbe listening on {endpoint}")
    print(f"  mode    : {recorder.mode.value}")
    print(f"  backend : {backend['base_url']} (wire_api={backend['wire_api']})")
    print(f"  model   : {model}")
    print(f"  log     : {recorder.log_path}\n")
    print("Codex config.toml fragment:\n")
    print(recorder.codex_config_toml(model=model))

    try:
        if args.manual:
            print("Proxy is up. Point Codex at it and run a task; press Enter here when done.")
            input()
        else:
            if shutil.which("codex") is None:
                print("error: `codex` not found. Install with: npm install -g @openai/codex",
                      file=sys.stderr)
                return 2
            workspace = build_workspace()
            print(f"Demo workspace: {workspace}\n")
            code = run_codex(args.task, endpoint, model, workspace)
            print(f"\ncodex exited with status {code}")
            result = (workspace / "hello.py").read_text()
            print(f"\nResulting hello.py:\n{'-' * 40}\n{result}{'-' * 40}")
    finally:
        calls = recorder.stop()
        summarize(calls, recorder.log_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
