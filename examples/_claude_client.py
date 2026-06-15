"""Claude inference shim that mimics the same `chat(messages=..., model=..., temperature=...)`
contract as `_azure_client.chat`, so benchmark providers can swap between
Azure-served GPT and Claude with one line.

Uses the `claude` CLI in non-interactive (`-p`) mode with `--output-format
json`. Subscription auth is used (not ANTHROPIC_API_KEY); --bare is NOT
passed because it disables OAuth and we don't have an API key.

The CLI returns a JSON blob like:
    {"result": "<assistant text>", "total_cost_usd": ..., "duration_ms": ..., ...}
We reshape it to the OpenAI-compatible:
    {"choices": [{"message": {"content": "<text>"}}]}

Limitations:
  * `temperature` is not pluggable via this CLI; calls are at the default.
    The bench accepts this — Azure's T=0 isn't truly deterministic either.
  * Per-call overhead is ~3-5 s because the CLI spins up Claude Code's
    agent loop. We minimize this with `--max-turns 1` and explicit
    `--system-prompt`.
  * Models: aliases are 'sonnet', 'opus', 'haiku' OR the fully-qualified
    names in CLAUDE.md (claude-opus-4-7, claude-sonnet-4-6,
    claude-haiku-4-5-20251001).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time

# Map the model strings the bench passes (matched to /chat/completions
# convention) onto the names the `claude` CLI accepts.
_CLAUDE_MODEL_ALIASES = {
    "claude-opus-4-7": "opus",
    "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5": "haiku",
    "claude-haiku-4-5-20251001": "haiku",
    "opus": "opus",
    "sonnet": "sonnet",
    "haiku": "haiku",
}


def is_claude_model(model: str) -> bool:
    return model.lower().startswith("claude") or model.lower() in {"opus", "sonnet", "haiku"}


def _flatten_messages(messages: list[dict]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt). The Claude CLI takes a
    single --system-prompt and a single user prompt; OpenAI-style multi-
    turn messages get squashed into one user prompt with role markers."""
    sys_parts: list[str] = []
    body_parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            sys_parts.append(content)
        elif role == "assistant":
            body_parts.append(f"[assistant previously said]\n{content}")
        else:
            body_parts.append(content)
    return ("\n\n".join(sys_parts), "\n\n".join(body_parts))


def chat(messages: list[dict], model: str, temperature: float = 0.0,
         timeout_seconds: float = 60.0) -> dict:
    """Match _azure_client.chat's return shape."""
    sys_p, user_p = _flatten_messages(messages)
    cli_model = _CLAUDE_MODEL_ALIASES.get(model, model)
    cmd = [
        "claude", "-p", user_p,
        "--model", cli_model,
        "--output-format", "json",
        "--max-turns", "1",
    ]
    if sys_p:
        cmd.extend(["--system-prompt", sys_p])
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_seconds,
            env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return {
            "choices": [{"message": {"content": ""}}],
            "_claude_error": "timeout",
            "_claude_duration_s": time.time() - t0,
        }
    if proc.returncode != 0:
        return {
            "choices": [{"message": {"content": ""}}],
            "_claude_error": f"exit {proc.returncode}: {proc.stderr[:200]}",
            "_claude_duration_s": time.time() - t0,
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "choices": [{"message": {"content": proc.stdout.strip()}}],
            "_claude_error": "non-json output",
            "_claude_duration_s": time.time() - t0,
        }
    text = payload.get("result", "") or ""
    return {
        "choices": [{"message": {"content": text}}],
        "_claude_cost_usd": payload.get("total_cost_usd"),
        "_claude_duration_ms": payload.get("duration_ms"),
        "_claude_session_id": payload.get("session_id"),
        "_claude_duration_s": time.time() - t0,
    }


def load_env() -> None:
    """No-op — claude CLI uses OAuth subscription, no env loading."""
    pass


if __name__ == "__main__":
    # Quick smoke test
    msgs = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Reply with the word PONG and nothing else."},
    ]
    for model in ("sonnet", "haiku", "opus"):
        t0 = time.time()
        resp = chat(msgs, model=model)
        dt = time.time() - t0
        text = resp["choices"][0]["message"]["content"]
        cost = resp.get("_claude_cost_usd")
        print(f"  [{model:>6}] {dt:.1f}s  ${cost:.4f if cost else 0:>6}  → {text!r}")
