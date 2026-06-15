"""Single entry-point chat() that routes to Azure or Claude CLI based on
model name. Lets a single benchmark provider call `chat(messages, model)`
and silently dispatch to the right backend.

Model name conventions:
    - 'gpt-5.4', 'gpt-4.1' → routes to _azure_client.chat
    - 'claude-opus-4-7', 'claude-sonnet-4-6', 'claude-haiku-4-5'
      OR aliases 'opus', 'sonnet', 'haiku' → routes to _claude_client.chat
"""

from __future__ import annotations

from _azure_client import chat as _azure_chat, load_env as _azure_load_env
from _claude_client import chat as _claude_chat, is_claude_model


def load_env() -> None:
    _azure_load_env()  # claude side is a no-op


def chat(messages: list[dict], model: str, temperature: float = 0.0,
         timeout_seconds: float = 60.0) -> dict:
    if is_claude_model(model):
        return _claude_chat(messages, model=model, temperature=temperature,
                            timeout_seconds=timeout_seconds)
    return _azure_chat(messages=messages, model=model, temperature=temperature)


ALL_MODELS = [
    "gpt-5.4",
    "gpt-4.1",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]
