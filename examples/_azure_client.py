from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_env(env_path: str | Path | None = None) -> None:
    candidates: list[Path] = []
    if env_path is not None:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve().parent
    candidates.extend([here / ".env", here.parent / ".env"])
    for p in candidates:
        if p.exists():
            for raw in p.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
            return


def _endpoint(path: str) -> str:
    base = os.environ["AZURE_API_BASE"].rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    timeout: float = 120.0,
    extra_body: dict[str, Any] | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    api_key = os.environ["AZURE_API_KEY"]
    model = model or os.environ.get("AZURE_MODEL", "gpt-4o")

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    if extra_body:
        body.update(extra_body)

    url = _endpoint("chat/completions")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "api-key": api_key,
    }
    payload = json.dumps(body).encode("utf-8")

    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"HTTP {exc.code} from {url}:\n{err_body}")
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last_err
        except urllib.error.URLError as exc:
            last_err = RuntimeError(f"network error talking to {url}: {exc}")
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last_err
    if last_err:
        raise last_err
    raise RuntimeError("unreachable")


def first_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"empty choices in response: {response}")
    return choices[0].get("message") or {}


__all__ = ["load_env", "chat", "first_message"]
