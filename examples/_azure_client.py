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


_WORKING_MODEL: dict[tuple[str, str], str] = {}


def _list_models(base: str, key: str, timeout: float = 30.0) -> list[str]:
    """Best-effort discovery of available deployments on the v1 surface."""
    url = f"{base.rstrip('/')}/models"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "api-key": key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for m in items:
        if isinstance(m, dict) and m.get("id"):
            out.append(str(m["id"]))
    return out


def _candidate_models(requested: str) -> list[str]:
    """Generate model-name variations to try when the requested one 404s."""
    seen: set[str] = set()
    out: list[str] = []

    def add(name: str | None) -> None:
        if name and name not in seen:
            seen.add(name)
            out.append(name)

    add(requested)
    if "/" in requested:
        add(requested.rsplit("/", 1)[-1])
    return out


_VARIANT_SUFFIXES = (
    "mini", "nano", "pro", "chat", "codex", "audio", "realtime",
    "transcribe", "tts", "vision", "preview", "instruct",
)


def _best_match(requested: str, available: list[str]) -> str | None:
    """Pick the best available deployment for the requested model name.

    Strategy, in order:
      1. Exact match (case-insensitive).
      2. Prefix match where the suffix is a date version (e.g. ``gpt-5.4``
         → ``gpt-5.4-2026-03-05``), preferring the newest.
      3. Any prefix match.
      4. Any substring match.
    Variant suffixes like ``-mini``, ``-nano``, ``-pro`` are de-prioritized
    so a request for the base name doesn't pick a variant by accident.
    """
    if not available:
        return None

    norm = requested.lower()
    if "/" in norm:
        norm = norm.rsplit("/", 1)[-1]

    # 1. Exact match.
    for av in available:
        if av.lower() == norm:
            return av

    def strict_prefix(av_lower: str) -> bool:
        # The char after the prefix must be '-' (Azure's separator for
        # date / variant) or end-of-string. Don't let "gpt-5" match
        # "gpt-5.5" — those are different major versions.
        if not av_lower.startswith(norm):
            return False
        rest = av_lower[len(norm):]
        return not rest or rest.startswith("-")

    prefix_matches = [av for av in available if strict_prefix(av.lower())]
    if not prefix_matches:
        # 4. Substring fallback.
        for av in available:
            if norm in av.lower():
                return av
        return None

    def is_pure_date_version(name: str) -> bool:
        """True if `name` extends `norm` with purely a date suffix
        (digits and dashes only), not any variant qualifier like
        ``-mini`` or ``-lite`` or ``-pro``."""
        extra = name[len(norm):].lstrip("-")
        if not extra:
            return True
        parts = extra.split("-")
        return all(p.isdigit() for p in parts) and len(parts[0]) == 4

    base_matches = [m for m in prefix_matches if is_pure_date_version(m)]
    if base_matches:
        # Date-stamped names sort chronologically as strings; newest first.
        return sorted(base_matches, reverse=True)[0]

    # Fall back to any prefix match.
    return sorted(prefix_matches, reverse=True)[0]


def _post_chat(
    base: str,
    key: str,
    body: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    url = f"{base.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "api-key": key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


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
    base = os.environ["AZURE_API_BASE"].rstrip("/")
    api_key = os.environ["AZURE_API_KEY"]
    requested = model or os.environ.get("AZURE_MODEL", "gpt-4o-mini")

    def build_body(name: str) -> dict[str, Any]:
        b: dict[str, Any] = {
            "model": name,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            b["tools"] = tools
            if tool_choice is not None:
                b["tool_choice"] = tool_choice
        if extra_body:
            b.update(extra_body)
        return b

    cache_key = (base, requested)
    if cache_key in _WORKING_MODEL:
        candidates = [_WORKING_MODEL[cache_key]]
    else:
        candidates = _candidate_models(requested)

    failed_404: list[str] = []
    last_http_err: Exception | None = None

    for candidate in candidates:
        body = build_body(candidate)
        for attempt in range(retries):
            try:
                response = _post_chat(base, api_key, body, timeout)
                if cache_key not in _WORKING_MODEL:
                    _WORKING_MODEL[cache_key] = candidate
                    if candidate != requested:
                        print(
                            f"[examples/_azure_client] using deployment "
                            f"'{candidate}' (the value of AZURE_MODEL was "
                            f"'{requested}', which Azure could not find)."
                        )
                return response
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 404 and "DeploymentNotFound" in err_body:
                    failed_404.append(candidate)
                    last_http_err = RuntimeError(
                        f"HTTP 404 from {base}/chat/completions for model "
                        f"'{candidate}':\n{err_body}"
                    )
                    break  # try the next candidate
                if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"HTTP {exc.code} from {base}/chat/completions:\n{err_body}"
                )
            except urllib.error.URLError as exc:
                if attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"network error talking to {base}/chat/completions: {exc}"
                )

    # Every literal candidate 404'd. Discover what's actually deployed on this
    # endpoint and try the closest match by name.
    available = _list_models(base, api_key)
    discovered = _best_match(requested, available)
    if discovered and discovered not in failed_404:
        try:
            response = _post_chat(base, api_key, build_body(discovered), timeout)
            _WORKING_MODEL[cache_key] = discovered
            print(
                f"[examples/_azure_client] AZURE_MODEL was '{requested}' but "
                f"Azure has no deployment by that exact name. Using the "
                f"closest match: '{discovered}'. To silence this notice, set "
                f"AZURE_MODEL='{discovered}' in examples/.env."
            )
            return response
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            failed_404.append(discovered)
            last_http_err = RuntimeError(
                f"HTTP {exc.code} from {base}/chat/completions for discovered "
                f"deployment '{discovered}':\n{err_body}"
            )

    lines = [
        "Could not find a working Azure deployment for AZURE_MODEL.",
        f"AZURE_MODEL in your .env: {requested!r}",
        f"Names tried (all failed): {failed_404}",
    ]
    if available:
        lines.append("")
        lines.append(f"Available deployments on this resource ({len(available)} total, first 20):")
        for name in available[:20]:
            lines.append(f"  - {name}")
        lines.append("")
        lines.append("Set AZURE_MODEL in examples/.env to one of those exact names.")
    else:
        lines.append("")
        lines.append(
            "The /models endpoint returned nothing. Either no deployments exist "
            "on this resource, or the API key does not have permission to list "
            "them. Create a deployment in Azure (or use a different resource) "
            "and set AZURE_MODEL to its name."
        )
    raise RuntimeError("\n".join(lines)) from last_http_err


def first_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"empty choices in response: {response}")
    return choices[0].get("message") or {}


__all__ = ["load_env", "chat", "first_message"]
