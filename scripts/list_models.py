"""Find out which model IDs your provider keys can actually reach.

Model IDs are retired on a rolling basis, and a retired model answers HTTP 404
— which the bot correctly treats as a configuration error. When that happens
*every* ticket is escalated with "the AI service is unavailable", which looks
like a broken bot but is really a stale model name in ``.env``:

* Groq shut down ``llama-3.3-70b-versatile`` and ``llama-3.1-8b-instant`` on
  2026-08-16 (replacements: ``openai/gpt-oss-120b``, ``openai/gpt-oss-20b``).
* Cerebras shut down ``llama-3.3-70b`` on 2026-02-16
  (replacement: ``gpt-oss-120b``).
* Google shut down ``gemini-2.0-flash`` on 2026-06-01
  (replacement: the current ``gemini-3.x-flash`` generation).

Run this whenever the bot starts escalating everything, or right after you
rotate keys, and it asks each configured provider which models the key can
use, picks the best chat model on offer, and prints ready-to-paste lines::

    python -m scripts.list_models

Keys are read from ``.env`` (or the environment) and are **never printed** —
only the key length is shown, matching the rest of the tooling.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
HTTP_TIMEOUT = 20.0

#: Substrings that mark a listing as "not a chat model" for ticket answering.
NON_CHAT_MARKERS: tuple[str, ...] = (
    "embed",
    "guard",
    "whisper",
    "tts",
    "audio",
    "speech",
    "realtime",
    "transcribe",
    "image",
    "imagen",
    "veo",
    "lyria",
    "robotics",
    "rerank",
    "moderation",
    "aqa",
    "learnlm",
    "live",
)


@dataclass(frozen=True)
class Provider:
    """One AI provider and how to ask it for its model list."""

    name: str
    key_var: str
    model_var: str
    list_url: str
    auth_header: str
    preferences: tuple[str, ...]
    signup_url: str

    def headers(self, api_key: str) -> dict[str, str]:
        if self.auth_header == "x-goog-api-key":
            return {"x-goog-api-key": api_key, "Accept": "application/json"}
        return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="groq",
        key_var="GROQ_API_KEY",
        model_var="GROQ_MODEL",
        list_url="https://api.groq.com/openai/v1/models",
        auth_header="authorization",
        preferences=(
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
        ),
        signup_url="https://console.groq.com/keys",
    ),
    Provider(
        name="cerebras",
        key_var="CEREBRAS_API_KEY",
        model_var="CEREBRAS_MODEL",
        list_url="https://api.cerebras.ai/v1/models",
        auth_header="authorization",
        preferences=(
            "gpt-oss-120b",
            "zai-glm-4.7",
            "qwen-3-32b",
            "llama3.1-8b",
            "llama-3.3-70b",
        ),
        signup_url="https://cloud.cerebras.ai",
    ),
    Provider(
        name="gemini",
        key_var="GEMINI_API_KEY",
        model_var="GEMINI_MODEL",
        list_url="https://generativelanguage.googleapis.com/v1beta/models",
        auth_header="x-goog-api-key",
        preferences=(
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-flash-latest",
        ),
        signup_url="https://aistudio.google.com/apikey",
    ),
)


class ModelListError(RuntimeError):
    """The provider refused to list models (bad key, network, endpoint)."""


# --------------------------------------------------------------------------- #
# .env loading
# --------------------------------------------------------------------------- #
def parse_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines, tolerating quotes, comments and ``export``."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip().strip('"').strip("'")
    return values


def load_env(env_file: Path | None = None) -> tuple[Path, dict[str, str]]:
    """Merge ``.env`` values with the process environment (environment wins)."""
    path = env_file if env_file is not None else REPO_ROOT / ".env"
    values = parse_env_file(path)
    for provider in PROVIDERS:
        for name in (provider.key_var, provider.model_var):
            from_process = os.environ.get(name, "").strip()
            if from_process:
                values[name] = from_process
    return path, values


# --------------------------------------------------------------------------- #
# model discovery
# --------------------------------------------------------------------------- #
def is_chat_model(model: str) -> bool:
    """True when a model ID looks like a text-chat model we can use."""
    lowered = model.lower()
    return not any(marker in lowered for marker in NON_CHAT_MARKERS)


def pick_model(provider: Provider, available: Iterable[str]) -> str | None:
    """Pick the best chat model on offer: known-good first, then any of them."""
    usable = [model for model in available if is_chat_model(model)]
    for preferred in provider.preferences:
        if preferred in usable:
            return preferred
    for model in sorted(usable):
        return model
    return None


def _http_get_json(url: str, headers: dict[str, str], *, timeout: float = HTTP_TIMEOUT) -> Any:
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URLs
        return json.loads(response.read().decode("utf-8"))


def _http_error_hint(provider: Provider, exc: urllib.error.HTTPError) -> str:
    status = exc.code
    if status in (401, 403):
        return (
            f"the key was rejected (HTTP {status}) — regenerate {provider.key_var} "
            f"at {provider.signup_url}"
        )
    if status == 404:
        return f"the model-list endpoint returned HTTP 404 — check {provider.list_url}"
    if status == 429:
        return "rate limited (HTTP 429) — wait a minute and run this again"
    return f"HTTP {status} from {provider.list_url}"


def _parse_model_ids(provider: Provider, payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    if provider.auth_header == "x-goog-api-key":
        models: list[str] = []
        for entry in payload.get("models") or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "")
            if not name:
                continue
            methods = entry.get("supportedGenerationMethods")
            if isinstance(methods, list) and methods and "generateContent" not in methods:
                continue
            models.append(name.removeprefix("models/"))
        return models
    return [
        str(entry.get("id"))
        for entry in payload.get("data") or []
        if isinstance(entry, dict) and entry.get("id")
    ]


def fetch_models(
    provider: Provider,
    api_key: str,
    *,
    get_json: Callable[..., Any] | None = None,
) -> list[str]:
    """Return the model IDs this key can use, or raise :class:`ModelListError`."""
    http_get = get_json or _http_get_json
    try:
        payload = http_get(provider.list_url, provider.headers(api_key))
    except urllib.error.HTTPError as exc:
        raise ModelListError(_http_error_hint(provider, exc)) from exc
    except urllib.error.URLError as exc:
        raise ModelListError(f"network error talking to {provider.list_url} ({exc.reason})") from exc
    except json.JSONDecodeError as exc:
        raise ModelListError(f"unreadable response from {provider.list_url} ({exc})") from exc
    return _parse_model_ids(provider, payload)


@dataclass
class Report:
    """What we learned about one provider."""

    provider: Provider
    key_present: bool
    key_length: int = 0
    models: list[str] = field(default_factory=list)
    chosen: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.key_present and self.error is None and self.chosen is not None


def discover(
    provider: Provider,
    api_key: str,
    *,
    fetch: Callable[[Provider, str], list[str]] | None = None,
) -> Report:
    """Ask one provider for its models and pick the best chat model."""
    api_key = (api_key or "").strip()
    if not api_key:
        return Report(provider=provider, key_present=False)
    fetch_fn = fetch or fetch_models
    try:
        models = fetch_fn(provider, api_key)
    except ModelListError as exc:
        return Report(
            provider=provider, key_present=True, key_length=len(api_key), error=str(exc)
        )
    return Report(
        provider=provider,
        key_present=True,
        key_length=len(api_key),
        models=models,
        chosen=pick_model(provider, models),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_report(report: Report, *, limit: int = 25) -> None:
    provider = report.provider
    if not report.key_present:
        print(f"  - {provider.name}: no {provider.key_var} set — skipped")
        return
    if report.error:
        print(f"  x {provider.name}: {report.error}")
        return
    print(f"  + {provider.name}: key loaded ({report.key_length} chars), {len(report.models)} models")
    listing = sorted(report.models)
    for model in listing[:limit]:
        print(f"      {model}")
    if len(listing) > limit:
        print(f"      ... and {len(listing) - limit} more")
    if report.chosen:
        print(f"      -> use {report.chosen}")
    else:
        print("      -> no chat model recognised; pick one from the list above")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.list_models",
        description="Ask each AI provider which models your key can use and print .env lines.",
    )
    parser.add_argument("--env-file", type=Path, default=None, help="path to .env (default: repo root)")
    args = parser.parse_args(argv)

    env_file, values = load_env(args.env_file)
    print(f"Reading keys from {env_file}")
    print()

    reports = [discover(provider, values.get(provider.key_var, "")) for provider in PROVIDERS]
    for report in reports:
        _print_report(report)
    print()

    if not any(report.key_present for report in reports):
        print("No API keys found. Put at least one of these in .env:")
        for provider in PROVIDERS:
            print(f"  {provider.key_var}=   (free key: {provider.signup_url})")
        return 2

    lines = [f"{r.provider.model_var}={r.chosen}" for r in reports if r.ok]
    if not lines:
        print("No provider could be reached — fix the errors above and run this again.")
        return 1

    print("Paste these into .env (replace the existing lines), then restart the bot:")
    print()
    for line in lines:
        print(line)
    print()
    print("Leave a provider out if you don't use it — the bot fails over in AI_PROVIDER_CHAIN order.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
