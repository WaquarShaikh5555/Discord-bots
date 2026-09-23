"""Tests for ``scripts.list_models`` — model discovery without leaking keys.

The point of the tool is to repair a *stale model name* in ``.env``: retired
IDs answer HTTP 404 and the bot then escalates every ticket with "the AI
service is unavailable". Every network call here is faked, so the suite also
documents the exact provider payload shapes we parse.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from scripts import list_models as lm

FAKE_KEY = "test-key-not-real-0000"

ENV_VARS = tuple(name for provider in lm.PROVIDERS for name in (provider.key_var, provider.model_var))


@pytest.fixture
def clean_env(monkeypatch):
    """No stray provider variables leaking in from the ambient environment."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _provider(name: str) -> lm.Provider:
    return next(provider for provider in lm.PROVIDERS if provider.name == name)


def _http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid", status, "boom", {}, None)


# --------------------------------------------------------------------------- #
# .env parsing
# --------------------------------------------------------------------------- #
def test_parse_env_file_handles_comments_quotes_and_export(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "GROQ_API_KEY=plain-value",
                'CEREBRAS_API_KEY="quoted value"',
                "export GEMINI_API_KEY='exported'",
                "NOT_ASSIGNMENT",
            ]
        ),
        encoding="utf-8",
    )
    values = lm.parse_env_file(env)
    assert values["GROQ_API_KEY"] == "plain-value"
    assert values["CEREBRAS_API_KEY"] == "quoted value"
    assert values["GEMINI_API_KEY"] == "exported"
    assert "NOT_ASSIGNMENT" not in values


def test_parse_env_file_missing_file_is_empty(tmp_path):
    assert lm.parse_env_file(tmp_path / "nope.env") == {}


def test_load_env_prefers_process_environment(tmp_path, clean_env):
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=from-file\n", encoding="utf-8")
    clean_env.setenv("GROQ_API_KEY", "from-process")
    _, values = lm.load_env(env)
    assert values["GROQ_API_KEY"] == "from-process"


# --------------------------------------------------------------------------- #
# model selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-oss-120b", True),
        ("gemini-3.8-flash", True),
        ("llama3.1-8b", True),
        ("whisper-large-v3", False),
        ("text-embedding-3-large", False),
        ("meta-llama/llama-guard-4-12b", False),
        ("gemini-3.8-flash-tts", False),
        ("gemini-3.8-live", False),
        ("playai-tts", False),
        ("gemini-3.1-flash-image", False),
    ],
)
def test_is_chat_model_filters_non_chat_models(model, expected):
    assert lm.is_chat_model(model) is expected


def test_pick_model_prefers_current_groq_model():
    available = ["llama-3.3-70b-versatile", "whisper-large-v3", "openai/gpt-oss-120b"]
    assert lm.pick_model(_provider("groq"), available) == "openai/gpt-oss-120b"


def test_pick_model_prefers_cerebras_gpt_oss():
    available = ["llama-3.3-70b", "llama3.1-8b", "gpt-oss-120b"]
    assert lm.pick_model(_provider("cerebras"), available) == "gpt-oss-120b"


def test_pick_model_prefers_newest_gemini_flash():
    available = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.7-flash"]
    assert lm.pick_model(_provider("gemini"), available) == "gemini-3.8-flash"


def test_pick_model_falls_back_to_any_usable_model():
    assert lm.pick_model(_provider("groq"), ["zzz-new-model"]) == "zzz-new-model"


def test_pick_model_returns_none_when_nothing_is_usable():
    assert lm.pick_model(_provider("groq"), ["whisper-large-v3", "playai-tts"]) is None
    assert lm.pick_model(_provider("groq"), []) is None


# --------------------------------------------------------------------------- #
# payload parsing
# --------------------------------------------------------------------------- #
def test_fetch_models_reads_openai_style_payload():
    payload = {"data": [{"id": "openai/gpt-oss-120b"}, {"id": "whisper-large-v3"}, {"nope": 1}]}
    models = lm.fetch_models(_provider("groq"), FAKE_KEY, get_json=lambda *a, **k: payload)
    assert models == ["openai/gpt-oss-120b", "whisper-large-v3"]


def test_fetch_models_strips_gemini_prefix_and_filters_methods():
    payload = {
        "models": [
            {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-3.8-flash-tts"},
        ]
    }
    models = lm.fetch_models(_provider("gemini"), FAKE_KEY, get_json=lambda *a, **k: payload)
    # A model without a method list is kept (be liberal); the media one is dropped.
    assert models == ["gemini-3.8-flash", "gemini-3.8-flash-tts"]


def test_fetch_models_sends_the_right_auth_header():
    seen: dict[str, dict[str, str]] = {}

    def fake_get_json(url, headers, **kwargs):
        seen["headers"] = headers
        return {"data": []}

    lm.fetch_models(_provider("groq"), FAKE_KEY, get_json=fake_get_json)
    assert seen["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    lm.fetch_models(_provider("gemini"), FAKE_KEY, get_json=fake_get_json)
    assert seen["headers"]["x-goog-api-key"] == FAKE_KEY


def test_fetch_models_maps_401_to_a_key_hint():
    def boom(*args, **kwargs):
        raise _http_error(401)

    with pytest.raises(lm.ModelListError) as excinfo:
        lm.fetch_models(_provider("groq"), FAKE_KEY, get_json=boom)
    assert "rejected" in str(excinfo.value)
    assert "GROQ_API_KEY" in str(excinfo.value)


def test_fetch_models_maps_network_failure():
    def boom(*args, **kwargs):
        raise urllib.error.URLError("no route to host")

    with pytest.raises(lm.ModelListError) as excinfo:
        lm.fetch_models(_provider("cerebras"), FAKE_KEY, get_json=boom)
    assert "network error" in str(excinfo.value)


def test_fetch_models_maps_unreadable_body():
    def boom(*args, **kwargs):
        raise json.JSONDecodeError("bad", "{}", 0)

    with pytest.raises(lm.ModelListError) as excinfo:
        lm.fetch_models(_provider("groq"), FAKE_KEY, get_json=boom)
    assert "unreadable response" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# discover()
# --------------------------------------------------------------------------- #
def test_discover_without_key_reports_absent():
    report = lm.discover(_provider("groq"), "")
    assert report.key_present is False
    assert report.ok is False


def test_discover_records_error_and_keeps_going():
    def boom(*args, **kwargs):
        raise _http_error(403)

    report = lm.discover(_provider("cerebras"), FAKE_KEY, fetch=lambda p, k: lm.fetch_models(p, k, get_json=boom))
    assert report.key_present is True
    assert report.ok is False
    assert "rejected" in (report.error or "")


def test_discover_picks_model_and_reports_key_length_only():
    report = lm.discover(
        _provider("cerebras"), FAKE_KEY, fetch=lambda p, k: ["llama-3.3-70b", "gpt-oss-120b"]
    )
    assert report.chosen == "gpt-oss-120b"
    assert report.key_length == len(FAKE_KEY)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_main_prints_env_lines_and_never_the_key(tmp_path, capsys, clean_env, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                f"GROQ_API_KEY={FAKE_KEY}",
                f"GEMINI_API_KEY={FAKE_KEY}",
            ]
        ),
        encoding="utf-8",
    )
    payloads = {
        "api.groq.com": {"data": [{"id": "llama-3.3-70b-versatile"}, {"id": "openai/gpt-oss-120b"}]},
        "generativelanguage.googleapis.com": {"models": [{"name": "models/gemini-3.8-flash"}]},
    }

    def fake_get_json(url, headers, **kwargs):
        for host, payload in payloads.items():
            if host in url:
                return payload
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(lm, "_http_get_json", fake_get_json)
    exit_code = lm.main(["--env-file", str(env)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "GROQ_MODEL=openai/gpt-oss-120b" in out
    assert "GEMINI_MODEL=gemini-3.8-flash" in out
    assert FAKE_KEY not in out, "the tool must never echo a key"
    assert "CEREBRAS_API_KEY" in out  # reported as skipped


def test_main_without_any_keys_points_at_the_signup_pages(tmp_path, capsys, clean_env):
    env = tmp_path / ".env"
    env.write_text("LOG_LEVEL=INFO\n", encoding="utf-8")
    exit_code = lm.main(["--env-file", str(env)])
    out = capsys.readouterr().out
    assert exit_code == 2
    for provider in lm.PROVIDERS:
        assert provider.key_var in out
        assert provider.signup_url in out


def test_main_returns_one_when_every_provider_fails(tmp_path, capsys, clean_env, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("\n".join(f"{p.key_var}={FAKE_KEY}" for p in lm.PROVIDERS), encoding="utf-8")

    def boom(provider, api_key, **kwargs):
        raise lm.ModelListError("HTTP 500")

    monkeypatch.setattr(lm, "fetch_models", boom)
    exit_code = lm.main(["--env-file", str(env)])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert out.count("HTTP 500") == len(lm.PROVIDERS)


def test_main_reports_skipped_providers_by_name(tmp_path, capsys, clean_env, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.setattr(lm, "fetch_models", lambda provider, key, **kw: ["openai/gpt-oss-120b"])
    assert lm.main(["--env-file", str(env)]) == 0
    out = capsys.readouterr().out
    assert "- cerebras: no CEREBRAS_API_KEY set" in out
    assert "- gemini: no GEMINI_API_KEY set" in out
