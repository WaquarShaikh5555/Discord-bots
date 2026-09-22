"""Secret scanner: detection, false-positive resistance, and repo hygiene.

Every credential literal in this file is assembled at runtime from fragments.
A fully written-out key in test source would be flagged by the repo-wide scan
below, and — more importantly — would put a realistic credential shape into a
public file for no reason.
"""

from __future__ import annotations

import random
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from scripts import secretscan
from scripts import install_hooks
from scripts.install_hooks import existing_local_hooks
from scripts.secretscan import (
    Finding,
    mask,
    scan_paths,
    scan_text,
    shannon_entropy,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DIGITS = "0123456789"


def _rand(length: int, alphabet: str = ALNUM, seed: int = 1234) -> str:
    return "".join(random.Random(seed + length).choices(alphabet, k=length))


# --- Synthetic credentials, correct shapes, no real values -------------------


def fake_groq() -> str:
    return "gsk" + "_" + _rand(52)


def fake_cerebras() -> str:
    return "csk" + "-" + _rand(48)


def fake_google_legacy() -> str:
    return "AIza" + _rand(35)


def fake_google_new() -> str:
    return "AQ" + "." + _rand(50)


def fake_discord_token() -> str:
    return _rand(24) + "." + _rand(6, seed=7) + "." + _rand(38, seed=9)


def fake_openai() -> str:
    return "sk" + "-" + _rand(48)


def fake_anthropic() -> str:
    return "sk" + "-ant-" + _rand(40)


def fake_github() -> str:
    return "ghp" + "_" + _rand(36)


def fake_github_pat() -> str:
    return "github" + "_pat_" + _rand(40)


def fake_slack() -> str:
    return "xoxb" + "-" + _rand(24)


def fake_aws() -> str:
    return "AKIA" + _rand(16, UPPER, seed=5)


def fake_stripe() -> str:
    return "sk" + "_live_" + _rand(24)


def fake_telegram() -> str:
    return _rand(9, DIGITS, seed=3) + ":" + _rand(35)


# --- Entropy and masking -----------------------------------------------------


def test_entropy_of_random_key_is_high():
    assert shannon_entropy(fake_groq()) > 3.5


def test_entropy_of_repeated_char_is_zero():
    assert shannon_entropy("aaaaaaaaaa") == 0.0


def test_entropy_of_empty_string_is_zero():
    assert shannon_entropy("") == 0.0


def test_entropy_of_lowercase_placeholder_is_lower_than_key():
    assert shannon_entropy("yourdiscordbottokenhere") < shannon_entropy(fake_groq())


def test_mask_hides_the_value():
    secret = fake_groq()
    masked = mask(secret)
    assert secret not in masked
    assert len(masked) < len(secret)
    assert str(len(secret)) in masked


def test_mask_reveals_at_most_a_short_prefix():
    secret = fake_cerebras()
    masked = mask(secret)
    # Prefix shown, then an ellipsis — never the tail, which is the secret part.
    assert masked.count("…") == 1
    shown = masked.split("…")[0]
    assert len(shown) <= 5


def test_mask_of_empty_value():
    assert mask("") == "<empty>"
    assert mask("   ") == "<empty>"


def test_mask_of_short_value_is_still_not_verbatim_for_long_input():
    assert mask("short") == "sh…[5 chars]"


def test_finding_format_is_masked():
    finding = Finding(".env.example", 12, "Groq API key", mask(fake_groq()))
    rendered = finding.format()
    assert rendered.startswith(".env.example:12:")
    assert "Groq API key" in rendered
    assert fake_groq() not in rendered


# --- Layer 1: known provider formats ----------------------------------------


@pytest.mark.parametrize(
    ("builder", "label"),
    [
        (fake_groq, "Groq API key"),
        (fake_cerebras, "Cerebras API key"),
        (fake_google_legacy, "Google API key (legacy)"),
        (fake_google_new, "Google API key (new format)"),
        (fake_discord_token, "Discord bot token"),
        (fake_openai, "OpenAI API key"),
        (fake_anthropic, "Anthropic API key"),
        (fake_github, "GitHub token"),
        (fake_github_pat, "GitHub fine-grained PAT"),
        (fake_slack, "Slack token"),
        (fake_aws, "AWS access key ID"),
        (fake_stripe, "Stripe secret key"),
        (fake_telegram, "Telegram bot token"),
    ],
)
def test_known_formats_are_detected(builder, label):
    findings = scan_text(builder(), ".env.example")
    assert any(f.label == label for f in findings), [f.label for f in findings]


def test_google_new_format_is_detected_by_layer_one_even_with_plain_name():
    """Regression: an `AQ.`-prefixed key must not depend on the variable name."""
    findings = scan_text("WHATEVER=" + fake_google_new(), "notes.txt")
    assert any("Google API key (new format)" in f.label for f in findings)


def test_known_format_detected_in_any_file_type():
    assert scan_text("# " + fake_groq(), "README.md")
    assert scan_text("key = " + fake_groq(), "bot/config.py")


def test_pem_block_detected():
    # Assembled from fragments: a literal PEM header in committed source would be
    # reported by the repo-wide scan below (and by the pre-commit hook).
    dashes = "-" * 5
    text = (
        f"{dashes}BEGIN {'RSA PRIVATE KEY'}{dashes}\n"
        f"MIIEow...\n"
        f"{dashes}END {'RSA PRIVATE KEY'}{dashes}\n"
    )
    findings = scan_text(text, "id_rsa.txt")
    assert any("private key" in f.label for f in findings)


# --- Layer 2: generic secret-named assignment -------------------------------


def test_unfamiliar_secret_name_is_caught_by_entropy_rule():
    findings = scan_text("SUPPORT_DESK_TOKEN=" + _rand(40, seed=11), ".env")
    assert any("high-entropy" in f.label for f in findings)


def test_quoted_value_is_caught():
    findings = scan_text('SUPPORT_DESK_TOKEN="' + _rand(40, seed=12) + '"', ".env")
    assert findings


def test_export_prefix_is_caught():
    findings = scan_text("export SUPPORT_DESK_TOKEN=" + _rand(40, seed=13), "run.sh")
    assert findings


def test_yaml_mapping_is_caught():
    findings = scan_text("  api_secret: " + _rand(40, seed=14), "docker-compose.yml")
    assert findings


def test_commented_out_secret_is_still_caught():
    findings = scan_text("# GROQ_API_KEY=" + fake_groq(), ".env.example")
    assert findings


def test_secret_in_dockerfile_env_is_caught():
    findings = scan_text("ENV GROQ_API_KEY=" + fake_groq(), "Dockerfile")
    assert findings


def test_generic_rule_skips_source_code_keyword_arguments():
    """`api_key=_env_secret(...)` in Python is a kwarg, not a credential."""
    line = "        api_key=_env_secret(PROVIDER_API_KEY_ENV[name]),"
    assert scan_text(line, "bot/config.py") == []


def test_generic_rule_skips_markdown_prose():
    assert scan_text("Set SOME_API_TOKEN to the value from your provider dashboard.", "README.md") == []


def test_generic_rule_still_catches_known_format_inside_python():
    line = 'api_key = "' + fake_groq() + '"'
    findings = scan_text(line, "bot/config.py")
    assert any(f.label == "Groq API key" for f in findings)


# --- False positives --------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "DISCORD_BOT_TOKEN=",
        "GROQ_API_KEY=",
        "CEREBRAS_API_KEY=",
        "GEMINI_API_KEY=",
        "DEV_GUILD_IDS=",
        "GROQ_API_KEY=gsk_your_key_here",
        "DISCORD_BOT_TOKEN=<paste-your-token>",
        "DISCORD_BOT_TOKEN=xxxx",
        "GROQ_API_KEY=changeme",
        "GROQ_API_KEY=gsk_PLACEHOLDER",
        "AI_PROVIDER_CHAIN=groq,cerebras,gemini",
        "GROQ_MODEL=llama-3.3-70b-versatile",
        "CEREBRAS_MODEL=llama3.3-70b",
        "GEMINI_MODEL=gemini-2.0-flash",
        "DATABASE_URL=sqlite+aiosqlite:///data/tickets.db",
        "TICKET_NAME_PATTERN=^(ticket|support|help)[-_]",
        "LOG_LEVEL=INFO",
        "AI_TEMPERATURE=0.2",
        "REQUIRE_CONFIGURED_CATEGORY=true",
        "HISTORY_MESSAGE_LIMIT=10",
    ],
)
def test_benign_template_lines_are_not_flagged(line):
    assert scan_text(line, ".env.example") == [], line


def test_documentation_mentioning_prefixes_is_not_flagged():
    text = (
        "# Groq keys start with gsk_ and Cerebras keys with csk-.\n"
        "# Google legacy keys start with AIza.\n"
        "# See https://console.groq.com/keys\n"
    )
    assert scan_text(text, ".env.example") == []


def test_postgres_example_with_placeholder_password_is_not_flagged():
    line = "# DATABASE_URL=postgresql+asyncpg://postgres.PROJECT_REF:PASSWORD@aws-0-region.pooler.supabase.com:5432/postgres"
    assert scan_text(line, ".env.example") == []


def test_low_entropy_long_value_is_not_flagged():
    assert scan_text("SUPPORT_DESK_TOKEN=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", ".env") == []


def test_short_value_is_not_flagged():
    assert scan_text("SUPPORT_DESK_TOKEN=abc123", ".env") == []


def test_allowlisted_names_are_never_flagged_by_layer_two():
    assert scan_text("TICKET_NAME_PATTERN=" + _rand(40, seed=21), ".env") == []


def test_empty_text_has_no_findings():
    assert scan_text("", ".env") == []


def test_duplicate_matches_are_deduplicated():
    secret = fake_groq()
    findings = scan_text(secret + " " + secret, ".env")
    labels = [(f.line_no, f.label, f.preview) for f in findings]
    assert len(labels) == len(set(labels))


# --- The incident this module exists for ------------------------------------


def test_filled_env_example_is_rejected():
    """Replay of the real leak: four live keys pasted into the committed template."""
    leaked = "\n".join(
        [
            "DISCORD_BOT_TOKEN=" + fake_discord_token(),
            "GROQ_API_KEY=" + fake_groq(),
            "CEREBRAS_API_KEY=" + fake_cerebras(),
            "GEMINI_API_KEY=" + fake_google_new(),
        ]
    )
    findings = scan_text(leaked, ".env.example")
    flagged_lines = {f.line_no for f in findings}
    assert flagged_lines == {1, 2, 3, 4}, findings


def test_committed_env_example_is_clean():
    """Guard on the actual file: it is public, so it must hold no credentials."""
    example = REPO_ROOT / ".env.example"
    assert example.is_file(), "expected .env.example to exist"
    findings = scan_text(example.read_text(encoding="utf-8"), ".env.example")
    assert findings == [], [f.format() for f in findings]


def test_committed_env_example_leaves_every_secret_empty():
    """The four secret lines must be bare `KEY=` so nobody pastes into them."""
    example = REPO_ROOT / ".env.example"
    values = {}
    for line in example.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    for key in ("DISCORD_BOT_TOKEN", "GROQ_API_KEY", "CEREBRAS_API_KEY", "GEMINI_API_KEY"):
        assert values.get(key) == "", f"{key} must be empty in the committed template"


def test_env_example_warns_that_it_is_committed():
    """Root cause of the leak: nothing said this file is public."""
    header = "\n".join((REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()[:20])
    assert "COMMITTED" in header.upper()
    assert ".env" in header


def test_env_example_has_no_inline_comments_on_value_lines():
    """python-dotenv handles `KEY=value # note` inconsistently across versions."""
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        value = stripped.split("=", 1)[1]
        assert "#" not in value, f"inline comment on value line: {line!r}"


def test_no_tracked_file_contains_a_credential():
    """Repo-wide guard: fails the suite the moment a real key is committed."""
    tracked = [p for p in secretscan.iter_repo_files(str(REPO_ROOT))]
    assert tracked, "expected to find tracked files to scan"
    findings = scan_paths(tracked, str(REPO_ROOT))
    assert findings == [], [f.format() for f in findings]


# --- Staged-content scanning (hook mode) ------------------------------------

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    return tmp_path


@requires_git
def test_scan_staged_finds_a_secret_added_to_the_index(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / ".env.example").write_text("GROQ_API_KEY=" + fake_groq() + "\n")
    _git(tmp_path, "add", ".env.example")

    findings = secretscan.scan_staged(str(tmp_path))
    assert any(f.label == "Groq API key" for f in findings), findings


@requires_git
def test_scan_staged_passes_a_clean_index(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / ".env.example").write_text("GROQ_API_KEY=\n")
    _git(tmp_path, "add", ".env.example")

    assert secretscan.scan_staged(str(tmp_path)) == []


@requires_git
def test_scan_staged_ignores_unstaged_working_tree_changes(tmp_path):
    """The hook must judge what is being committed, not what is on disk."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / ".env.example").write_text("GROQ_API_KEY=\n")
    _git(tmp_path, "add", ".env.example")
    # Dirty the working tree AFTER staging; the index is still clean.
    (tmp_path / ".env.example").write_text("GROQ_API_KEY=" + fake_groq() + "\n")

    assert secretscan.scan_staged(str(tmp_path)) == []


def test_scan_paths_reports_missing_files_without_crashing(tmp_path, capsys):
    findings = scan_paths(["does-not-exist.env"], str(tmp_path))
    assert findings == []
    assert "cannot read" in capsys.readouterr().err


def test_is_text_candidate_covers_env_files():
    assert secretscan._is_text_candidate(".env")
    assert secretscan._is_text_candidate(".env.example")
    assert secretscan._is_text_candidate("Dockerfile")
    assert secretscan._is_text_candidate("deploy/ticket-bot.service")
    assert not secretscan._is_text_candidate("data/tickets.db")
    assert not secretscan._is_text_candidate("logo.png")


# --- Hook and installer -----------------------------------------------------


def test_pre_commit_hook_exists_and_is_a_shell_script():
    hook = REPO_ROOT / "scripts" / "hooks" / "pre-commit"
    assert hook.is_file(), "expected scripts/hooks/pre-commit"
    body = hook.read_text(encoding="utf-8")
    assert body.startswith("#!"), "hook needs a shebang or git will not run it"
    assert "secretscan" in body
    assert "--staged" in body


def test_pre_commit_hook_is_executable():
    hook = REPO_ROOT / "scripts" / "hooks" / "pre-commit"
    mode = hook.stat().st_mode
    assert mode & stat.S_IXUSR, "hook must be executable (git skips non-executable hooks)"


def test_pre_commit_hook_fails_open_when_scanner_is_absent():
    body = (REPO_ROOT / "scripts" / "hooks" / "pre-commit").read_text(encoding="utf-8")
    assert "[ -f scripts/secretscan.py ] || exit 0" in body


def test_pre_commit_hook_blocks_on_findings():
    body = (REPO_ROOT / "scripts" / "hooks" / "pre-commit").read_text(encoding="utf-8")
    assert "exit 1" in body


def test_existing_local_hooks_ignores_samples_and_backups(tmp_path):
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "pre-commit.sample").write_text("#!/bin/sh\n")
    (hooks / "post-checkout.sample").write_text("#!/bin/sh\n")
    (hooks / "pre-commit.pre-secretscan.bak").write_text("#!/bin/sh\n")
    assert existing_local_hooks(tmp_path) == []


def test_existing_local_hooks_reports_real_hooks(tmp_path):
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "pre-commit.sample").write_text("#!/bin/sh\n")
    (hooks / "commit-msg").write_text("#!/bin/sh\n")
    assert existing_local_hooks(tmp_path) == ["commit-msg"]


def test_existing_local_hooks_handles_missing_directory(tmp_path):
    assert existing_local_hooks(tmp_path) == []


def test_shim_delegates_to_the_version_controlled_hook():
    assert "scripts/hooks/pre-commit" in install_hooks.SHIM
    assert install_hooks.MARKER in install_hooks.SHIM


def test_shim_chains_a_preexisting_hook_instead_of_replacing_it():
    assert install_hooks.BACKUP_NAME in install_hooks.SHIM
    assert "exec" in install_hooks.SHIM


@requires_git
def test_install_does_not_set_core_hookspath(tmp_path, monkeypatch):
    """core.hooksPath hijacks ALL hooks, silently disabling unrelated ones."""
    repo = _init_repo(tmp_path)
    shutil.copytree(REPO_ROOT / "scripts", repo / "scripts")
    monkeypatch.chdir(repo)
    assert install_hooks.main([]) == 0
    result = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=repo, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "", "installer must not take over the hooks directory"


def test_is_ours_recognises_only_the_managed_shim(tmp_path):
    ours = tmp_path / "ours"
    ours.write_text(install_hooks.SHIM)
    theirs = tmp_path / "theirs"
    theirs.write_text("#!/bin/sh\necho hi\n")
    assert install_hooks.is_ours(ours) is True
    assert install_hooks.is_ours(theirs) is False
    assert install_hooks.is_ours(tmp_path / "missing") is False


@requires_git
def test_install_preserves_other_hooks(tmp_path, monkeypatch):
    """Regression: an existing commit-msg hook must keep working after install."""
    repo = _init_repo(tmp_path)
    shutil.copytree(REPO_ROOT / "scripts", repo / "scripts")
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    sentinel = hooks / "commit-msg"
    sentinel.write_text("#!/bin/sh\nexit 0\n")
    sentinel.chmod(0o755)

    monkeypatch.chdir(repo)
    assert install_hooks.main([]) == 0

    assert sentinel.exists(), "the unrelated commit-msg hook must be left alone"
    installed = hooks / "pre-commit"
    assert installed.exists()
    assert installed.stat().st_mode & stat.S_IXUSR


@requires_git
def test_install_backs_up_and_chains_a_preexisting_pre_commit(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    shutil.copytree(REPO_ROOT / "scripts", repo / "scripts")
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    previous = hooks / "pre-commit"
    previous.write_text("#!/bin/sh\necho previous-ran\nexit 0\n")
    previous.chmod(0o755)

    monkeypatch.chdir(repo)
    assert install_hooks.main([]) == 0

    backup = hooks / install_hooks.BACKUP_NAME
    assert backup.exists(), "the developer's own hook must be preserved"
    assert install_hooks.is_ours(previous)
    # The shim chains the backup, so it still executes.
    out = subprocess.run(
        ["sh", str(previous)], cwd=repo, capture_output=True, text=True
    )
    assert "previous-ran" in out.stdout


@requires_git
def test_install_is_idempotent(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    shutil.copytree(REPO_ROOT / "scripts", repo / "scripts")
    monkeypatch.chdir(repo)
    assert install_hooks.main([]) == 0
    assert install_hooks.main([]) == 0
    assert not (repo / ".git" / "hooks" / install_hooks.BACKUP_NAME).exists()


@requires_git
def test_uninstall_restores_the_previous_hook(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    shutil.copytree(REPO_ROOT / "scripts", repo / "scripts")
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    previous = hooks / "pre-commit"
    previous.write_text("#!/bin/sh\necho previous-ran\nexit 0\n")
    previous.chmod(0o755)

    monkeypatch.chdir(repo)
    assert install_hooks.main([]) == 0
    assert install_hooks.main(["--uninstall"]) == 0

    restored = hooks / "pre-commit"
    assert restored.exists()
    assert not install_hooks.is_ours(restored)
    assert "previous-ran" in restored.read_text()
    assert not (hooks / install_hooks.BACKUP_NAME).exists()


@requires_git
def test_uninstall_is_a_noop_without_our_shim(tmp_path, monkeypatch, capsys):
    repo = _init_repo(tmp_path)
    monkeypatch.chdir(repo)
    assert install_hooks.main(["--uninstall"]) == 0
    assert "nothing to remove" in capsys.readouterr().out


@requires_git
def test_uninstall_refuses_to_delete_a_foreign_hook(tmp_path, monkeypatch, capsys):
    repo = _init_repo(tmp_path)
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    foreign = hooks / "pre-commit"
    foreign.write_text("#!/bin/sh\necho mine\n")

    monkeypatch.chdir(repo)
    assert install_hooks.main(["--uninstall"]) == 0
    assert foreign.exists()
    assert "leaving it alone" in capsys.readouterr().out


# --- CLI --------------------------------------------------------------------


def test_cli_returns_zero_on_clean_paths(tmp_path, capsys):
    target = tmp_path / "clean.env"
    target.write_text("GROQ_API_KEY=\n")
    assert secretscan.main([str(target)]) == 0


def test_cli_returns_one_on_findings(tmp_path, capsys):
    target = tmp_path / "dirty.env"
    target.write_text("GROQ_API_KEY=" + fake_groq() + "\n")
    assert secretscan.main([str(target)]) == 1
    err = capsys.readouterr().err
    assert "suspected credential" in err
    assert "rotate" in err.lower()
    # The CLI must never echo the credential it found.
    assert fake_groq() not in err


def test_cli_quiet_mode_still_reports_findings(tmp_path, capsys):
    target = tmp_path / "dirty.env"
    target.write_text("GROQ_API_KEY=" + fake_groq() + "\n")
    assert secretscan.main([str(target), "--quiet"]) == 1
    assert "suspected credential" in capsys.readouterr().err


def test_cli_quiet_mode_is_silent_when_clean(tmp_path, capsys):
    target = tmp_path / "clean.env"
    target.write_text("GROQ_API_KEY=\n")
    assert secretscan.main([str(target), "--quiet"]) == 0
    assert capsys.readouterr().out == ""
