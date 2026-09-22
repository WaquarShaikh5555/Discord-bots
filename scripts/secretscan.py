#!/usr/bin/env python3
"""Scan files or staged git content for accidentally committed credentials.

Standard library only, on purpose: this module backs a git pre-commit hook,
which has to keep working before anyone has installed the project's
dependencies.

Three detection layers, from specific to generic:

1. Known provider formats (Groq, Cerebras, Google, OpenAI, Anthropic, GitHub,
   Slack, AWS, Discord bot tokens). These are matched on prefix *plus* a
   minimum length so that prose like "Groq keys start with ``gsk_``" does not
   trip the scanner.
2. Secret-named assignments (``*_TOKEN``, ``*_API_KEY``, ``*_SECRET``,
   ``*_PASSWORD``, ...) whose value looks like a real credential: non-empty,
   not a documented placeholder, no internal whitespace, and high enough
   Shannon entropy. This catches formats layer 1 has never heard of.
3. Private key blocks (PEM).

Exit codes: 0 = clean, 1 = findings, 2 = usage/internal error.

Usage:
    python -m scripts.secretscan                 # scan tracked text files
    python -m scripts.secretscan --staged        # scan git index (hook mode)
    python -m scripts.secretscan path/to/file    # scan explicit paths
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

# --- Known credential formats -----------------------------------------------
# Each pattern requires enough trailing entropy that documentation mentioning
# the prefix in prose is not reported.
KNOWN_FORMATS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Groq API key", re.compile(r"\bgsk_[A-Za-z0-9]{16,}\b")),
    ("Cerebras API key", re.compile(r"\bcsk-[A-Za-z0-9]{16,}\b")),
    ("Google API key (legacy)", re.compile(r"\bAIza[A-Za-z0-9_\-]{30,}\b")),
    ("Google API key (new format)", re.compile(r"\bAQ\.[A-Za-z0-9_\-]{20,}\b")),
    ("OpenAI API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("GitHub fine-grained PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("AWS access key ID", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "Discord bot token",
        re.compile(r"\b[A-Za-z0-9_\-]{24}\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{25,}\b"),
    ),
    ("Stripe secret key", re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("Telegram bot token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{30,}\b")),
)

PEM_BLOCK = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")

# --- Generic secret-named assignment ----------------------------------------
# KEY=VALUE or export KEY=VALUE, in .env style files and shell scripts.
ASSIGNMENT = re.compile(
    r"""^\s*(?:export\s+)?
        (?P<key>[A-Za-z_][A-Za-z0-9_]*)
        \s*(?::|=)\s*
        (?P<quote>["']?)
        (?P<value>[^"'\r\n#]*)
    """,
    re.VERBOSE,
)

SECRET_NAME = re.compile(
    r"(?:TOKEN|API_KEY|APIKEY|SECRET|PASSWORD|PASSWD|PASSPHRASE|PRIVATE_KEY"
    r"|ACCESS_KEY|AUTH|CREDENTIAL|DSN)$",
    re.IGNORECASE,
)

# Values that are clearly stand-ins rather than credentials.
PLACEHOLDER = re.compile(
    r"""(?:
        ^\s*$                     |  # empty
        <.*>                      |  # <your-token>
        \{\{.*\}\}                |  # {{ secret }}
        \$\{[^}]*\}               |  # ${ENV_VAR}
        \b(?:your|my|the|some|any|new|old)\b     |
        \b(?:xxx+|placeholder|example|sample|dummy|fake|test|changeme|change_me) \b |
        \b(?:paste|insert|put|enter|fill|replace|add|set)\b |
        \b(?:here|there|value|token|key|secret|id)\b |
        \.{3}                     |  # ...
        _+$                          # trailing underscores
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Names whose values are legitimate non-secrets even when they look random.
NAME_ALLOWLIST = frozenset(
    {
        "TICKET_NAME_PATTERN",
        "AI_PROVIDER_CHAIN",
        "LOG_LEVEL",
        "DEV_GUILD_IDS",
        "GEMINI_MODEL",
        "GROQ_MODEL",
        "CEREBRAS_MODEL",
    }
)

MIN_GENERIC_LEN = 20
MIN_GENERIC_ENTROPY = 3.2

# Layer 2 only runs where ``KEY=value`` is genuine configuration syntax. In
# source code the same shape is a keyword argument (``api_key=_env_secret(...)``)
# and reporting it is noise. Layer 1 (known formats) and PEM blocks still scan
# every file type.
GENERIC_ASSIGNMENT_NAMES = frozenset({"Dockerfile", ".env", ".env.example"})
GENERIC_ASSIGNMENT_SUFFIXES = frozenset(
    {
        ".env", ".sh", ".bash", ".yml", ".yaml", ".toml", ".ini", ".cfg",
        ".conf", ".properties", ".service", ".env.example",
    }
)


def _generic_assignment_applies(path: str) -> bool:
    """True for config-style files where KEY=value carries a real value."""
    base = os.path.basename(path)
    if base in GENERIC_ASSIGNMENT_NAMES:
        return True
    if base.startswith("Dockerfile"):
        return True
    if base.startswith(".env"):  # .env, .env.example, .env.local, ...
        return True
    for suffix in GENERIC_ASSIGNMENT_SUFFIXES:
        if path.endswith(suffix):
            return True
    return False



@dataclass(frozen=True)
class Finding:
    """One suspected credential."""

    path: str
    line_no: int
    label: str
    preview: str

    def format(self) -> str:
        return f"{self.path}:{self.line_no}: {self.label} -> {self.preview}"


def shannon_entropy(value: str) -> float:
    """Bits of information per character; real keys sit well above 3.5."""
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for char in value:
        counts[char] = counts.get(char, 0) + 1
    length = len(value)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def mask(value: str) -> str:
    """Render a credential safely for logs: length + short prefix, never more.

    A scanner that prints the secret it found would leak it a second time --
    into CI logs, terminal scrollback, or a pasted bug report.
    """
    value = value.strip()
    if not value:
        return "<empty>"
    prefix = value[:5] if len(value) > 12 else value[:2]
    return f"{prefix}…[{len(value)} chars]"


def _looks_like_placeholder(value: str) -> bool:
    if not value.strip():
        return True
    if PLACEHOLDER.search(value):
        return True
    # Real credentials never contain spaces; prose and shell comments do.
    if re.search(r"\s", value.strip()):
        return True
    return False


def scan_text(text: str, path: str = "<string>") -> list[Finding]:
    """Return every suspected credential in ``text``."""
    findings: list[Finding] = []
    run_generic = _generic_assignment_applies(path)
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        if PEM_BLOCK.search(line):
            findings.append(Finding(path, line_no, "private key block", "<PEM>"))
            continue

        for label, pattern in KNOWN_FORMATS:
            for match in pattern.finditer(line):
                findings.append(Finding(path, line_no, label, mask(match.group(0))))

        if not run_generic:
            continue

        # Commented-out credentials are still committed credentials, so the
        # assignment check runs on the line with any leading '#' removed.
        assignment = ASSIGNMENT.match(stripped.lstrip("#").strip())
        if not assignment:
            continue
        key = assignment.group("key")
        value = assignment.group("value").strip()
        if key.upper() in NAME_ALLOWLIST:
            continue
        if not SECRET_NAME.search(key):
            continue
        if _looks_like_placeholder(value):
            continue
        if len(value) < MIN_GENERIC_LEN:
            continue
        if shannon_entropy(value) < MIN_GENERIC_ENTROPY:
            continue
        findings.append(
            Finding(path, line_no, f"{key} holds a high-entropy value", mask(value))
        )

    return _dedupe(findings)


def _dedupe(findings: Sequence[Finding]) -> list[Finding]:
    seen: set[tuple[str, int, str, str]] = set()
    unique: list[Finding] = []
    for finding in findings:
        key = (finding.path, finding.line_no, finding.label, finding.preview)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return list(unique)


# --- Sources -----------------------------------------------------------------

TEXT_SUFFIXES = frozenset(
    {
        ".py", ".js", ".ts", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini",
        ".md", ".txt", ".sh", ".bash", ".env", ".example", ".service", ".sql",
        ".html", ".css", ".dockerfile", "",
    }
)

SKIP_DIRS = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", "data", ".mypy_cache",
     ".pytest_cache", ".ruff_cache", "dist", "build"}
)


def _is_text_candidate(path: str) -> bool:
    base = os.path.basename(path)
    if base in {".env", ".env.example"}:
        return True
    if base.startswith("Dockerfile"):
        return True
    _, ext = os.path.splitext(base)
    return ext.lower() in TEXT_SUFFIXES


def iter_repo_files(root: str = ".") -> Iterable[str]:
    """Yield tracked text files; falls back to a directory walk without git."""
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                full = os.path.join(dirpath, name)
                if _is_text_candidate(full):
                    yield os.path.relpath(full, root)
        return
    for rel in out.splitlines():
        rel = rel.strip()
        if rel and _is_text_candidate(rel):
            yield rel


def scan_paths(paths: Sequence[str], root: str = ".") -> list[Finding]:
    findings: list[Finding] = []
    for rel in paths:
        full = os.path.join(root, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as handle:
                findings.extend(scan_text(handle.read(), rel))
        except OSError as exc:
            print(f"secretscan: cannot read {rel}: {exc}", file=sys.stderr)
    return findings


def staged_entries(root: str = ".") -> list[tuple[str, str]]:
    """Return (path, staged content) for added/copied/modified index entries."""
    listing = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    entries: list[tuple[str, str]] = []
    for rel in listing.splitlines():
        rel = rel.strip()
        if not rel or not _is_text_candidate(rel):
            continue
        blob = subprocess.run(
            ["git", "show", f":{rel}"], cwd=root, capture_output=True, text=True
        )
        if blob.returncode == 0:
            entries.append((rel, blob.stdout))
    return entries


def scan_staged(root: str = ".") -> list[Finding]:
    findings: list[Finding] = []
    for rel, content in staged_entries(root):
        findings.extend(scan_text(content, rel))
    return findings


REMEDIATION = """
What to do
----------
1. Do NOT push. Remove the value from the file and keep real credentials in a
   local `.env` (git-ignored). `.env.example` is COMMITTED -- it must only ever
   hold empty or obviously fake values.
2. If it was already pushed, the credential is compromised. ROTATE it in the
   provider's console -- that is the only real fix. Rewriting git history does
   not undo the exposure: scrapers read public repos within minutes, and GitHub
   keeps orphaned commits reachable by SHA until garbage collection.
3. Bypass only if you are certain this is a false positive:
       git commit --no-verify
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="files to scan (default: tracked)")
    parser.add_argument("--staged", action="store_true", help="scan the git index")
    parser.add_argument("--quiet", action="store_true", help="only report failures")
    args = parser.parse_args(argv)

    try:
        if args.staged:
            findings = scan_staged()
        elif args.paths:
            findings = scan_paths(args.paths)
        else:
            findings = scan_paths(list(iter_repo_files()))
    except subprocess.CalledProcessError as exc:
        print(f"secretscan: git failed: {exc}", file=sys.stderr)
        return 2

    if findings:
        print(f"secretscan: {len(findings)} suspected credential(s) found\n", file=sys.stderr)
        for finding in findings:
            print(f"  {finding.format()}", file=sys.stderr)
        print(REMEDIATION, file=sys.stderr)
        return 1

    if not args.quiet:
        print("secretscan: no credentials found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
