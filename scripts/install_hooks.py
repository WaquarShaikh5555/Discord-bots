#!/usr/bin/env python3
"""Install the repository's git hooks.

    python -m scripts.install_hooks

Installs a small shim at ``.git/hooks/pre-commit`` that delegates to the
version-controlled hook in ``scripts/hooks/pre-commit``, so the logic stays in
git and later edits take effect without re-installing.

Deliberately does NOT set ``core.hooksPath``: that would silently disable every
other hook already present in ``.git/hooks`` (CI integrations, commit-msg
trailers, husky, and so on). Instead any pre-existing ``pre-commit`` is renamed
to ``pre-commit.pre-secretscan.bak`` and still runs, after the secret scan
passes.

    python -m scripts.install_hooks --uninstall   # restore the previous state

Safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path

SOURCE_HOOK = Path("scripts/hooks/pre-commit")
HOOK_NAME = "pre-commit"
BACKUP_NAME = "pre-commit.pre-secretscan.bak"
MARKER = "Managed by scripts/install_hooks.py"

# The shim resolves the repo root at run time so it survives the clone being
# moved, and chains any hook that was already installed.
SHIM = f"""#!/bin/sh
# {MARKER} — do not edit; re-run the installer instead.
# Delegates to the version-controlled hook, then chains any pre-existing one.
root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
here=$(dirname "$0")

if [ -x "$root/{SOURCE_HOOK}" ]; then
    "$root/{SOURCE_HOOK}" "$@" || exit 1
fi

if [ -x "$here/{BACKUP_NAME}" ]; then
    exec "$here/{BACKUP_NAME}" "$@"
fi

exit 0
"""


def _git(*args: str, check: bool = True, cwd: str | Path = ".") -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


def repo_root() -> Path:
    return Path(_git("rev-parse", "--show-toplevel").stdout.strip())


def hooks_dir(root: Path) -> Path:
    """The real hooks directory, honouring worktrees and core.hooksPath."""
    out = _git("rev-parse", "--git-path", "hooks", cwd=root).stdout.strip()
    path = Path(out)
    return path if path.is_absolute() else (root / path).resolve()


def existing_local_hooks(git_dir: Path) -> list[str]:
    """Hooks already present in a git directory, ignoring git's own samples."""
    hooks = git_dir / "hooks" if (git_dir / "hooks").is_dir() else git_dir
    if not hooks.is_dir():
        return []
    return sorted(
        p.name
        for p in hooks.iterdir()
        if p.is_file() and not p.name.endswith(".sample") and not p.name.endswith(".bak")
    )


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def is_ours(path: Path) -> bool:
    try:
        return MARKER in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def install(root: Path) -> int:
    source = root / SOURCE_HOOK
    if not source.is_file():
        print(f"install_hooks: {SOURCE_HOOK} not found at {root}", file=sys.stderr)
        return 2

    # git skips hooks without the executable bit, silently.
    _make_executable(source)

    target_dir = hooks_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / HOOK_NAME
    backup = target_dir / BACKUP_NAME

    if target.exists() and not is_ours(target):
        if backup.exists():
            print(
                f"install_hooks: leaving the existing {HOOK_NAME} hook in place — "
                f"{backup.name} already exists, so it cannot be preserved automatically."
            )
            print(f"               resolve {target} by hand, then re-run.")
            return 1
        target.rename(backup)
        _make_executable(backup)
        print(f"install_hooks: kept your previous hook as {BACKUP_NAME} (it still runs)")

    target.write_text(SHIM, encoding="utf-8")
    _make_executable(target)

    if not is_ours(target):
        print("install_hooks: wrote the shim but could not verify it", file=sys.stderr)
        return 1

    others = [h for h in existing_local_hooks(target_dir) if h not in {HOOK_NAME, BACKUP_NAME}]
    print(f"install_hooks: {HOOK_NAME} secret scanner installed at {target}")
    if others:
        print(f"  (other hooks left untouched: {', '.join(others)})")
    print()
    print("  Blocks commits whose staged content looks like it contains real credentials.")
    print("  Verify with:  python -m scripts.secretscan --staged")
    print("  Bypass once:  git commit --no-verify")
    return 0


def uninstall(root: Path) -> int:
    target_dir = hooks_dir(root)
    target = target_dir / HOOK_NAME
    backup = target_dir / BACKUP_NAME

    if not target.exists():
        print(f"install_hooks: nothing to remove ({target} does not exist)")
        return 0
    if not is_ours(target):
        print(f"install_hooks: {target} was not installed by this script — leaving it alone")
        return 0

    target.unlink()
    if backup.exists():
        backup.rename(target)
        _make_executable(target)
        print(f"install_hooks: restored your previous {HOOK_NAME} hook from {BACKUP_NAME}")
    else:
        print(f"install_hooks: removed {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install this repo's git hooks.")
    parser.add_argument("--uninstall", action="store_true", help="restore the previous state")
    args = parser.parse_args(argv)

    try:
        root = repo_root()
    except subprocess.CalledProcessError:
        print("install_hooks: not inside a git repository", file=sys.stderr)
        return 2
    if not os.path.isdir(root / ".git") and not os.path.isfile(root / ".git"):
        print(f"install_hooks: {root} does not look like a git checkout", file=sys.stderr)
        return 2

    return uninstall(root) if args.uninstall else install(root)


if __name__ == "__main__":
    raise SystemExit(main())
