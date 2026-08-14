from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


def file_metadata(path: Path) -> dict[str, Any]:
    """Bind an external executable or configuration to the exact bytes used."""

    resolved = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def repository_root() -> Path:
    """Return the source checkout root for repository-bound commands.

    The benchmark package can also be installed into an isolated environment.  In
    that case no checkout can be inferred from ``__file__``; callers that need a
    repository should pass it explicitly or run from the checkout root.
    """

    candidates = (Path.cwd(), *Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "results").is_dir():
            return candidate
    raise RuntimeError(
        "cannot locate the vllm-switch-bench checkout; run from the repository root "
        "or pass an explicit path"
    )


def git_metadata(path: Path) -> dict[str, Any]:
    """Capture source identity without assuming that *path* is a Git checkout."""

    import subprocess

    def run(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(path), *arguments],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain", "--untracked-files=all")
    fingerprint = None
    listed = run("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if listed is not None:
        digest = hashlib.sha256()
        for relative in sorted(item for item in listed.split("\0") if item):
            candidate = path / relative
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            try:
                if candidate.is_symlink():
                    digest.update(b"L\0")
                    digest.update(os.readlink(candidate).encode("utf-8", errors="surrogateescape"))
                elif candidate.is_file():
                    digest.update(b"F\0")
                    with candidate.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                else:
                    digest.update(b"MISSING\0")
            except OSError as exc:
                digest.update(f"ERROR:{exc.errno}".encode())
            digest.update(b"\0")
        fingerprint = digest.hexdigest()
    return {
        "path": str(path.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_porcelain": status,
        "working_tree_sha256": fingerprint,
    }
