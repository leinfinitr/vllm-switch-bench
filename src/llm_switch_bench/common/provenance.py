from __future__ import annotations

from pathlib import Path
from typing import Any


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
        "cannot locate the llm-switch-bench checkout; run from the repository root "
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
    return {
        "path": str(path.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_porcelain": status,
    }
