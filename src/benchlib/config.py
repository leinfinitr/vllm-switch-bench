from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .sampling import run_cmd


def expand_path(value: str) -> Path:
    return Path(value).expanduser().resolve()



def validate_repo_path(repo_path: Path) -> Path:
    if not repo_path.exists():
        raise SystemExit(f"repo path does not exist: {repo_path}")
    if not (repo_path / ".git").exists():
        raise SystemExit(f"repo path is not a git checkout: {repo_path}")
    return repo_path.resolve()



def load_baseline3_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "model" in raw:
        raw["model"] = expand_path(str(raw["model"]))
    systems = raw.get("systems") or {}
    for name, cfg in systems.items():
        if "repo" in cfg:
            cfg["repo"] = validate_repo_path(expand_path(str(cfg["repo"])))
    raw["systems"] = systems
    raw["config_path"] = path.resolve()
    return raw



def _git_output(repo: Path, *args: str) -> str:
    return run_cmd(["git", "-C", str(repo), *args], timeout=30).stdout.strip()



def collect_repo_metadata(repo: Path) -> dict[str, Any]:
    repo = validate_repo_path(repo)
    return {
        "repo_path": str(repo),
        "git_remote": _git_output(repo, "remote", "get-url", "origin"),
        "git_branch": _git_output(repo, "branch", "--show-current"),
        "git_commit": _git_output(repo, "rev-parse", "HEAD"),
    }
