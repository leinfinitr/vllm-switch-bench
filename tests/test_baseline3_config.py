from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from benchlib.config import collect_repo_metadata, expand_path, load_baseline3_config


@pytest.fixture()
def external_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    return repo


def test_expand_path_and_validate_repo_path(tmp_path):
    repo = tmp_path / "serverless"
    repo.mkdir()
    (repo / ".git").mkdir()
    config_path = tmp_path / "baseline3.yaml"
    config_path.write_text(
        f"""
model: ~/models/hf/Qwen2.5-0.5B-Instruct
systems:
  serverless_llm:
    repo: {repo}
""",
        encoding="utf-8",
    )
    cfg = load_baseline3_config(config_path)
    assert str(cfg["model"]).startswith(str(Path.home()))
    assert cfg["systems"]["serverless_llm"]["repo"] == repo.resolve()


def test_collect_repo_metadata_returns_commit_sha(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    def fake_run_cmd(cmd, timeout=30, check=False):
        class CP:
            stdout = {
                "git -C <repo> remote get-url origin": "git@example.com:repo.git\n",
                "git -C <repo> branch --show-current": "main\n",
                "git -C <repo> rev-parse HEAD": "abc123\n",
            }[" ".join(cmd).replace(str(repo), "<repo>")]
        return CP()

    monkeypatch.setattr("benchlib.config.run_cmd", fake_run_cmd)
    meta = collect_repo_metadata(repo)
    assert meta["repo_path"] == str(repo.resolve())
    assert meta["git_commit"] == "abc123"
    assert meta["git_branch"] == "main"


def test_missing_repo_gives_actionable_error(tmp_path):
    missing = tmp_path / "missing"
    config_path = tmp_path / "baseline3.yaml"
    config_path.write_text(
        f"""
model: /tmp/model
systems:
  swapserve_llm:
    repo: {missing}
""",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        load_baseline3_config(config_path)
    assert "repo path does not exist" in str(exc.value)
