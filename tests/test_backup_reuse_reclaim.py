from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_switch_bench.experiments.backup_reuse_reclaim import run as repeated


def test_module_repo_metadata_records_import_path_and_checkout(monkeypatch, tmp_path: Path):
    package = tmp_path / "checkout" / "package"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.touch()
    (tmp_path / "checkout" / ".git").touch()
    monkeypatch.setattr(
        repeated.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(module)),
    )
    monkeypatch.setattr(
        repeated,
        "git_metadata",
        lambda path: {"repo_path": str(path), "git_commit": "abc"},
    )

    metadata = repeated.module_repo_metadata("package")

    assert metadata == {
        "repo_path": str(tmp_path / "checkout"),
        "git_commit": "abc",
        "module_path": str(module),
    }


def test_repeated_sleep_requires_explicit_model_specs():
    with pytest.raises(SystemExit):
        repeated.parse_args([])

    args = repeated.parse_args(["--models", "model-a=/models/a", "--vllm-repo", "/vllm"])
    assert args.models == [repeated.ModelSpec(name="model-a", path="/models/a")]
    assert args.enforce_eager is False


def test_model_load_kwargs_forwards_eager_mode():
    args = repeated.parse_args(
        ["--models", "model-a=/models/a", "--vllm-repo", "/vllm", "--enforce-eager"]
    )

    kwargs = repeated.model_load_kwargs(args, args.models[0])

    assert kwargs["enforce_eager"] is True


def test_invocation_paths_survive_runtime_workdir_change(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(repeated.INVOCATION_DIR_ENV, raising=False)
    args = repeated.parse_args(
        [
            "--models",
            "model-a=models/a",
            "--vllm-repo",
            "runtime",
            "--out-dir",
            "output",
            "--coordinator-url",
            "http://127.0.0.1:9000",
            "--coordinator-repo",
            "controller",
            "--coordinator-config",
            "controller.yaml",
        ]
    )

    repeated.normalize_invocation_paths(args)

    assert args.models[0].path == str(tmp_path / "models/a")
    assert args.vllm_repo == tmp_path / "runtime"
    assert args.out_dir == str(tmp_path / "output")
    assert args.coordinator_repo == tmp_path / "controller"
    assert args.coordinator_config == tmp_path / "controller.yaml"


@pytest.mark.parametrize(
    "argv",
    [
        ["--models", "a=/models/a", "a=/models/b"],
        ["--models", "a=/models/a", "--iterations", "0"],
        ["--models", "a=/models/a", "--post-release-observation-s", "-1"],
        ["--models", "a=/models/a", "--min-worker-rss-reclaim-bytes", "1"],
        ["--models", "a=/models/a", "--expect-release", "--expect-reuse"],
        ["--models", "a=/models/a", "--iterations", "1", "--expect-release"],
        ["--models", "a=/models/a", "--iterations", "1", "--no-expect-release"],
        ["--models", "a=/models/a", "--iterations", "1", "--expect-reuse"],
        ["--models", "a=/models/a", "--coordinator-url", "http://127.0.0.1:9000"],
        ["--models", "a=/models/a", "--coordinator-repo", "/controller"],
    ],
)
def test_repeated_sleep_rejects_ambiguous_or_invalid_controls(argv):
    with pytest.raises(SystemExit):
        repeated.parse_args([*argv, "--vllm-repo", "/vllm"])


def test_profile_cursor_reads_only_new_jsonl_records(tmp_path: Path):
    path = tmp_path / "profile.jsonl"
    path.write_text(json.dumps({"phase": "first"}) + "\n", encoding="utf-8")

    events, offset = repeated.load_profile_events_since(path, 0)
    assert events == [{"phase": "first"}]

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"phase": "second"}) + "\n")
    events, next_offset = repeated.load_profile_events_since(path, offset)

    assert events == [{"phase": "second"}]
    assert next_offset > offset


def test_profile_cursor_defers_partial_record_and_handles_truncate(tmp_path: Path):
    path = tmp_path / "profile.jsonl"
    complete = json.dumps({"phase": "first"}) + "\n"
    path.write_text(complete + '{"phase": "part', encoding="utf-8")

    events, offset = repeated.load_profile_events_since(path, 0)
    assert events == [{"phase": "first"}]
    assert offset == len(complete.encode())

    with path.open("a", encoding="utf-8") as handle:
        handle.write('ial"}\n')
    events, old_offset = repeated.load_profile_events_since(path, offset)
    assert events == [{"phase": "partial"}]

    path.write_text(json.dumps({"phase": "replacement"}) + "\n", encoding="utf-8")
    events, new_offset = repeated.load_profile_events_since(path, old_offset)
    assert events == [{"phase": "replacement"}]
    assert new_offset < old_offset


def test_evidence_collection_errors_fail_run_closed():
    summary = {"ok": True}

    repeated.record_evidence_errors(summary, ["profile: synthetic failure"])

    assert summary["ok"] is False
    assert summary["evidence_collection_errors"] == ["profile: synthetic failure"]


def test_repeated_sleep_preserves_virtualenv_bin_on_path(tmp_path: Path, monkeypatch):
    base_python = tmp_path / "base" / "python3"
    base_python.parent.mkdir()
    base_python.touch()
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.symlink_to(base_python)
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", str(venv_bin), "/bin"]))

    observed = repeated.prepend_python_bin_to_path(venv_python)

    path_entries = os.environ["PATH"].split(os.pathsep)
    assert observed == venv_bin.absolute()
    assert path_entries[0] == str(venv_bin.absolute())
    assert path_entries.count(str(venv_bin.absolute())) == 1


def test_release_assertions_distinguish_logical_and_physical_reclaim():
    args = argparse.Namespace(
        expect_release=True,
        expect_reuse=False,
        min_worker_rss_reclaim_bytes=512,
    )
    step = {
        "cpu_backup_release_delta_bytes": 1024,
        "wake_allocator_cpu_backup_host_cache_flush_errors": 0,
        "pre_wake_coordinator_released_bytes_total": 0,
        "post_wake_coordinator_released_bytes_total": 1024,
        "pre_wake_process_tree_rss_bytes": 4096,
        "post_wake_process_tree_rss_bytes": 3072,
        "pre_wake_host_memavailable_bytes": 8192,
        "post_wake_host_memavailable_bytes": 9216,
        "output_matches_reference": True,
    }
    coordinator_stats = {
        "requested_release_bytes_total": 1024,
        "released_bytes_total": 1024,
        "pending_release_bytes": 0,
    }
    assert repeated.validate_results(args, [step], coordinator_stats) == []

    step["post_wake_process_tree_rss_bytes"] = 4000
    failures = repeated.validate_results(args, [step], coordinator_stats)
    assert "below required" in failures[0]

    args.expect_release = False
    assert "expected no release" in repeated.validate_results(args, [step])[0]


def test_release_assertions_reject_unattributed_sleep_counter_delta():
    args = argparse.Namespace(
        expect_release=True,
        expect_reuse=False,
        min_worker_rss_reclaim_bytes=512,
    )
    step = {
        "cpu_backup_release_delta_bytes": 1024,
        "sleep_allocator_cpu_backup_release_delta_bytes": 1024,
        "sleep_allocator_cpu_backup_host_cache_flush_errors": 0,
        "pre_sleep_process_tree_rss_bytes": 4096,
        "post_sleep_process_tree_rss_bytes": 3072,
        "pre_sleep_host_memavailable_bytes": 8192,
        "post_sleep_host_memavailable_bytes": 9216,
        "output_matches_reference": True,
    }
    coordinator = {
        "requested_release_bytes_total": 1024,
        "released_bytes_total": 1024,
        "pending_release_bytes": 0,
    }

    failures = repeated.validate_results(args, [step], coordinator)
    assert any("positive coordinator release delta" in failure for failure in failures)


def test_wake_reclaim_delta_uses_boundary_coordinator_counters():
    step = {
        "pre_wake_coordinator_released_bytes_total": 1024,
        "post_wake_coordinator_released_bytes_total": 4096,
    }

    assert repeated.wake_reclaim_delta(step) == 3072


def test_release_assertion_requires_flush_and_settled_protocol_evidence():
    args = argparse.Namespace(
        expect_release=True,
        expect_reuse=False,
        min_worker_rss_reclaim_bytes=0,
    )
    step = {
        "cpu_backup_release_delta_bytes": 1024,
        "wake_allocator_cpu_backup_host_cache_flush_errors": 1,
        "output_matches_reference": False,
    }
    stats = {
        "requested_release_bytes_total": 2048,
        "released_bytes_total": 1024,
        "pending_release_bytes": 1024,
    }

    failures = repeated.validate_results(args, [step], stats)

    assert any("output changed" in failure for failure in failures)
    assert any("flush reported" in failure for failure in failures)
    assert any("did not settle" in failure for failure in failures)


def test_release_assertion_accepts_whole_block_over_release():
    args = argparse.Namespace(
        expect_release=True,
        expect_reuse=False,
        min_worker_rss_reclaim_bytes=0,
    )
    step = {
        "cpu_backup_release_delta_bytes": 2048,
        "wake_allocator_cpu_backup_host_cache_flush_errors": 0,
        "output_matches_reference": True,
    }
    over_release = {
        "requested_release_bytes_total": 1024,
        "released_bytes_total": 2048,
        "pending_release_bytes": 0,
    }
    assert repeated.validate_results(args, [step], over_release) == []

    under_release = {
        "requested_release_bytes_total": 2048,
        "released_bytes_total": 1024,
        "pending_release_bytes": 1024,
    }
    failures = repeated.validate_results(args, [step], under_release)
    assert any("did not settle" in failure for failure in failures)


def test_wait_for_stats_accepts_whole_block_over_release(monkeypatch):
    stats = {
        "requested_release_bytes_total": 1024,
        "released_bytes_total": 2048,
        "pending_release_bytes": 0,
    }
    calls = 0

    def fetch(*_args):
        nonlocal calls
        calls += 1
        return stats

    monkeypatch.setattr(repeated, "fetch_run_coordinator_stats", fetch)

    result = repeated.wait_for_run_coordinator_stats(
        "http://unused",
        "run-a",
        timeout_s=1.0,
        expect_release=True,
    )

    assert result == stats
    assert calls == 1


def test_fetch_run_coordinator_stats_filters_unrelated_clients(monkeypatch):
    payload = {
        "clients": {
            "run-a:model-1-1-1": {
                "requested_release_bytes_total": 1024,
                "released_bytes_total": 1024,
                "pending_release_bytes": 0,
            },
            "other:model-2-2-2": {
                "requested_release_bytes_total": 4096,
                "released_bytes_total": 0,
                "pending_release_bytes": 4096,
            },
        }
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(payload).encode()

    class Opener:
        def open(self, _request, timeout):
            assert timeout == 0.5
            return Response()

    monkeypatch.setattr(
        repeated.urllib.request,
        "build_opener",
        lambda *_args: Opener(),
    )

    stats = repeated.fetch_run_coordinator_stats("http://127.0.0.1:9000", "run-a", 0.5)

    assert stats["client_count"] == 1
    assert stats["requested_release_bytes_total"] == 1024
    assert stats["released_bytes_total"] == 1024
    assert stats["pending_release_bytes"] == 0


def test_git_metadata_distinguishes_untracked_artifacts(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "test"], check=True)
    (tmp_path / "raw.json").write_text("{}\n", encoding="utf-8")

    metadata = repeated.git_metadata(tmp_path)

    assert metadata["git_dirty"] is True
    assert metadata["git_tracked_dirty"] is False

    tracked.write_text("modified\n", encoding="utf-8")
    assert repeated.git_metadata(tmp_path)["git_tracked_dirty"] is True


def test_release_delta_can_be_observed_on_wake_or_later_sleep():
    last_seen: dict[str, int] = {}
    wake_step: dict[str, int] = {}
    repeated.record_release_delta(
        wake_step,
        "wake_allocator",
        {"cpu_backup_release_bytes": 1024},
        "model-a",
        last_seen,
    )
    assert wake_step["cpu_backup_release_delta_bytes"] == 1024

    sleep_step: dict[str, int] = {}
    repeated.record_release_delta(
        sleep_step,
        "sleep_allocator",
        {"cpu_backup_release_bytes": 4096},
        "model-a",
        last_seen,
    )
    assert sleep_step["cpu_backup_release_delta_bytes"] == 3072


def test_reuse_assertion_requires_reused_bytes_and_zero_d2h():
    args = argparse.Namespace(
        expect_release=False,
        expect_reuse=True,
        min_worker_rss_reclaim_bytes=0,
    )
    step = {
        "cpu_backup_release_delta_bytes": 0,
        "sleep_allocator_cpu_backup_reused_bytes": 4096,
        "sleep_allocator_copy_d2h_s": 0.0,
    }
    assert repeated.validate_results(args, [step]) == []

    step["sleep_allocator_copy_d2h_s"] = 0.1
    assert "expected backup reuse" in repeated.validate_results(args, [step])[0]

    step["sleep_allocator_copy_d2h_s"] = 0.0
    stats = {
        "client_count": 1,
        "requested_release_bytes_total": 1,
        "released_bytes_total": 1,
        "pending_release_bytes": 0,
    }
    failures = repeated.validate_results(args, [step], stats)
    assert any("no-pressure control" in failure for failure in failures)

    stats.update(
        client_count=0,
        requested_release_bytes_total=0,
        released_bytes_total=0,
    )
    failures = repeated.validate_results(args, [step], stats)
    assert any("no run-local" in failure for failure in failures)
