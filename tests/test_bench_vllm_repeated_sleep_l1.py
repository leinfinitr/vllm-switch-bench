from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


repeated = load_module(
    "bench_vllm_repeated_sleep_l1_test",
    SRC / "bench_vllm_repeated_sleep_l1.py",
)
pin_compare = load_module(
    "bench_vllm_pin_compare_test",
    SRC / "bench_vllm_pin_compare.py",
)


def test_repeated_sleep_requires_explicit_model_specs():
    with pytest.raises(SystemExit):
        repeated.parse_args([])

    args = repeated.parse_args(["--models", "model-a=/models/a"])
    assert args.models == [repeated.ModelSpec(name="model-a", path="/models/a")]


@pytest.mark.parametrize(
    "argv",
    [
        ["--models", "a=/models/a", "a=/models/b"],
        ["--models", "a=/models/a", "--iterations", "0"],
        ["--models", "a=/models/a", "--post-wake-observation-s", "-1"],
        ["--models", "a=/models/a", "--min-worker-rss-reclaim-bytes", "1"],
        ["--models", "a=/models/a", "--expect-release", "--expect-reuse"],
        ["--models", "a=/models/a", "--iterations", "1", "--expect-release"],
        ["--models", "a=/models/a", "--iterations", "1", "--no-expect-release"],
        ["--models", "a=/models/a", "--iterations", "1", "--expect-reuse"],
    ],
)
def test_repeated_sleep_rejects_ambiguous_or_invalid_controls(argv):
    with pytest.raises(SystemExit):
        repeated.parse_args(argv)


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


def test_release_assertions_distinguish_logical_and_physical_reclaim():
    args = argparse.Namespace(
        expect_release=True,
        expect_reuse=False,
        min_worker_rss_reclaim_bytes=512,
    )
    step = {
        "cpu_backup_release_delta_bytes": 1024,
        "wake_allocator_cpu_backup_host_cache_flush_errors": 0,
        "pre_wake_worker_vmrss_bytes": 4096,
        "post_wake_worker_vmrss_bytes": 3072,
        "output_matches_reference": True,
    }
    coordinator_stats = {
        "requested_release_bytes_total": 1024,
        "released_bytes_total": 1024,
        "pending_release_bytes": 0,
    }
    assert repeated.validate_results(args, [step], coordinator_stats) == []

    step["post_wake_worker_vmrss_bytes"] = 4000
    failures = repeated.validate_results(args, [step], coordinator_stats)
    assert "below required" in failures[0]

    args.expect_release = False
    assert "expected no release" in repeated.validate_results(args, [step])[0]


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

    stats = repeated.fetch_run_coordinator_stats(
        "http://127.0.0.1:9000", "run-a", 0.5
    )

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
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "test"], check=True
    )
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


def test_pin_compare_parses_model_agnostic_specs():
    case = pin_compare.parse_model_case("model-a=/models/a,0.75")
    assert case.name == "model-a"
    assert case.path == "/models/a"
    assert case.gpu_memory_utilization == 0.75

    default_utilization = pin_compare.parse_model_case("model-b=/models/b")
    assert default_utilization.gpu_memory_utilization == 0.55

    with pytest.raises(argparse.ArgumentTypeError):
        pin_compare.parse_model_case("/models/missing-name")
