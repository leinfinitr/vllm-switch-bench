from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_ROOT = (ROOT / "runtime" / "exact-disk-backups").resolve()


def make_fake_benchmark(tmp_path: Path) -> Path:
    path = tmp_path / "fake_benchmark.py"
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

# Keep the process alive long enough for the sampler to observe both the
# pre-demotion high-RSS/low-MemAvailable state and the post-demotion state.
hold = bytearray(64 * 1024 * 1024)
time.sleep(0.05)
profile = Path(os.environ["VLLM_SLEEP_PROFILE_PATH"])
profile.write_text(
    json.dumps({
        "phase": "exact_disk_spill",
        "disk_spill_bytes": 4,
        "disk_spill_s": 0.01,
    }) + "\\n" +
    json.dumps({
        "phase": "exact_disk_restore",
        "disk_read_bytes": 4,
        "disk_read_s": 0.01,
        "source_medium": "disk",
        "fallback": False,
    }) + "\\n",
    encoding="utf-8",
)
output = Path(os.environ["LLM_SWITCH_BENCH_OUTPUT_OBSERVATION"])
assert Path(os.environ["LLM_SWITCH_BENCH_OUT_DIR"]).is_absolute()
output.write_text(json.dumps({
    "schema_version": 1,
    "before": {"token_ids": [1], "text": "ok"},
    "after": {"token_ids": [1], "text": "ok"},
}), encoding="utf-8")
backup = Path(os.environ["VLLM_EXACT_DISK_BACKUP_DIR"])
assert os.environ["VLLM_EXACT_DISK_BACKUP_ENABLED"] == "1"
assert os.environ["VLLM_CPU_BACKUP_DISK_DIR"] == str(backup)
backup.mkdir(parents=True, exist_ok=True)
(backup / "payload.ready").write_bytes(b"data")
del hold
time.sleep(0.05)
print("synthetic benchmark complete")
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def run_script(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_switch_bench.experiments.exact_disk.run", *args],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )


def test_runner_kills_same_process_group_descendants_after_leader_exit(tmp_path: Path):
    output = tmp_path / "orphan-run"
    fixture = ROOT / "tests" / "fixtures" / "spawn_orphan.py"

    proc = run_script(
        "--model",
        "model-a=/models/a",
        "--backup-root",
        str(tmp_path / "orphan-backup"),
        "--out-dir",
        str(output),
        "--",
        sys.executable,
        str(fixture),
    )

    assert proc.returncode == 2
    child_pid = int((output / "raw" / "command.stdout.log").read_text().strip())
    assert not Path(f"/proc/{child_pid}").exists()


def test_dry_run_uses_repo_local_default_backup_root(tmp_path: Path):
    output = tmp_path / "run"

    proc = run_script(
        "--model",
        "model-a=/models/a",
        "--out-dir",
        str(output),
        "--allow-nonempty-backup-root",
        "--dry-run",
        "--",
        sys.executable,
        "fake.py",
    )

    assert proc.returncode == 0, proc.stderr
    plan = json.loads((output / "raw" / "run.json").read_text(encoding="utf-8"))
    assert Path(plan["backup_root"]) == DEFAULT_BACKUP_ROOT
    assert plan["model"] == {"name": "model-a", "path": "/models/a"}
    assert plan["evidence_tier"] == "local_raw"
    assert not (output / "curated" / "summary.json").exists()


def test_dry_run_records_selected_runtime_provenance(tmp_path: Path):
    output = tmp_path / "run"
    proc = run_script(
        "--model",
        "model-a=/models/a",
        "--out-dir",
        str(output),
        "--dry-run",
        "--",
        sys.executable,
        "fake.py",
        env={
            "VLLM_EXACT_DISK_BACKUP_DIRECT_IO": "1",
            "LLM_SWITCH_BENCH_VLLM_PYTHON": "/venv/bin/python",
            "LLM_SWITCH_BENCH_VLLM_IMPORT_PATH": "/repo/vllm/__init__.py",
            "LLM_SWITCH_BENCH_MODEL_REVISION": "revision-a",
            "LLM_SWITCH_BENCH_MODEL_CONFIG_SHA256": "a" * 64,
            "LLM_SWITCH_BENCH_BACKUP_FILESYSTEM": "ext4 /dev/nvme0n1",
            "LLM_SWITCH_BENCH_GPU_IDENTITY": "gpu-a",
        },
    )

    assert proc.returncode == 0, proc.stderr
    plan = json.loads((output / "raw" / "run.json").read_text(encoding="utf-8"))
    runtime = plan["environment"]["runtime"]
    assert runtime["selected_environment"]["VLLM_EXACT_DISK_BACKUP_DIRECT_IO"] == "1"
    assert runtime["vllm_python"] == "/venv/bin/python"
    assert runtime["declared_vllm_import_path"] == "/repo/vllm/__init__.py"
    assert runtime["vllm_import_path"] is None
    assert runtime["model_revision"] == "revision-a"
    assert runtime["model_config_sha256"] == "a" * 64
    assert runtime["filesystem"] == "ext4 /dev/nvme0n1"
    assert runtime["gpu"] == "gpu-a"


def test_runner_executes_model_agnostic_command_and_builds_curated_summary(
    tmp_path: Path,
):
    fake = make_fake_benchmark(tmp_path)
    output = tmp_path / "run"
    backup = tmp_path / "backup"

    proc = run_script(
        "--model",
        "model-a=/models/a",
        "--backup-root",
        str(backup),
        "--out-dir",
        str(output),
        "--sample-interval-s",
        "0.01",
        "--worker-pid",
        str(os.getpid()),
        "--",
        sys.executable,
        str(fake),
    )

    assert proc.returncode == 0, proc.stderr
    run = json.loads((output / "raw" / "run.json").read_text(encoding="utf-8"))
    assert run["command"] == [sys.executable, str(fake)]
    assert run["command_return_code"] == 0
    assert run["worker_pid"] > 0
    assert run["environment"]["benchmark_repo"]["commit"]
    assert "status_porcelain" in run["environment"]["benchmark_repo"]
    assert run["model"] == {"name": "model-a", "path": "/models/a"}
    summary = json.loads((output / "curated" / "summary.json").read_text(encoding="utf-8"))
    assertions = json.loads((output / "curated" / "assertions.json").read_text(encoding="utf-8"))
    assert summary["profile"]["disk_spill_bytes"] == 4
    assert summary["profile"]["disk_read_bytes"] == 4
    assert summary["resources"]["sample_count"] >= 1
    assert summary["resources"]["worker_rss_sample_count"] >= 1
    assert summary["resources"]["mem_available_sample_count"] >= 1
    assert summary["resources"]["disk_footprint_peak_bytes"] == 4
    assert summary["output_equality"]["output_equal"] is True
    assert assertions["ok"] is True
    assert (output / "raw" / "command.stdout.log").read_text().strip() == (
        "synthetic benchmark complete"
    )
    manifest = json.loads((output / "raw" / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence_tier"] == "local_raw"
    assert "exact_disk_profile.jsonl" in manifest["files"]
    assert "resources.jsonl" in manifest["files"]


def test_runner_preserves_failed_raw_evidence_without_curating(tmp_path: Path):
    output = tmp_path / "run"

    proc = run_script(
        "--model",
        "model-a=/models/a",
        "--backup-root",
        str(tmp_path / "backup"),
        "--out-dir",
        str(output),
        "--",
        sys.executable,
        "-c",
        "raise SystemExit(7)",
    )

    assert proc.returncode == 7
    run = json.loads((output / "raw" / "run.json").read_text(encoding="utf-8"))
    assert run["command_return_code"] == 7
    assert (output / "raw" / "evidence_manifest.json").exists()
    assert not (output / "curated").exists()


def test_runner_refuses_prepopulated_backup_root(tmp_path: Path):
    output = tmp_path / "run"
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "unrelated.bin").write_bytes(b"old")

    proc = run_script(
        "--model",
        "model-a=/models/a",
        "--backup-root",
        str(backup),
        "--out-dir",
        str(output),
        "--dry-run",
        "--",
        sys.executable,
        "fake.py",
    )

    assert proc.returncode == 2
    assert "backup root must be empty" in proc.stderr
    assert not output.exists()


def test_runner_refuses_to_overwrite_existing_output(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing", encoding="utf-8")

    proc = run_script(
        "--model",
        "model-a=/models/a",
        "--out-dir",
        str(output),
        "--dry-run",
        "--",
        sys.executable,
        "fake.py",
    )

    assert proc.returncode == 2
    assert "already exists" in proc.stderr
    assert sentinel.read_text(encoding="utf-8") == "existing"


def test_runner_requires_explicit_model_and_command(tmp_path: Path):
    missing_model = run_script("--out-dir", str(tmp_path / "a"), "--", sys.executable, "fake.py")
    assert missing_model.returncode == 2

    missing_command = run_script("--model", "model-a=/models/a", "--out-dir", str(tmp_path / "b"))
    assert missing_command.returncode == 2
