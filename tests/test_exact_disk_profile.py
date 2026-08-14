from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_switch_bench.experiments.exact_disk.evidence import (
    ExactDiskRequirements,
    build_curated_artifacts,
    parse_exact_disk_profile,
    validate_exact_disk_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_raw_run(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "run" / "raw"
    raw_dir.mkdir(parents=True)
    write_jsonl(
        raw_dir / "exact_disk_profile.jsonl",
        [
            {
                "phase": "exact_disk_spill",
                "disk_spill_bytes": 4096,
                "disk_spill_s": 0.25,
            },
            {
                "phase": "exact_disk_restore",
                "disk_read_bytes": 4096,
                "disk_read_s": 0.125,
                "source_medium": "disk",
                "fallback": False,
            },
        ],
    )
    write_jsonl(
        raw_dir / "resources.jsonl",
        [
            {
                "elapsed_s": 0.0,
                "worker_pid": 123,
                "worker_rss_bytes": 3000,
                "mem_available_bytes": 6000,
                "disk_footprint_bytes": 0,
            },
            {
                "elapsed_s": 1.0,
                "worker_pid": 123,
                "worker_rss_bytes": 3000,
                "mem_available_bytes": 6000,
                "disk_footprint_bytes": 4096,
            },
            {
                "elapsed_s": 2.0,
                "worker_pid": 123,
                "worker_rss_bytes": 2000,
                "mem_available_bytes": 7000,
                "disk_footprint_bytes": 4096,
            },
        ],
    )
    (raw_dir / "output_observation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "before": {"token_ids": [1, 2], "text": "same"},
                "after": {"token_ids": [1, 2], "text": "same"},
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_tier": "local_raw",
                "command": ["synthetic"],
                "command_return_code": 0,
                "model": {"name": "model-a", "path": "/models/a"},
                "backup_root": "/tmp/backups",
            }
        ),
        encoding="utf-8",
    )
    evidence = {}
    for name in (
        "exact_disk_profile.jsonl",
        "resources.jsonl",
        "output_observation.json",
        "run.json",
    ):
        path = raw_dir / name
        evidence[name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    (raw_dir / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_tier": "local_raw",
                "files": evidence,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return raw_dir


def test_profile_parser_collects_exact_disk_metrics_and_ignores_other_events(
    tmp_path: Path,
):
    profile = tmp_path / "profile.jsonl"
    write_jsonl(
        profile,
        [
            {"phase": "allocator_sleep", "latency_s": 9.0},
            {
                "phase": "exact_disk_spill",
                "disk_spill_bytes": 1024,
                "disk_spill_s": 0.1,
            },
            {
                "phase": "exact_disk_spill",
                "disk_spill_bytes": 2048,
                "disk_spill_s": 0.2,
            },
            {
                "phase": "exact_disk_restore",
                "disk_read_bytes": 3072,
                "disk_read_s": 0.15,
                "source_medium": "disk",
                "fallback": False,
            },
        ],
    )

    summary = parse_exact_disk_profile(profile)

    assert summary == {
        "profile_event_count": 4,
        "exact_disk_event_count": 3,
        "disk_spill_bytes": 3072,
        "disk_spill_s": pytest.approx(0.3),
        "disk_read_bytes": 3072,
        "disk_read_s": pytest.approx(0.15),
        "source_media": ["disk"],
        "fallback_count": 0,
        "fallback_reasons": [],
    }


@pytest.mark.parametrize(
    "row,match",
    [
        ({"disk_spill_bytes": -1, "disk_spill_s": 0.1}, "non-negative"),
        ({"disk_spill_bytes": 1}, "must appear together"),
        (
            {"disk_read_bytes": 1, "disk_read_s": 0.1, "fallback": False},
            "source_medium",
        ),
        (
            {
                "disk_read_bytes": 1,
                "disk_read_s": 0.1,
                "source_medium": "disk",
                "fallback": "false",
            },
            "boolean",
        ),
    ],
)
def test_profile_parser_rejects_invalid_exact_disk_schema(tmp_path: Path, row: dict, match: str):
    profile = tmp_path / "profile.jsonl"
    write_jsonl(profile, [row])

    with pytest.raises(ValueError, match=match):
        parse_exact_disk_profile(profile)


def test_profile_parser_rejects_malformed_jsonl(tmp_path: Path):
    profile = tmp_path / "profile.jsonl"
    profile.write_text('{"disk_spill_bytes": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        parse_exact_disk_profile(profile)


def test_collect_cli_builds_curated_artifacts_from_existing_raw_run(tmp_path: Path):
    raw_dir = make_raw_run(tmp_path)
    curated_dir = raw_dir.parent / "curated"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_switch_bench.experiments.exact_disk.collect",
            "--raw-dir",
            str(raw_dir),
            "--curated-dir",
            str(curated_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["summary"] == str(curated_dir / "summary.json")
    assert (curated_dir / "assertions.json").exists()


def test_curated_builder_keeps_raw_and_curated_evidence_distinct(tmp_path: Path):
    raw_dir = make_raw_run(tmp_path)
    curated_dir = raw_dir.parent / "curated"

    result = build_curated_artifacts(raw_dir, curated_dir)

    summary = result["summary"]
    assertions = result["assertions"]
    assert summary["evidence_tier"] == "local_curated"
    assert summary["raw_evidence"]["evidence_tier"] == "local_raw"
    assert summary["raw_evidence"]["relative_dir"] == "../raw"
    assert summary["profile"]["disk_spill_bytes"] == 4096
    assert summary["profile"]["disk_read_bytes"] == 4096
    assert summary["profile"]["source_media"] == ["disk"]
    assert summary["resources"]["worker_rss_peak_bytes"] == 3000
    assert summary["resources"]["mem_available_min_bytes"] == 6000
    assert summary["resources"]["disk_footprint_peak_bytes"] == 4096
    assert summary["output_equality"]["output_equal"] is True
    assert assertions["ok"] is True
    assert assertions["failures"] == []
    assert (curated_dir / "summary.json").exists()
    assert (curated_dir / "assertions.json").exists()


def test_assertions_fail_closed_on_fallback_wrong_medium_and_output_mismatch():
    summary = {
        "run": {"command_return_code": 0},
        "profile": {
            "exact_disk_event_count": 2,
            "disk_spill_bytes": 4096,
            "disk_read_bytes": 4096,
            "source_media": ["cpu"],
            "fallback_count": 1,
            "fallback_reasons": ["disk unavailable"],
        },
        "resources": {
            "sample_count": 2,
            "worker_rss_sample_count": 2,
            "worker_rss_first_bytes": 1000,
            "worker_rss_peak_bytes": 1000,
            "worker_rss_last_bytes": 2000,
            "mem_available_sample_count": 2,
            "mem_available_first_bytes": 8000,
            "mem_available_min_bytes": 7000,
            "mem_available_last_bytes": 7000,
            "disk_footprint_sample_count": 1,
            "disk_footprint_peak_bytes": 4096,
        },
        "output_equality": {"available": True, "output_equal": False},
    }

    failures = validate_exact_disk_summary(summary, ExactDiskRequirements())

    assert any("source medium" in failure for failure in failures)
    assert any("fallback" in failure for failure in failures)
    assert any("worker RSS did not decrease" in failure for failure in failures)
    assert any("output" in failure for failure in failures)


def test_assertions_require_os_and_disk_footprint_evidence():
    summary = {
        "run": {"command_return_code": 0},
        "profile": {
            "exact_disk_event_count": 2,
            "disk_spill_bytes": 1,
            "disk_read_bytes": 1,
            "source_media": ["disk"],
            "fallback_count": 0,
        },
        "resources": {
            "sample_count": 0,
            "worker_rss_sample_count": 0,
            "mem_available_sample_count": 0,
            "disk_footprint_sample_count": 0,
            "disk_footprint_peak_bytes": 0,
        },
        "output_equality": {"available": True, "output_equal": True},
    }

    failures = validate_exact_disk_summary(summary, ExactDiskRequirements())

    assert any("worker RSS" in failure for failure in failures)
    assert any("MemAvailable" in failure for failure in failures)
    assert any("disk footprint" in failure for failure in failures)


def test_curated_builder_rejects_tampered_raw_evidence(tmp_path: Path):
    raw_dir = make_raw_run(tmp_path)
    with (raw_dir / "exact_disk_profile.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        build_curated_artifacts(raw_dir, raw_dir.parent / "curated")


def test_curated_builder_requires_sibling_raw_and_curated_directories(tmp_path: Path):
    raw_dir = make_raw_run(tmp_path)

    with pytest.raises(ValueError, match="sibling"):
        build_curated_artifacts(raw_dir, raw_dir / "curated")
