from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from llm_switch_bench import check_docs, tracked_ignore
from llm_switch_bench.artifacts import (
    EXTERNAL_CONTRACTS,
    build_all,
    e2e_summary,
    lifecycle_summary_rows,
)
from llm_switch_bench.validation.backup_reuse_reclaim.validate import (
    validate_family as validate_backup,
)
from llm_switch_bench.validation.exact_disk.validate import validate_family as validate_exact_disk
from llm_switch_bench.validation.lifecycle_latency.validate import (
    validate_family as validate_lifecycle,
)
from llm_switch_bench.validation.request_driven_switch.validate import (
    validate_family as validate_request,
)
from llm_switch_bench.validation import validate_all as validate_all_module
from llm_switch_bench.validation.validate_all import validate_all

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FAMILIES = {"lifecycle-latency", "request-driven-switch", "backup-reuse-reclaim", "exact-disk"}


@pytest.mark.parametrize(
    "entrypoint",
    [check_docs.main, tracked_ignore.main, validate_all_module.main],
)
def test_policy_entrypoints_reject_unknown_arguments(entrypoint) -> None:
    with pytest.raises(SystemExit) as error:
        entrypoint(["--definitely-invalid"])
    assert error.value.code == 2


def digest_tree(root: Path = RESULTS) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def copy_family(tmp_path: Path, name: str) -> Path:
    build_all()
    target = tmp_path / name
    shutil.copytree(RESULTS / name, target)
    return target


def test_build_all_is_deterministic_and_validates() -> None:
    build_all()
    first = digest_tree()
    build_all()
    second = digest_tree()
    assert first == second
    assert {path.name for path in RESULTS.iterdir()} == {"README.md", *FAMILIES}
    validate_all()


def test_migrated_headline_aggregates_are_exactly_preserved() -> None:
    build_all()
    source = json.loads(
        __import__("subprocess").check_output(
            ["git", "show", "v0.1.8:results/release-v0.1/summary.json"],
            cwd=ROOT,
            text=True,
        )
    )
    lifecycle = json.loads((RESULTS / "lifecycle-latency" / "summary.json").read_text())[
        "lifecycle"
    ]
    request = json.loads((RESULTS / "request-driven-switch" / "summary.json").read_text())["e2e"]
    assert len(lifecycle) == 30
    assert lifecycle == source["lifecycle"] == lifecycle_summary_rows(RESULTS / "lifecycle-latency")
    assert request == source["e2e"] == e2e_summary(RESULTS / "request-driven-switch")


def test_external_assets_are_contracts_not_tracked_binaries() -> None:
    build_all()
    metadata = json.loads((RESULTS / "lifecycle-latency" / "metadata.json").read_text())
    assert metadata["external_artifacts"] == EXTERNAL_CONTRACTS
    assert all(
        len(item["sha256"]) == 64 and item["size_bytes"] > 0 for item in EXTERNAL_CONTRACTS.values()
    )
    largest = max(path.stat().st_size for path in RESULTS.rglob("*") if path.is_file())
    assert largest < 1_000_000


def test_builder_failure_does_not_delete_family_raw_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = RESULTS / "lifecycle-latency" / "raw" / "proposed" / "qwen-0.5b.json"
    before = raw.read_bytes()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic plot failure")

    monkeypatch.setattr("llm_switch_bench.artifacts.write_lifecycle_figure", fail)
    from llm_switch_bench.artifacts import build_lifecycle

    with pytest.raises(RuntimeError, match="synthetic plot failure"):
        build_lifecycle()
    assert raw.read_bytes() == before


def test_lifecycle_validator_rejects_output_mismatch(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "lifecycle-latency")
    raw = target / "raw" / "proposed" / "qwen-0.5b.json"
    data = json.loads(raw.read_text())
    data["rows"][0]["output_match"] = False
    raw.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="output mismatch"):
        validate_lifecycle(target)


def test_lifecycle_validator_rejects_non_positive_raw_phase(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "lifecycle-latency")
    raw = target / "raw" / "proposed" / "qwen-0.5b.json"
    data = json.loads(raw.read_text())
    data["rows"][0]["sleep_s"] = -1.0
    raw.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid raw phase latency"):
        validate_lifecycle(target)


def test_lifecycle_validator_rejects_non_finite_llama_phase(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "lifecycle-latency")
    raw = target / "raw" / "llama-swap" / "lifecycle.json"
    data = json.loads(raw.read_text())
    data["rows"][0]["sleep"]["state_machine_latency_s"] = "inf"
    raw.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid llama-swap phase"):
        validate_lifecycle(target)


def test_lifecycle_validator_rejects_failed_gpu_postcondition(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "lifecycle-latency")
    raw = target / "raw" / "swapserve" / "qwen-0.5b.json"
    data = json.loads(raw.read_text())
    data["rows"][0]["sleep_gpu_mib"] = 1
    raw.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="physically release"):
        validate_lifecycle(target)


def test_request_validator_rejects_changed_dispatch_semantics(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "request-driven-switch")
    raw = target / "raw" / "proposed" / "e2e-alternating.json"
    rows = json.loads(raw.read_text())
    rows[0]["max_tokens"] = 64
    raw.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_request(target)


def test_request_validator_rejects_semantically_empty_success(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "request-driven-switch")
    raw = target / "raw" / "llama-swap" / "e2e-alternating.json"
    rows = json.loads(raw.read_text())
    rows[0]["output_text"] = "   "
    raw.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="strict failure"):
        validate_request(target)


def test_backup_validator_requires_material_os_reclaim(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "backup-reuse-reclaim")
    raw = target / "raw" / "proposed" / "controller-pressure-release.json"
    data = json.loads(raw.read_text())
    data["memavailable_delta_bytes"] = 1
    data["after"]["memavailable_bytes"] = data["before"]["memavailable_bytes"] + 1
    raw.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="MemAvailable recovery is not material"):
        validate_backup(target)


def test_exact_disk_validator_authenticates_manifest_commit(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "exact-disk")
    manifest = target / "raw" / "exact-disk" / "bundle-manifest.json"
    data = json.loads(manifest.read_text())
    data["segments"][0]["size_bytes"] -= 1
    manifest.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(ValueError, match="commit marker"):
        validate_exact_disk(target)


def test_exact_disk_validator_rejects_fallback(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "exact-disk")
    profile = target / "raw" / "exact-disk" / "exact_disk_profile.jsonl"
    rows = [json.loads(line) for line in profile.read_text().splitlines()]
    rows[0]["fallback"] = True
    profile.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="fallback"):
        validate_exact_disk(target)
