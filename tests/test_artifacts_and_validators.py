from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from vllm_switch_bench import check_docs, tracked_ignore
from vllm_switch_bench.artifacts import build_all
from vllm_switch_bench.experiments.lifecycle_latency.artifacts import (
    EXTERNAL_CONTRACTS,
    build as build_lifecycle,
    summary_rows as lifecycle_summary_rows,
)
from vllm_switch_bench.experiments.request_driven_switch.artifacts import summary as e2e_summary
from vllm_switch_bench.experiments.vllm_profiling.artifacts import (
    build as build_vllm_profiling,
)
from vllm_switch_bench.families import FAMILY_NAMES
from vllm_switch_bench.validation.backup_reuse_reclaim.validate import (
    validate_family as validate_backup,
)
from vllm_switch_bench.validation.exact_disk.validate import validate_family as validate_exact_disk
from vllm_switch_bench.validation.lifecycle_latency.validate import (
    validate_family as validate_lifecycle,
)
from vllm_switch_bench.validation.request_driven_switch.validate import (
    validate_family as validate_request,
)
from vllm_switch_bench.validation.vllm_profiling.validate import (
    validate_family as validate_vllm_profiling,
)
from vllm_switch_bench.validation import validate_all as validate_all_module
from vllm_switch_bench.validation.common import validate_top_level_results
from vllm_switch_bench.validation.validate_all import validate_all

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FAMILIES = set(FAMILY_NAMES)


def test_family_registry_matches_publication_tree() -> None:
    experiment_packages = {
        path.name
        for path in (ROOT / "src" / "vllm_switch_bench" / "experiments").iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }
    validation_packages = {
        path.name.replace("_", "-")
        for path in (ROOT / "src" / "vllm_switch_bench" / "validation").iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }
    result_families = {
        path.name for path in RESULTS.iterdir() if path.is_dir() and path.name != "tmp"
    }

    assert experiment_packages == {name.replace("-", "_") for name in FAMILY_NAMES}
    assert validation_packages == FAMILIES
    assert result_families == FAMILIES


@pytest.mark.parametrize(
    "entrypoint",
    [check_docs.main, tracked_ignore.main, validate_all_module.main],
)
def test_policy_entrypoints_reject_unknown_arguments(entrypoint) -> None:
    with pytest.raises(SystemExit) as error:
        entrypoint(["--definitely-invalid"])
    assert error.value.code == 2


def test_docs_checker_rejects_broken_local_links(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text("[missing](does-not-exist.md)\n", encoding="utf-8")

    with pytest.raises(ValueError, match="broken local link"):
        check_docs.check_markdown(path)


def test_docs_checker_does_not_prescribe_prose(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    target.touch()
    path = tmp_path / "README.md"
    path.write_text("Arbitrary wording with a [valid link](target.md).\n", encoding="utf-8")

    check_docs.check_markdown(path)


def digest_tree(root: Path = RESULTS) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "tmp" not in path.relative_to(root).parts
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
    assert {path.name for path in RESULTS.iterdir() if path.name != "tmp"} == {
        "README.md",
        *FAMILIES,
    }
    validate_all()


def test_top_level_validator_allows_reserved_local_tmp(tmp_path: Path) -> None:
    (tmp_path / "README.md").touch()
    for family in FAMILIES:
        (tmp_path / family).mkdir()
    (tmp_path / "tmp").mkdir()

    validate_top_level_results(tmp_path)


def test_top_level_validator_rejects_unreserved_entry(tmp_path: Path) -> None:
    (tmp_path / "README.md").touch()
    for family in FAMILIES:
        (tmp_path / family).mkdir()
    (tmp_path / "ad-hoc-results").mkdir()

    with pytest.raises(ValueError, match="unexpected top-level results entries"):
        validate_top_level_results(tmp_path)


def test_current_headline_aggregates_recompute_from_retained_raw() -> None:
    build_all()
    lifecycle = json.loads((RESULTS / "lifecycle-latency" / "summary.json").read_text())[
        "lifecycle"
    ]
    request = json.loads((RESULTS / "request-driven-switch" / "summary.json").read_text())["e2e"]
    assert len(lifecycle) == 30
    assert lifecycle == lifecycle_summary_rows(RESULTS / "lifecycle-latency")
    assert request == e2e_summary(RESULTS / "request-driven-switch")
    assert all(item["n"] == 5 for item in lifecycle)
    assert all(item["failed"] == 0 and item["requests"] == 20 for item in request.values())


def test_external_assets_are_contracts_not_tracked_binaries() -> None:
    build_all()
    metadata = json.loads((RESULTS / "lifecycle-latency" / "metadata.json").read_text())
    artifacts = metadata["external_artifacts"]
    for name in EXTERNAL_CONTRACTS:
        item = artifacts[name]
        assert len(item["sha256"]) == 64 and item["size_bytes"] > 0
    image = artifacts["SwapServeLLM-container-image"]
    assert image["id"] and image["digest"].startswith("sha256:")
    checkpoint = artifacts["SwapServeLLM-cuda-checkpoint"]
    assert len(checkpoint["sha256"]) == 64 and checkpoint["size_bytes"] > 0
    largest = max(
        path.stat().st_size
        for path in RESULTS.rglob("*")
        if path.is_file() and "tmp" not in path.relative_to(RESULTS).parts
    )
    assert largest < 1_000_000


def test_builder_failure_does_not_delete_family_raw_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = RESULTS / "lifecycle-latency" / "raw" / "vllm-switch" / "qwen-0.5b.json"
    before = raw.read_bytes()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic plot failure")

    monkeypatch.setattr(
        "vllm_switch_bench.experiments.lifecycle_latency.artifacts.write_figure", fail
    )

    with pytest.raises(RuntimeError, match="synthetic plot failure"):
        build_lifecycle()
    assert raw.read_bytes() == before


def test_vllm_profiling_builder_failure_does_not_delete_raw_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = RESULTS / "vllm-profiling" / "raw" / "profile-samples.json"
    before = raw.read_bytes()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic profile plot failure")

    monkeypatch.setattr(
        "vllm_switch_bench.experiments.vllm_profiling.artifacts.write_artifacts", fail
    )
    with pytest.raises(RuntimeError, match="synthetic profile plot failure"):
        build_vllm_profiling()
    assert raw.read_bytes() == before


def test_vllm_profiling_validator_rejects_non_closing_phase_accounting(
    tmp_path: Path,
) -> None:
    target = copy_family(tmp_path, "vllm-profiling")
    raw = target / "raw" / "profile-samples.json"
    data = json.loads(raw.read_text())
    data["samples"][0]["wake_phases_s"]["Process + engine startup"] -= 1
    raw.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="wake phase accounting"):
        validate_vllm_profiling(target)


def test_lifecycle_validator_rejects_output_mismatch(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "lifecycle-latency")
    raw = target / "raw" / "vllm-switch" / "qwen-0.5b.json"
    data = json.loads(raw.read_text())
    data["rows"][0]["output_match"] = False
    raw.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="output mismatch"):
        validate_lifecycle(target)


def test_lifecycle_validator_rejects_non_positive_raw_phase(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "lifecycle-latency")
    raw = target / "raw" / "vllm-switch" / "qwen-0.5b.json"
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
    raw = target / "raw" / "vllm-switch" / "e2e-alternating.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    rows[0]["max_tokens"] = 64
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_request(target)


def test_request_validator_rejects_semantically_empty_success(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "request-driven-switch")
    raw = target / "raw" / "llama-swap" / "e2e-alternating.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    rows[0]["output_text"] = "   "
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="strict failure"):
        validate_request(target)


def test_backup_validator_requires_material_os_reclaim(tmp_path: Path) -> None:
    target = copy_family(tmp_path, "backup-reuse-reclaim")
    raw = target / "raw" / "vllm-switch" / "reclaim.json"
    data = json.loads(raw.read_text())
    settled = next(step for step in data["steps"] if step.get("pre_wake_host_memavailable_bytes"))
    settled["post_wake_host_memavailable_bytes"] = settled["pre_wake_host_memavailable_bytes"] + 1
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
