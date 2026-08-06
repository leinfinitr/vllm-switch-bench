from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from llm_switch_bench.artifacts import build_all, lifecycle_summary_rows
from llm_switch_bench.validation.validate_all import main as validate_all

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FAMILIES = {"lifecycle-latency", "request-driven-switch", "backup-reuse-reclaim", "exact-disk"}


def digest_tree() -> dict[str, str]:
    return {
        str(path.relative_to(RESULTS)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(RESULTS.rglob("*"))
        if path.is_file()
    }


def test_build_all_is_deterministic_and_validates() -> None:
    build_all()
    first = digest_tree()
    build_all()
    second = digest_tree()
    assert first == second
    assert {path.name for path in RESULTS.iterdir()} == {"README.md", *FAMILIES}
    assert validate_all() == 0


def test_lifecycle_summary_preserves_v0_1_cells() -> None:
    build_all()
    family = RESULTS / "lifecycle-latency"
    data = json.loads((family / "summary.json").read_text())["lifecycle"]
    assert len(data) == 30
    assert data == lifecycle_summary_rows(family)


def test_validator_rejects_corrupted_lifecycle_sample(tmp_path: Path) -> None:
    build_all()
    source = RESULTS / "lifecycle-latency"
    target = tmp_path / "lifecycle-latency"
    shutil.copytree(source, target)
    raw = target / "raw" / "proposed" / "qwen-0.5b.json"
    data = json.loads(raw.read_text())
    data["rows"][0]["output_match"] = False
    raw.write_text(json.dumps(data), encoding="utf-8")
    from llm_switch_bench.validation.lifecycle_latency.validate import validate_family

    with pytest.raises(ValueError, match="output mismatch"):
        validate_family(target)
