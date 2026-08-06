from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_switch_bench.common.provenance import repository_root

ROOT = repository_root()
RESULTS = ROOT / "results"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_metadata(family_dir: Path, family: str) -> dict[str, Any]:
    meta = read_json(family_dir / "metadata.json")
    require(meta.get("schema_version") == 1, f"{family}: metadata schema_version must be 1")
    require(meta.get("family") == family, f"{family}: metadata family mismatch")
    required = {"README.md", "summary.json"}
    tracked = set(meta.get("tracked_files", []))
    missing = required - tracked
    require(not missing, f"{family}: metadata missing tracked files {sorted(missing)}")
    actual = sorted(
        str(p.relative_to(family_dir))
        for p in family_dir.rglob("*")
        if p.is_file() and p.name != "metadata.json"
    )
    require(sorted(tracked) == actual, f"{family}: metadata tracked_files does not match files")
    return meta


def validate_top_level_results() -> None:
    expected = {
        "README.md",
        "lifecycle-latency",
        "request-driven-switch",
        "backup-reuse-reclaim",
        "exact-disk",
    }
    actual = {p.name for p in RESULTS.iterdir()}
    require(actual == expected, f"unexpected top-level results entries: {sorted(actual)}")
