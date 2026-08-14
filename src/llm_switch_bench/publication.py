"""Shared mechanics for deterministic result publication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_switch_bench.common.provenance import repository_root

SOURCE_COMMITS = {
    "vllm_collection": "1b3919d8c210af05f6ea8b29fff33fb8d07e6c1d",
    "vllm_upstream_baseline": "0decac0d96c42b49572498019f0a0e3600f50398",
    "vllm_stock_profiling": "03e5ae257135073ddddbcd1264697f24c1c62e08",
    "controller_collection": "70e29287609f8b6639fb1b68cbcb9ffe85ed5273",
    "benchmark_collection": "9ad35876ba1b7921f8e1547698a1a8412709078e",
    "benchmark_release_tag": "v0.1.8",
    "SwapServeLLM": "69f8aec0b11e49124f70754dc5149c36fd8327a5",
    "llama-swap": "c6adf57df1ac2e3dff2402dbb479cd5a133b6afe",
}
MIGRATION_NOTE = (
    "Migrated from tracked v0.1.8 evidence; no new data was generated during this "
    "refactor. The canonical GPU rerun is not complete."
)


def default_results_root() -> Path:
    return repository_root() / "results"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def family_files(family_dir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(family_dir))
        for path in family_dir.rglob("*")
        if path.is_file() and path.name != "metadata.json"
    )


def prepare_family(family_dir: Path) -> None:
    """Create output directories without touching retained measurement inputs."""

    family_dir.mkdir(parents=True, exist_ok=True)
    (family_dir / "figures").mkdir(parents=True, exist_ok=True)


def write_result_readme(family_dir: Path, text: str) -> None:
    family_dir.mkdir(parents=True, exist_ok=True)
    (family_dir / "README.md").write_text(text, encoding="utf-8")


def write_family_metadata(
    family: str,
    family_dir: Path,
    *,
    config: list[str],
    validation: dict[str, Any],
    extra: dict[str, Any] | None = None,
    status: str = "migrated-historical-evidence",
    collected_at: str = "2026-08-04",
    migration: str = MIGRATION_NOTE,
    source_commits: dict[str, str] | None = None,
) -> None:
    provenance_path = family_dir / "provenance.json"
    if provenance_path.is_file():
        provenance = read_json(provenance_path)
        status = str(provenance["status"])
        collected_at = str(provenance["collected_at"])
        migration = str(provenance["note"])
        source_commits = dict(provenance.get("source_commits", {}))
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "experiment": family,
        "status": status,
        "collected_at": collected_at,
        "provenance_note": migration,
        "source_commits": SOURCE_COMMITS if source_commits is None else source_commits,
        "config": config,
        "validation": validation,
        "files": family_files(family_dir),
    }
    if extra:
        metadata.update(extra)
    write_json(family_dir / "metadata.json", metadata)
