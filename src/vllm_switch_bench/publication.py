"""Shared mechanics for deterministic result publication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vllm_switch_bench.common.provenance import repository_root


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
    status: str = "retained-local-observation",
    collected_at: str = "2026-08-04",
    provenance_note: str = "No additional provenance note was recorded.",
) -> None:
    provenance_path = family_dir / "provenance.json"
    if provenance_path.is_file():
        provenance = read_json(provenance_path)
        status = str(provenance["status"])
        collected_at = str(provenance["collected_at"])
        provenance_note = str(provenance["note"])

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "experiment": family,
        "status": status,
        "collected_at": collected_at,
        "provenance_note": provenance_note,
        "config": config,
        "validation": validation,
        "files": family_files(family_dir),
    }
    if extra:
        metadata.update(extra)
    write_json(family_dir / "metadata.json", metadata)
