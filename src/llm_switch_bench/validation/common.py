from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_switch_bench.common.provenance import repository_root

FAMILY_NAMES = (
    "lifecycle-latency",
    "request-driven-switch",
    "backup-reuse-reclaim",
    "exact-disk",
)
RESULTS = repository_root() / "results"


def default_results_root() -> Path:
    return repository_root() / "results"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def family_files(family_dir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(family_dir))
        for path in family_dir.rglob("*")
        if path.is_file() and path.name != "metadata.json"
    )


def validate_metadata(family_dir: Path, experiment: str) -> dict[str, Any]:
    metadata = read_json(family_dir / "metadata.json")
    require(metadata.get("schema_version") == 1, f"{experiment}: schema_version must be 1")
    require(metadata.get("experiment") == experiment, f"{experiment}: metadata identity mismatch")
    require(
        metadata.get("status") == "migrated-historical-evidence",
        f"{experiment}: unsupported result status",
    )
    migration = str(metadata.get("migration", "")).lower()
    require("no new data was generated" in migration, f"{experiment}: migration disclosure missing")
    require(
        "canonical gpu rerun is not complete" in migration,
        f"{experiment}: canonical rerun disclosure missing",
    )
    declared = sorted(metadata.get("files", []))
    actual = family_files(family_dir)
    require(declared == actual, f"{experiment}: metadata file closure does not match the tree")
    require("README.md" in declared, f"{experiment}: result README is missing")
    require("summary.json" in declared, f"{experiment}: canonical summary is missing")
    configs = metadata.get("config")
    require(
        isinstance(configs, list) and len(configs) > 0,
        f"{experiment}: config declaration is missing",
    )
    assert isinstance(configs, list)
    for relative in configs:
        require(
            isinstance(relative, str) and not Path(relative).is_absolute(),
            f"{experiment}: config path must be relative",
        )
        config_path = family_dir / relative
        if ".." not in Path(relative).parts:
            require(
                config_path.is_file(),
                f"{experiment}: declared family config does not exist: {relative}",
            )
    require(
        any(path.startswith("config/") for path in declared),
        f"{experiment}: family config is missing",
    )
    return metadata


def validate_top_level_results(results_root: Path | None = None) -> None:
    root = results_root or default_results_root()
    expected = {"README.md", *FAMILY_NAMES}
    actual = {path.name for path in root.iterdir()}
    require(actual == expected, f"unexpected top-level results entries: {sorted(actual)}")
