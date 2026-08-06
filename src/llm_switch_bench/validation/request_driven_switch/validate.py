from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from llm_switch_bench.artifacts import e2e_summary
from llm_switch_bench.validation.common import RESULTS, require, validate_metadata

SYSTEM_DIRS = {"Proposed": "proposed", "llama-swap": "llama-swap"}


def row_success(row: dict) -> bool:
    return (
        row.get("error") in (None, "")
        and int(row.get("status", 0)) == 200
        and row.get("stream_done") is True
    )


def validate_family(path: Path | None = None) -> None:
    family = path or RESULTS / "request-driven-switch"
    meta = validate_metadata(family, "request-driven-switch")
    require(
        "historical local observation" in meta.get("historical_runtime_provenance_limitation", ""),
        "request: missing provenance limitation",
    )
    sequences: dict[str, list[tuple[str, str, float]]] = {}
    for system, raw_dir in SYSTEM_DIRS.items():
        rows = json.loads(
            (family / "raw" / raw_dir / "e2e-alternating.json").read_text(encoding="utf-8")
        )
        require(len(rows) == 20, f"request: {system} must have 20 requests")
        ids = [str(row["request_id"]) for row in rows]
        require(len(ids) == len(set(ids)), f"request: {system} duplicate request IDs")
        require(
            all(row_success(row) for row in rows), f"request: {system} contains failed requests"
        )
        require(
            all(
                math.isfinite(float(row["completion_latency_ms"]))
                and float(row["completion_latency_ms"]) > 0
                for row in rows
            ),
            f"request: {system} invalid latency",
        )
        sequences[system] = [
            (str(row["request_id"]), str(row["model"]), float(row["scheduled_offset_s"]))
            for row in rows
        ]
        require(
            [item[0] for item in sequences[system]] == [f"w1-{i:03d}" for i in range(20)],
            f"request: {system} unsupported historical identity sequence",
        )
    require(
        sequences["Proposed"] == sequences["llama-swap"],
        "request: systems do not share frozen request identity sequence",
    )
    summary = json.loads((family / "summary.json").read_text(encoding="utf-8"))["e2e"]
    require(set(summary) == set(SYSTEM_DIRS), "request: expected exactly two-system summary")
    require(summary == e2e_summary(family), "request: raw recomputation does not equal summary")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate request-driven-switch result semantics")
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    validate_family(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
