from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from llm_switch_bench.experiments.request_driven_switch.artifacts import (
    request_rows,
    strict_request_success,
    summary as e2e_summary,
)
from llm_switch_bench.common.traces import load_manifest
from llm_switch_bench.validation.common import (
    default_results_root,
    require,
    validate_metadata,
)

SYSTEM_DIRS = {"Proposed": "proposed", "llama-swap": "llama-swap"}
IDENTITY_FIELDS = (
    "request_id",
    "endpoint",
    "model",
    "prompt_name",
    "max_tokens",
    "temperature",
    "seed",
    "stream",
    "scheduled_offset_s",
)
TIMING_FIELDS = ("completion_latency_ms", "semantic_ttft_ms", "dispatch_lag_ms")


def validate_family(path: Path | None = None) -> None:
    family = path or default_results_root() / "request-driven-switch"
    metadata = validate_metadata(family, "request-driven-switch")
    if metadata["status"] == "migrated-historical-evidence":
        limitation = metadata.get("historical_provenance_limitation")
        require(
            isinstance(limitation, str) and bool(limitation.strip()),
            "request: provenance limitation is missing",
        )

    trace_path = (
        default_results_root().parent / "configs" / "traces" / "request-switch-alternating.jsonl"
    )
    trace_rows = load_manifest(trace_path)
    require(len(trace_rows) == 20, "request: frozen trace must contain 20 requests")
    trace_by_id = {str(row["request_id"]): row for row in trace_rows}
    require(len(trace_by_id) == len(trace_rows), "request: frozen trace has duplicate IDs")

    sequences: dict[str, list[tuple[str, str, float]]] = {}
    for system, raw_dir in SYSTEM_DIRS.items():
        rows = request_rows(family, raw_dir)
        if metadata["status"] == "local-rerun":
            run = json.loads(
                (family / "raw" / raw_dir / "e2e-alternating.run.json").read_text(encoding="utf-8")
            )
            require(run.get("failed") == 0, f"request: {system} run metadata reports failures")
            benchmark = run.get("benchmark_repo", {})
            require(
                benchmark.get("commit") and benchmark.get("working_tree_sha256"),
                f"request: {system} benchmark identity is incomplete",
            )
            require(
                run.get("runtime_repositories") and run.get("runtime_files"),
                f"request: {system} runtime identity is incomplete",
            )
        require(len(rows) == len(trace_rows), f"request: {system} must have 20 requests")
        by_id = {str(row.get("request_id")): row for row in rows}
        require("None" not in by_id and len(by_id) == len(rows), f"request: {system} duplicate IDs")
        require(set(by_id) == set(trace_by_id), f"request: {system} trace/output IDs differ")
        for request_id, expected in trace_by_id.items():
            observed = by_id[request_id]
            require(
                all(observed.get(field) == expected.get(field) for field in IDENTITY_FIELDS),
                f"request: {system} identity mismatch for {request_id}",
            )
            require(strict_request_success(observed), f"request: {system} strict failure")
            for field in TIMING_FIELDS:
                value = observed.get(field)
                require(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= 0,
                    f"request: {system} invalid {field} for {request_id}",
                )
        sequences[system] = [
            (str(row["request_id"]), str(row["model"]), float(row["scheduled_offset_s"]))
            for row in rows
        ]

    require(
        sequences["Proposed"] == sequences["llama-swap"],
        "request: systems do not share the frozen identity sequence",
    )
    summary = json.loads((family / "summary.json").read_text(encoding="utf-8"))["e2e"]
    require(set(summary) == set(SYSTEM_DIRS), "request: expected exactly two systems")
    require(summary == e2e_summary(family), "request: raw recomputation differs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate request-driven-switch semantics")
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args(argv)
    validate_family(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
