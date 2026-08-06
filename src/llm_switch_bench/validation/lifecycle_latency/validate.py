from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from llm_switch_bench.artifacts import (
    EXTERNAL_CONTRACTS,
    MODELS,
    PHASES,
    SYSTEMS,
    lifecycle_summary_rows,
)
from llm_switch_bench.validation.common import RESULTS, require, validate_metadata


def validate_family(path: Path | None = None) -> None:
    family = path or RESULTS / "lifecycle-latency"
    meta = validate_metadata(family, "lifecycle-latency")
    require(
        meta.get("external_binary_contracts") == EXTERNAL_CONTRACTS,
        "lifecycle: external binary contracts mismatch",
    )
    summary = json.loads((family / "summary.json").read_text(encoding="utf-8"))["lifecycle"]
    expected_keys = {
        (model, system, phase) for model in MODELS for system in SYSTEMS for phase in PHASES
    }
    observed_keys = {(row["model"], row["system"], row["phase"]) for row in summary}
    require(observed_keys == expected_keys, "lifecycle: unexpected model/system/phase matrix")
    require(len(summary) == 30, "lifecycle: expected exactly 30 aggregate cells")
    for row in summary:
        require(row["n"] == 5, f"lifecycle: {row} does not have five samples")
        for key in ("median_s", "q1_s", "q3_s"):
            value = float(row[key])
            require(math.isfinite(value) and value > 0, f"lifecycle: invalid {key}")
    for model in MODELS:
        for system in SYSTEMS:
            raw_path = family / (
                "raw/llama-swap/lifecycle.json"
                if system == "llama-swap"
                else f"raw/{ {'Proposed': 'proposed', 'vLLM L1': 'vllm-stock', 'vLLM L2': 'vllm-l2', 'SwapServeLLM': 'swapserve'}[system] }/{model}.json"
            )
            data = json.loads(raw_path.read_text(encoding="utf-8"))
            rows = data["rows"]
            if system == "llama-swap":
                rows = [item for item in rows if item["model"] == model]
            require(len(rows) == 5, f"lifecycle: raw sample count mismatch for {system} {model}")
            require(
                all(item.get("output_match", True) is True for item in rows),
                f"lifecycle: output mismatch for {system} {model}",
            )
    recomputed = lifecycle_summary_rows(family)
    require(summary == recomputed, "lifecycle: raw recomputation does not equal summary")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate lifecycle-latency result semantics")
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    validate_family(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
