from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from llm_switch_bench.artifacts import (
    EXTERNAL_CONTRACTS,
    MODELS,
    PHASES,
    RAW_MAP,
    SYSTEMS,
    lifecycle_summary_rows,
)
from llm_switch_bench.validation.common import (
    default_results_root,
    require,
    validate_metadata,
)


def validate_family(path: Path | None = None) -> None:
    family = path or default_results_root() / "lifecycle-latency"
    metadata = validate_metadata(family, "lifecycle-latency")
    require(
        metadata.get("external_artifacts") == EXTERNAL_CONTRACTS,
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
        q1 = float(row["q1_s"])
        median = float(row["median_s"])
        q3 = float(row["q3_s"])
        require(
            all(math.isfinite(value) and value > 0 for value in (q1, median, q3)),
            "lifecycle: non-positive or non-finite aggregate",
        )
        require(q1 <= median <= q3, "lifecycle: invalid quartile ordering")

    for model in MODELS:
        for system in SYSTEMS:
            relative = (
                Path("raw/llama-swap/lifecycle.json")
                if system == "llama-swap"
                else Path("raw") / RAW_MAP[system] / f"{model}.json"
            )
            data = json.loads((family / relative).read_text(encoding="utf-8"))
            rows = data["rows"]
            if system == "llama-swap":
                rows = [item for item in rows if item["model"] == model]
            require(len(rows) == 5, f"lifecycle: raw sample count mismatch for {system} {model}")
            require(
                len({int(item["cycle"]) for item in rows}) == 5,
                f"lifecycle: duplicate cycle for {system} {model}",
            )
            if system == "llama-swap":
                require(
                    all(item.get("ok") is True for item in rows), "lifecycle: llama-swap failure"
                )
                require(
                    all(
                        item[phase].get("ok") is True
                        and float(item[phase]["state_machine_latency_s"]) > 0
                        for item in rows
                        for phase in PHASES
                    ),
                    f"lifecycle: invalid llama-swap phase for {model}",
                )
                require(
                    all(
                        int(item["sleep"]["postcondition"]["gpu_used_mib"])
                        <= int(item["sleep"]["postcondition"]["idle_threshold_mib"])
                        and not item["sleep"]["postcondition"]["running"]
                        for item in rows
                    ),
                    f"lifecycle: llama-swap physical sleep post-condition failed for {model}",
                )
            else:
                require(
                    all(item.get("output_match") is True for item in rows),
                    f"lifecycle: output mismatch for {system} {model}",
                )
                if system == "SwapServeLLM":
                    require(
                        all(float(item.get("sleep_gpu_mib", -1)) == 0 for item in rows),
                        f"lifecycle: SwapServeLLM did not physically release GPU memory for {model}",
                    )

    require(summary == lifecycle_summary_rows(family), "lifecycle: raw recomputation differs")
    csv_rows = list(csv.DictReader((family / "summary.csv").open(encoding="utf-8", newline="")))
    require(len(csv_rows) == 30, "lifecycle: summary.csv must contain 30 rows")
    normalized_csv = [
        {
            "model": row["model"],
            "system": row["system"],
            "phase": row["phase"],
            "n": int(row["n"]),
            "median_s": float(row["median_s"]),
            "q1_s": float(row["q1_s"]),
            "q3_s": float(row["q3_s"]),
        }
        for row in csv_rows
    ]
    require(normalized_csv == summary, "lifecycle: CSV and JSON summaries differ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate lifecycle-latency result semantics")
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args(argv)
    validate_family(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
