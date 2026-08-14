"""Deterministic summaries and figures for request-driven switching."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from llm_switch_bench.plotting.style import (
    apply_paper_style,
    save_figure,
    system_color,
    system_marker,
)
from llm_switch_bench.publication import (
    default_results_root,
    prepare_family,
    read_json,
    write_family_metadata,
    write_json,
    write_result_readme,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def request_rows(family: Path, raw_dir: str) -> list[dict[str, Any]]:
    jsonl = family / "raw" / raw_dir / "e2e-alternating.jsonl"
    provenance = family / "provenance.json"
    local_rerun = provenance.is_file() and read_json(provenance).get("status") == "local-rerun"
    return read_jsonl(jsonl) if local_rerun else read_json(jsonl.with_suffix(".json"))


E2E_LIMITATION = (
    "The historical E2E producer did not runtime-bind the controller/engine commits, "
    "dirty states, executable import paths, configuration hash, or model revision. "
    "These rows are a historical local observation, not an exact fresh-checkout "
    "reproduction of the executing services."
)
RESULT_README = """# Request-driven switch

Question: what completion latency was observed for the frozen 20-request alternating-model schedule?

- Configuration: [`config/workload.json`](config/workload.json)
- Raw evidence: Proposed and llama-swap JSONL rows plus sibling runtime manifests under [`raw/`](raw/)
- Summary: [`summary.json`](summary.json)
- Figure: [`figures/request-timeline.pdf`](figures/request-timeline.pdf) ([PNG](figures/request-timeline.png))
- Method and limitations: [`../../docs/experiments/request-driven-switch/README.md`](../../docs/experiments/request-driven-switch/README.md)

The validator binds every supplied dispatch field to the frozen trace and requires 20 unique strict-success rows per system plus raw-to-summary equality. The 2026-08-13 rerun retains runtime repository, configuration, executable, model, workload, and environment provenance in each sibling run manifest.
"""


def strict_request_success(row: dict[str, Any]) -> bool:
    status = row.get("status")
    return (
        isinstance(status, int)
        and not isinstance(status, bool)
        and 200 <= status < 300
        and row.get("error") in (None, "")
        and row.get("stream_done") is True
        and row.get("semantic_ttft_ms") is not None
        and bool(str(row.get("output_text", "")).strip())
    )


def summary(family_dir: Path | None = None) -> dict[str, dict[str, float | int]]:
    family = family_dir or default_results_root() / "request-driven-switch"
    result: dict[str, dict[str, float | int]] = {}
    for system, raw_dir in (("Proposed", "proposed"), ("llama-swap", "llama-swap")):
        rows = request_rows(family, raw_dir)
        latencies = [float(row["completion_latency_ms"]) / 1000 for row in rows]
        result[system] = {
            "requests": len(rows),
            "failed": sum(not strict_request_success(row) for row in rows),
            "median_s": statistics.median(latencies),
            "min_s": min(latencies),
            "max_s": max(latencies),
        }
    return result


def write_figure(family_dir: Path) -> None:
    apply_paper_style()
    fig, axis = plt.subplots(figsize=(3.4, 2.2))
    for system, raw_dir, linestyle in (
        ("Proposed", "proposed", "-"),
        ("llama-swap", "llama-swap", "--"),
    ):
        rows = request_rows(family_dir, raw_dir)
        axis.plot(
            range(1, len(rows) + 1),
            [float(row["completion_latency_ms"]) / 1000 for row in rows],
            label=system,
            color=system_color(system),
            marker=system_marker(system),
            linestyle=linestyle,
            linewidth=1,
            markersize=3,
        )
    axis.set_xlabel("Request sequence number")
    axis.set_ylabel("Completion latency (s)")
    axis.set_yscale("log")
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, family_dir / "figures" / "request-timeline")


def build(results_root: Path | None = None) -> None:
    family = (results_root or default_results_root()) / "request-driven-switch"
    prepare_family(family)
    write_json(family / "summary.json", {"e2e": summary(family)})
    write_figure(family)
    write_result_readme(family, RESULT_README)
    write_family_metadata(
        "request-driven-switch",
        family,
        config=[
            "config/workload.json",
            "../../configs/traces/request-switch-alternating.jsonl",
        ],
        validation={"systems": 2, "requests_per_system": 20, "strict_failures": 0},
        extra=(
            {}
            if (family / "provenance.json").is_file()
            and read_json(family / "provenance.json").get("status") == "local-rerun"
            else {"historical_provenance_limitation": E2E_LIMITATION}
        ),
    )
