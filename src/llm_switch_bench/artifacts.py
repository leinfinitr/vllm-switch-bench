from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt

from llm_switch_bench.common.provenance import repository_root
from llm_switch_bench.plotting.style import (
    apply_paper_style,
    save_figure,
    system_color,
    system_hatch,
    system_marker,
)

FAMILY_NAMES = (
    "lifecycle-latency",
    "request-driven-switch",
    "backup-reuse-reclaim",
    "exact-disk",
)
MODELS = ("qwen-0.5b", "qwen-1.5b", "qwen-3b")
SYSTEMS = ("Proposed", "vLLM L1", "vLLM L2", "SwapServeLLM", "llama-swap")
PHASES = ("sleep", "wake")
RAW_MAP = {
    "Proposed": "proposed",
    "vLLM L1": "vllm-stock",
    "vLLM L2": "vllm-l2",
    "SwapServeLLM": "swapserve",
}
EXTERNAL_CONTRACTS = {
    "SwapServeLLM": {
        "asset_url": (
            "https://github.com/leinfinitr/llm-switch-bench/releases/download/"
            "v0.1.8/SwapServeLLM"
        ),
        "size_bytes": 54_774_096,
        "sha256": "7d463c42e3d0c965cba078d77a2abb053ba02f2a27a2303d32e5dccecffae091",
    },
    "llama-swap-profiled": {
        "asset_url": (
            "https://github.com/leinfinitr/llm-switch-bench/releases/download/"
            "v0.1.8/llama-swap-profiled"
        ),
        "size_bytes": 20_973_543,
        "sha256": "196148236fad99b32cb86c04d9297cfe0eaca68d204c920e3aeff290d04a024b",
    },
}
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
E2E_LIMITATION = (
    "The historical E2E producer did not runtime-bind the controller/engine commits, "
    "dirty states, executable import paths, configuration hash, or model revision. "
    "These rows are a historical local observation, not an exact fresh-checkout "
    "reproduction of the executing services."
)


def default_results_root() -> Path:
    return repository_root() / "results"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def quantiles(values: list[float]) -> tuple[float, float, float]:
    """Return the v0.1 sample quartiles: second, median, fourth sorted values."""

    if len(values) != 5:
        raise ValueError(f"expected five lifecycle samples, got {len(values)}")
    ordered = sorted(values)
    return ordered[1], statistics.median(ordered), ordered[3]


def lifecycle_raw_path(system: str, model: str, family_dir: Path) -> Path:
    raw = family_dir / "raw"
    if system == "llama-swap":
        return raw / "llama-swap" / "lifecycle.json"
    return raw / RAW_MAP[system] / f"{model}.json"


def lifecycle_rows(system: str, model: str, family_dir: Path) -> list[dict[str, Any]]:
    data = read_json(lifecycle_raw_path(system, model, family_dir))
    rows = data["rows"]
    if system == "llama-swap":
        rows = [row for row in rows if row["model"] == model]
    return rows


def lifecycle_summary_rows(family_dir: Path | None = None) -> list[dict[str, Any]]:
    family = family_dir or default_results_root() / "lifecycle-latency"
    summary: list[dict[str, Any]] = []
    for model in MODELS:
        for system in SYSTEMS:
            samples = lifecycle_rows(system, model, family)
            for phase in PHASES:
                key = phase if system == "llama-swap" else f"{phase}_s"
                values = [
                    float(sample[key]["state_machine_latency_s"])
                    if system == "llama-swap"
                    else float(sample[key])
                    for sample in samples
                ]
                q1, median, q3 = quantiles(values)
                summary.append(
                    {
                        "model": model,
                        "system": system,
                        "phase": phase,
                        "n": len(values),
                        "median_s": median,
                        "q1_s": q1,
                        "q3_s": q3,
                    }
                )
    return summary


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


def e2e_summary(family_dir: Path | None = None) -> dict[str, dict[str, float | int]]:
    family = family_dir or default_results_root() / "request-driven-switch"
    result: dict[str, dict[str, float | int]] = {}
    for system, raw_dir in (("Proposed", "proposed"), ("llama-swap", "llama-swap")):
        rows = read_json(family / "raw" / raw_dir / "e2e-alternating.json")
        latencies = [float(row["completion_latency_ms"]) / 1000 for row in rows]
        result[system] = {
            "requests": len(rows),
            "failed": sum(not strict_request_success(row) for row in rows),
            "median_s": statistics.median(latencies),
            "min_s": min(latencies),
            "max_s": max(latencies),
        }
    return result


def backup_summary(family_dir: Path | None = None) -> dict[str, Any]:
    family = family_dir or default_results_root() / "backup-reuse-reclaim"
    reuse: list[dict[str, Any]] = []
    for model in MODELS:
        data = read_json(family / "raw" / "proposed" / f"{model}.json")
        events = data.get("sleep_events", [])
        reuse.append(
            {
                "model": model,
                "events": len(events),
                "min_reuse_count": min(int(event.get("cpu_backup_reuse_count", 0)) for event in events),
                "min_reused_bytes": min(
                    int(event.get("cpu_backup_reused_bytes", 0)) for event in events
                ),
                "max_copy_d2h_s": max(float(event.get("copy_d2h_s", 0.0)) for event in events),
            }
        )
    reclaim = read_json(family / "raw" / "proposed" / "controller-pressure-release.json")
    before = reclaim["before"]
    after = reclaim["after"]
    return {
        "reuse": reuse,
        "reclaim": {
            "requested_bytes": int(reclaim["release_response"]["queued_bytes"]),
            "released_delta_bytes": int(
                before["pool_stats"]["total_bytes"] - after["pool_stats"]["total_bytes"]
            ),
            "pending_release_bytes": int(after["pool_stats"]["pending_release_bytes"]),
            "pending_release_request_count": int(
                after["pool_stats"]["pending_release_request_count"]
            ),
            "memavailable_delta_bytes": int(reclaim["memavailable_delta_bytes"]),
            "client_rss_delta_bytes": reclaim["client_rss_delta_bytes"],
            "flush_success": bool(reclaim["release_response"]["ok"]),
        },
    }


def exact_disk_summary(family_dir: Path | None = None) -> dict[str, Any]:
    family = family_dir or default_results_root() / "exact-disk"
    raw = family / "raw" / "exact-disk"
    events = [
        json.loads(line)
        for line in (raw / "exact_disk_profile.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = read_json(raw / "output_observation.json")
    payload = read_json(raw / "payload-hash.json")
    return {
        "disk_spill_bytes": sum(int(event.get("disk_spill_bytes", 0)) for event in events),
        "disk_released_bytes": sum(
            int(event.get("released_bytes", 0))
            for event in events
            if event.get("phase") == "exact_disk_demotion"
        ),
        "restore_reused_bytes": sum(int(event.get("disk_read_bytes", 0)) for event in events),
        "payload_bytes": int(payload["payload_size_bytes"]),
        "payload_sha256": payload["payload_sha256"],
        "output_match": output.get("before") == output.get("after"),
    }


def write_lifecycle_figure(summary: list[dict[str, Any]], family_dir: Path) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15), sharey=True)
    x = range(len(SYSTEMS))
    for index, (axis, model) in enumerate(zip(axes, MODELS, strict=True)):
        model_rows = [row for row in summary if row["model"] == model]
        for phase, offset, marker in (("sleep", -0.12, "o"), ("wake", 0.12, "^")):
            rows = [
                next(
                    row
                    for row in model_rows
                    if row["system"] == system and row["phase"] == phase
                )
                for system in SYSTEMS
            ]
            medians = [float(row["median_s"]) for row in rows]
            errors = [
                [float(row["median_s"]) - float(row["q1_s"]) for row in rows],
                [float(row["q3_s"]) - float(row["median_s"]) for row in rows],
            ]
            axis.errorbar(
                [item + offset for item in x],
                medians,
                yerr=errors,
                label=phase.capitalize(),
                marker=marker,
                linestyle="none",
                color="#222222",
                capsize=2,
                markersize=3.5,
            )
        axis.set_title(model)
        axis.set_xticks(list(x), SYSTEMS, rotation=38, ha="right")
        axis.set_yscale("log")
        if index == 0:
            axis.set_ylabel("Latency (s, log scale)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, family_dir / "figures" / "lifecycle-latency")


def write_request_figure(family_dir: Path) -> None:
    apply_paper_style()
    fig, axis = plt.subplots(figsize=(3.4, 2.2))
    for system, raw_dir, linestyle in (
        ("Proposed", "proposed", "-"),
        ("llama-swap", "llama-swap", "--"),
    ):
        rows = read_json(family_dir / "raw" / raw_dir / "e2e-alternating.json")
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


def write_backup_figure(summary: dict[str, Any], family_dir: Path) -> None:
    apply_paper_style()
    fig, axis = plt.subplots(figsize=(3.4, 2.2))
    labels = [item["model"] for item in summary["reuse"]]
    values = [item["min_reused_bytes"] / (1024**3) for item in summary["reuse"]]
    bars = axis.bar(
        labels,
        values,
        color=[system_color("Proposed")] * len(labels),
        hatch=system_hatch("vLLM L1"),
    )
    for bar, item in zip(bars, summary["reuse"], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(item["min_reuse_count"]),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axis.set_ylabel("Minimum reused backup (GiB)")
    axis.set_xlabel("Model (label above bar: allocation count)")
    fig.tight_layout()
    save_figure(fig, family_dir / "figures" / "backup-reuse")


def write_exact_disk_figure(summary: dict[str, Any], family_dir: Path) -> None:
    apply_paper_style()
    fig, axis = plt.subplots(figsize=(3.4, 2.1))
    fields = ("disk_spill_bytes", "disk_released_bytes", "restore_reused_bytes")
    axis.bar(
        ("Spill", "Release", "Restore"),
        [summary[field] / (1024**3) for field in fields],
        color=system_color("exact-disk"),
        hatch=system_hatch("exact-disk"),
    )
    axis.set_ylabel("Data volume (GiB)")
    axis.set_title("Exact-disk lifecycle evidence")
    fig.tight_layout()
    save_figure(fig, family_dir / "figures" / "exact-disk")


def family_files(family_dir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(family_dir))
        for path in family_dir.rglob("*")
        if path.is_file() and path.name != "metadata.json"
    )


def write_family_metadata(
    family: str,
    family_dir: Path,
    *,
    config: list[str],
    validation: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "experiment": family,
        "status": "migrated-historical-evidence",
        "collected_at": "2026-08-04",
        "migration": MIGRATION_NOTE,
        "source_commits": SOURCE_COMMITS,
        "config": config,
        "validation": validation,
        "files": family_files(family_dir),
    }
    if extra:
        metadata.update(extra)
    write_json(family_dir / "metadata.json", metadata)


def _remove_generated(family_dir: Path) -> None:
    for relative in ("summary.json", "summary.csv", "metadata.json", "README.md"):
        path = family_dir / relative
        if path.exists():
            path.unlink()
    figures = family_dir / "figures"
    if figures.exists():
        for path in figures.iterdir():
            if path.is_file():
                path.unlink()


def build_lifecycle(results_root: Path | None = None) -> None:
    family = (results_root or default_results_root()) / "lifecycle-latency"
    _remove_generated(family)
    summary = lifecycle_summary_rows(family)
    write_json(family / "summary.json", {"lifecycle": summary})
    with (family / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "system", "phase", "n", "median_s", "q1_s", "q3_s"),
            lineterminator="\n",
        )
        writer.writeheader()
        handle.flush()
        for row in summary:
            handle.write(
                ",".join(str(row[field]) for field in writer.fieldnames) + "\n"
            )
    write_lifecycle_figure(summary, family)
    write_family_readme(family, "lifecycle-latency")
    write_family_metadata(
        "lifecycle-latency",
        family,
        config=[],
        validation={"aggregate_cells": 30, "samples_per_cell": 5},
        extra={"external_artifacts": EXTERNAL_CONTRACTS},
    )


def build_request(results_root: Path | None = None) -> None:
    family = (results_root or default_results_root()) / "request-driven-switch"
    _remove_generated(family)
    summary = {"e2e": e2e_summary(family)}
    write_json(family / "summary.json", summary)
    write_request_figure(family)
    write_family_readme(family, "request-driven-switch")
    write_family_metadata(
        "request-driven-switch",
        family,
        config=["../../configs/traces/request-switch-alternating.jsonl"],
        validation={"systems": 2, "requests_per_system": 20, "strict_failures": 0},
        extra={"historical_provenance_limitation": E2E_LIMITATION},
    )


def build_backup(results_root: Path | None = None) -> None:
    family = (results_root or default_results_root()) / "backup-reuse-reclaim"
    _remove_generated(family)
    summary = backup_summary(family)
    write_json(family / "summary.json", summary)
    write_backup_figure(summary, family)
    write_family_readme(family, "backup-reuse-reclaim")
    write_family_metadata(
        "backup-reuse-reclaim",
        family,
        config=[],
        validation={
            "reuse_models": 3,
            "reuse_samples_per_model": 5,
            "physical_reclaim_evidence": True,
        },
    )


def build_exact_disk(results_root: Path | None = None) -> None:
    family = (results_root or default_results_root()) / "exact-disk"
    _remove_generated(family)
    summary = exact_disk_summary(family)
    write_json(family / "summary.json", summary)
    write_exact_disk_figure(summary, family)
    write_family_readme(family, "exact-disk")
    write_family_metadata(
        "exact-disk",
        family,
        config=[],
        validation={
            "payload_bytes": summary["payload_bytes"],
            "output_match": summary["output_match"],
            "runtime_checksum_retained": True,
        },
    )


_BUILDERS: dict[str, Callable[[Path | None], None]] = {
    "lifecycle-latency": build_lifecycle,
    "request-driven-switch": build_request,
    "backup-reuse-reclaim": build_backup,
    "exact-disk": build_exact_disk,
}


def write_family_readme(family_dir: Path, family: str) -> None:
    titles = {
        "lifecycle-latency": "Lifecycle Latency Result",
        "request-driven-switch": "Request-Driven Switch Result",
        "backup-reuse-reclaim": "Backup Reuse and Reclaim Result",
        "exact-disk": "Exact-Disk Result",
    }
    family_dir.mkdir(parents=True, exist_ok=True)
    (family_dir / "README.md").write_text(
        f"# {titles[family]}\n\n"
        "This is the current claim-supporting result directory. It retains the minimum raw "
        "evidence consumed by the builder, a canonical summary, run metadata, and the PDF/PNG "
        "paper figure. See the matching experiment document under "
        f"`../../docs/experiments/{family}/README.md`.\n\n"
        f"{MIGRATION_NOTE}\n\n"
        "Rebuild from the repository root with:\n\n"
        f"```bash\nuv run python -m llm_switch_bench.artifacts {family}\n"
        f"uv run python -m llm_switch_bench.validation.{family.replace('-', '_')}.validate\n"
        "```\n",
        encoding="utf-8",
    )


def build_all(results_root: Path | None = None) -> None:
    root = results_root or default_results_root()
    for builder in _BUILDERS.values():
        builder(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic current result artifacts")
    parser.add_argument("family", nargs="?", default="all", choices=("all", *FAMILY_NAMES))
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args(argv)
    if args.family == "all":
        build_all(args.results_root)
    else:
        _BUILDERS[args.family](args.results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
