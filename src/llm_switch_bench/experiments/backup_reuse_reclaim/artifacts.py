"""Deterministic summaries and figures for backup reuse and reclaim."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from llm_switch_bench.plotting.style import (
    apply_paper_style,
    save_figure,
    system_color,
    system_hatch,
)
from llm_switch_bench.publication import (
    default_results_root,
    prepare_family,
    read_json,
    write_family_metadata,
    write_json,
    write_result_readme,
)
from llm_switch_bench.experiments.backup_reuse_reclaim.run import wake_reclaim_delta

MODELS = ("qwen-0.5b", "qwen-1.5b", "qwen-3b")


def reuse_path(family: Path, model: str) -> Path:
    current = family / "raw" / "proposed" / "reuse" / f"{model}.json"
    legacy = family / "raw" / "proposed" / f"{model}.json"
    return current if current.is_file() else legacy


def reuse_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    return data.get("steps", data.get("rows", []))


def reuse_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in data.get("sleep_profile_events", data.get("sleep_events", []))
        if event.get("phase") == "allocator_sleep"
    ]


def reclaim_path(family: Path) -> Path:
    current = family / "raw" / "proposed" / "reclaim.json"
    legacy = family / "raw" / "proposed" / "controller-pressure-release.json"
    return current if current.is_file() else legacy


def reclaim_metrics(data: dict[str, Any]) -> dict[str, Any]:
    if "coordinator_stats" in data:
        release_steps = [
            step
            for step in data["steps"]
            if int(step.get("cpu_backup_release_delta_bytes", 0) or 0) > 0
        ]
        released = int(data["coordinator_stats"]["released_bytes_total"])
        reclaim_steps = [step for step in data["steps"] if wake_reclaim_delta(step) > 0]
        rss_delta = 0
        memavailable_delta = 0
        for step in reclaim_steps:
            rss_delta += int(step["post_wake_process_tree_rss_bytes"]) - int(
                step["pre_wake_process_tree_rss_bytes"]
            )
            memavailable_delta += int(step["post_wake_host_memavailable_bytes"]) - int(
                step["pre_wake_host_memavailable_bytes"]
            )
        coordinator = data["coordinator_stats"]
        return {
            "requested_bytes": int(coordinator["requested_release_bytes_total"]),
            "released_delta_bytes": released,
            "pending_release_bytes": int(coordinator["pending_release_bytes"]),
            "pending_release_request_count": sum(
                int(client.get("pending_release_bytes", 0)) > 0
                for client in coordinator["clients"].values()
            ),
            "memavailable_delta_bytes": memavailable_delta,
            "client_rss_delta_bytes": {"benchmark_process_tree": rss_delta},
            "physical_settlement_windows": len(reclaim_steps),
            "flush_success": all(
                int(value or 0) == 0
                for step in release_steps
                for key, value in step.items()
                if key.endswith("cpu_backup_host_cache_flush_errors")
            ),
        }
    before = data["before"]
    after = data["after"]
    return {
        "requested_bytes": int(data["release_response"]["queued_bytes"]),
        "released_delta_bytes": int(
            before["pool_stats"]["total_bytes"] - after["pool_stats"]["total_bytes"]
        ),
        "pending_release_bytes": int(after["pool_stats"]["pending_release_bytes"]),
        "pending_release_request_count": int(after["pool_stats"]["pending_release_request_count"]),
        "memavailable_delta_bytes": int(data["memavailable_delta_bytes"]),
        "client_rss_delta_bytes": data["client_rss_delta_bytes"],
        "flush_success": bool(data["release_response"]["ok"]),
    }


RESULT_README = """# Backup reuse and reclaim

Question: do repeated sleeps reuse clean CPU backups without another D2H copy, and does pressure reclaim complete logically and physically?

- Configuration: [`config/claims.json`](config/claims.json)
- Raw evidence: three five-event model profiles and one pressure-release observation under [`raw/`](raw/)
- Summary: [`summary.json`](summary.json)
- Figure: [`figures/backup-reuse.pdf`](figures/backup-reuse.pdf) ([PNG](figures/backup-reuse.png))
- Method and limitations: [`../../docs/experiments/backup-reuse-reclaim/README.md`](../../docs/experiments/backup-reuse-reclaim/README.md)

The validator checks positive reused bytes/count, zero repeated D2H time, matching requested/released bytes, zero pending accounting, and material process-tree RSS plus `MemAvailable` recovery in a run-local coordinator settlement window. These local measurements were collected on 2026-08-13.
"""


def summary(family_dir: Path | None = None) -> dict[str, Any]:
    family = family_dir or default_results_root() / "backup-reuse-reclaim"
    reuse: list[dict[str, Any]] = []
    for model in MODELS:
        data = read_json(reuse_path(family, model))
        events = reuse_events(data)
        reuse.append(
            {
                "model": model,
                "events": len(events),
                "min_reuse_count": min(
                    int(event.get("cpu_backup_reuse_count", 0)) for event in events
                ),
                "min_reused_bytes": min(
                    int(event.get("cpu_backup_reused_bytes", 0)) for event in events
                ),
                "max_copy_d2h_s": max(float(event.get("copy_d2h_s", 0.0)) for event in events),
            }
        )
    reclaim = reclaim_metrics(read_json(reclaim_path(family)))
    return {
        "reuse": reuse,
        "reclaim": {
            **reclaim,
        },
    }


def write_figure(data: dict[str, Any], family_dir: Path) -> None:
    apply_paper_style()
    fig, axis = plt.subplots(figsize=(3.4, 2.2))
    labels = [item["model"] for item in data["reuse"]]
    values = [item["min_reused_bytes"] / (1024**3) for item in data["reuse"]]
    bars = axis.bar(
        labels,
        values,
        color=[system_color("Proposed")] * len(labels),
        hatch=system_hatch("vLLM L1"),
    )
    for bar, item in zip(bars, data["reuse"], strict=True):
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


def build(results_root: Path | None = None) -> None:
    family = (results_root or default_results_root()) / "backup-reuse-reclaim"
    prepare_family(family)
    data = summary(family)
    write_json(family / "summary.json", data)
    write_figure(data, family)
    write_result_readme(family, RESULT_README)
    write_family_metadata(
        "backup-reuse-reclaim",
        family,
        config=["config/claims.json"],
        validation={
            "reuse_models": 3,
            "reuse_samples_per_model": 5,
            "physical_reclaim_evidence": True,
        },
    )
