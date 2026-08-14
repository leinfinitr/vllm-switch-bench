"""Deterministic summaries and figures for exact-disk lifecycle evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from vllm_switch_bench.plotting.style import (
    apply_paper_style,
    save_figure,
    system_color,
    system_hatch,
)
from vllm_switch_bench.publication import (
    default_results_root,
    prepare_family,
    read_json,
    write_family_metadata,
    write_json,
    write_result_readme,
)

RESULT_README = """# Exact disk

Question: does exact disk spill, restore, and physically release a 1 GiB exact-runtime payload while retaining integrity and output equality?

- Configuration: [`config/claims.json`](config/claims.json)
- Raw evidence: seven files under [`raw/exact-disk/`](raw/exact-disk/)
- Summary: [`summary.json`](summary.json)
- Figure: [`figures/exact-disk.pdf`](figures/exact-disk.pdf) ([PNG](figures/exact-disk.png))
- Method and limitations: [`../../docs/experiments/exact-disk/README.md`](../../docs/experiments/exact-disk/README.md)

The payload is intentionally omitted. Its SHA-256 and runtime bundle/chunk checksums remain correctness evidence. The validator checks phase coverage, no fallback, size/hash/chunk/manifest-commit consistency, material footprint, completed host-cache release and demotion, actual restore reads, run identity/config metadata, and equal output for the 2026-08-13 local run.
"""


def summary(family_dir: Path | None = None) -> dict[str, Any]:
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
        "disk_released_bytes": max(
            [
                int(event.get("released_bytes", 0))
                for event in events
                if event.get("phase") == "exact_disk_demotion"
            ],
            default=0,
        ),
        "restore_reused_bytes": sum(int(event.get("disk_read_bytes", 0)) for event in events),
        "payload_bytes": int(payload["payload_size_bytes"]),
        "payload_sha256": payload["payload_sha256"],
        "output_match": output.get("before") == output.get("after"),
    }


def write_figure(data: dict[str, Any], family_dir: Path) -> None:
    apply_paper_style()
    fig, axis = plt.subplots(figsize=(3.4, 2.1))
    fields = ("disk_spill_bytes", "disk_released_bytes", "restore_reused_bytes")
    axis.bar(
        ("Spill", "Release", "Restore"),
        [data[field] / (1024**3) for field in fields],
        color=system_color("exact-disk"),
        hatch=system_hatch("exact-disk"),
    )
    axis.set_ylabel("Data volume (GiB)")
    axis.set_title("Exact-disk lifecycle evidence")
    fig.tight_layout()
    save_figure(fig, family_dir / "figures" / "exact-disk")


def build(results_root: Path | None = None) -> None:
    family = (results_root or default_results_root()) / "exact-disk"
    prepare_family(family)
    data = summary(family)
    write_json(family / "summary.json", data)
    write_figure(data, family)
    write_result_readme(family, RESULT_README)
    write_family_metadata(
        "exact-disk",
        family,
        config=["config/claims.json"],
        validation={
            "payload_bytes": data["payload_bytes"],
            "output_match": data["output_match"],
            "runtime_checksum_retained": True,
        },
    )
