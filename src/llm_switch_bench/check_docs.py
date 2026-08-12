from __future__ import annotations

import argparse

from llm_switch_bench.common.provenance import repository_root

ROOT = repository_root()
REQUIRED_DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "README.md",
    ROOT / "results" / "README.md",
    ROOT / "scripts" / "README.md",
    ROOT / "src" / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "CONTRIBUTING.md",
]
EXPERIMENT_DOCS = [
    ROOT / "docs" / "experiments" / name / "README.md"
    for name in ("lifecycle-latency", "request-driven-switch", "backup-reuse-reclaim", "exact-disk")
]
FORBIDDEN = [
    "src/bench_",
    "src/benchlib",
    "src/tool",
    "results/release-v0.1",
    "checksums.sha256",
    "all-files.sha256",
]
REQUIRED_DISCLOSURES = [
    "no new data was generated",
    "historical local observation",
    "canonical gpu rerun is not complete",
]
REQUIRED_EXPERIMENT_SECTIONS = (
    "## Question",
    "## Metric",
    "## Method",
    "## Retained result",
    "## Threats to validity",
    "## Limitations",
    "## Reproduce",
    "### Deterministic CPU rebuild and validation",
)
REQUIRED_LIVE_RUN_MARKERS = {
    "lifecycle-latency": ("### Live measurement", "results/tmp/", "Stop terminal 1"),
    "request-driven-switch": ("### Live single-trace measurement", "results/tmp/", "Cleanup"),
    "backup-reuse-reclaim": ("### Live same-process reuse and reclaim", "results/tmp/", "Cleanup"),
    "exact-disk": ("### Live exact-disk lifecycle capture", "results/tmp/", "Afterward confirm"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check current benchmark documentation policy.")
    parser.parse_args(argv)
    paths = REQUIRED_DOCS + EXPERIMENT_DOCS
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing docs: {missing}")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN:
            if needle in text:
                raise SystemExit(f"{path} contains obsolete reference {needle!r}")
    root_text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for needle in REQUIRED_DISCLOSURES:
        if needle.lower() not in root_text:
            raise SystemExit(f"README.md missing disclosure: {needle}")
    for family in (
        "lifecycle-latency",
        "request-driven-switch",
        "backup-reuse-reclaim",
        "exact-disk",
    ):
        text = (ROOT / "docs" / "experiments" / family / "README.md").read_text(encoding="utf-8")
        if f"../../../results/{family}/figures/" not in text:
            raise SystemExit(f"{family} doc does not link to result figure")
        for section in REQUIRED_EXPERIMENT_SECTIONS:
            if section not in text:
                raise SystemExit(f"{family} doc is missing required section {section!r}")
        for marker in REQUIRED_LIVE_RUN_MARKERS[family]:
            if marker not in text:
                raise SystemExit(f"{family} doc is missing live-run marker {marker!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
