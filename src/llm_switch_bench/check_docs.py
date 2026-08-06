from __future__ import annotations

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
    "canonical GPU rerun is not complete",
]


def main() -> int:
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
        if needle not in root_text:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
