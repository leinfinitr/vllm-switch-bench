#!/usr/bin/env python3
"""Create deterministic SHA-256 manifests for results/release-v0.1."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "release-v0.1"
PUBLICATION = ARTIFACT / "checksums.sha256"
COMPLETE = ARTIFACT / "all-files.sha256"
PUBLICATION_PATHS = [
    "figures/e2e-alternating-request-latency.pdf",
    "figures/lifecycle-qwen-0.5b.pdf",
    "figures/lifecycle-qwen-1.5b.pdf",
    "figures/lifecycle-qwen-3b.pdf",
    "lifecycle-summary.csv",
    "summary.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(paths: list[Path]) -> str:
    return "".join(
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
        for path in sorted(paths)
    )


def main() -> None:
    PUBLICATION.write_text(
        render([ARTIFACT / relative for relative in PUBLICATION_PATHS]),
        encoding="utf-8",
    )
    all_files = [
        path
        for path in ARTIFACT.rglob("*")
        if path.is_file() and path != COMPLETE
    ]
    COMPLETE.write_text(render(all_files), encoding="utf-8")
    print(PUBLICATION.relative_to(ROOT))
    print(COMPLETE.relative_to(ROOT))


if __name__ == "__main__":
    main()
