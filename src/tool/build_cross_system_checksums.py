from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    include = [
        root / "configs/traces/request-switch-alternating.jsonl",
        root / "configs/traces/request-switch-burst.jsonl",
        root / "results/cross_system/latest/external-systems.json",
        root / "results/cross_system/latest/summary.json",
        root / "results/cross_system/latest/lifecycle-latency.png",
        root / "results/cross_system/latest/trace-ttft.png",
    ]
    include.extend(sorted((root / "results/cross_system/raw").rglob("*.json")))
    include.extend(sorted((root / "results/cross_system/raw").rglob("*.csv")))
    include.extend(
        path
        for path in sorted((root / "results/cross_system/raw").rglob("*.jsonl"))
        if "repeated-l1" not in path.parts
    )
    unique = sorted({path.resolve() for path in include})
    missing = [path for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"artifact inputs missing: {missing}")
    manifest = {
        "format": "cross-system-checksums-v1",
        "files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in unique
            if path.name != "checksums.json"
        ],
    }
    output = root / "results/cross_system/latest/checksums.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
