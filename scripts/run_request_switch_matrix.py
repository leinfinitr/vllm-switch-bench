#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated request-switch traces")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traces = {
        "w0": root / "configs/traces/request-switch-steady.jsonl",
        "w1": root / "configs/traces/request-switch-alternating.jsonl",
        "w2": root / "configs/traces/request-switch-burst.jsonl",
    }
    metadata = {
        "base_url": args.base_url,
        "repeats": args.repeats,
        "workloads": list(traces),
        "started_at": time.time(),
        "bench_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "runs": [],
    }
    for repeat in range(args.repeats):
        for workload, manifest in traces.items():
            output = out / f"{workload}-r{repeat}.jsonl"
            command = [
                str(root / ".venv/bin/python"),
                str(root / "src/bench_request_driven_switch.py"),
                "--base-url",
                args.base_url,
                "--manifest",
                str(manifest),
                "--output",
                str(output),
            ]
            completed = subprocess.run(command, cwd=root, check=False, text=True, capture_output=True)
            metadata["runs"].append(
                {
                    "workload": workload,
                    "repeat": repeat,
                    "manifest": str(manifest.relative_to(root)),
                    "output": output.name,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                }
            )
            if completed.returncode != 0:
                Path(out / "metadata.json").write_text(json.dumps(metadata, indent=2))
                raise SystemExit(completed.returncode)
    for manifest in traces.values():
        shutil.copy2(manifest, out / manifest.name)
    metadata["completed_at"] = time.time()
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
