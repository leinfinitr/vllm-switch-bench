#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


def gpu_used() -> int:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return sum(int(line) for line in text.splitlines() if line.strip())


def infer(base: str, model: str, api_key: str) -> str:
    response = requests.post(
        f"{base}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply exactly OK."}],
            "max_tokens": 8,
            "temperature": 0,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def post(base: str, path: str, model: str) -> float:
    started = time.perf_counter()
    response = requests.post(f"{base}{path}", json={"model": model}, timeout=300)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = infer(args.base_url, args.model, args.api_key)
    rows: list[dict[str, Any]] = []
    for cycle in range(args.cycles):
        sleep_s = post(args.base_url, "/api/swapout", args.model)
        sleep_gpu_mib = gpu_used()
        wake_s = post(args.base_url, "/api/swapin", args.model)
        output = infer(args.base_url, args.model, args.api_key)
        row = {
            "cycle": cycle,
            "sleep_s": sleep_s,
            "wake_s": wake_s,
            "sleep_gpu_mib": sleep_gpu_mib,
            "output_match": output == reference,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "system": "SwapServeLLM",
        "model": args.model,
        "rows": rows,
        "medians": {
            "sleep_s": statistics.median(row["sleep_s"] for row in rows),
            "wake_s": statistics.median(row["wake_s"] for row in rows),
        },
        "all_outputs_match": all(row["output_match"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["all_outputs_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
