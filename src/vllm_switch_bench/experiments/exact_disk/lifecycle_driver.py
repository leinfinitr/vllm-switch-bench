#!/usr/bin/env python3
"""Exercise exact CPU→disk demotion and disk→GPU restore through vLLM HTTP.

The server must be launched with sleep mode and developer mode enabled. The script
writes the deterministic before/after observation required by the exact-disk
collector and exits non-zero on any lifecycle or equality failure.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read()
    return json.loads(body) if body else None


def wait_ready(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"vLLM did not become ready: {last_error!r}")


def infer(base_url: str, model: str, prompt: str, timeout_s: float) -> dict[str, Any]:
    response = post_json(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 16,
            "seed": 0,
        },
        timeout_s,
    )
    choice = response["choices"][0]
    return {
        "text": choice["text"],
        "finish_reason": choice.get("finish_reason"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--served-model-name",
        default=os.environ.get("VLLM_SWITCH_BENCH_MODEL_NAME", "bench-model"),
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of sleep/wake validation cycles to execute in the same server.",
    )
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be positive")

    output_path_value = os.environ.get("VLLM_SWITCH_BENCH_OUTPUT_OBSERVATION")
    if not output_path_value:
        raise RuntimeError("VLLM_SWITCH_BENCH_OUTPUT_OBSERVATION is required")
    output_path = Path(output_path_value)

    wait_ready(args.base_url, args.ready_timeout_s)
    before = infer(args.base_url, args.served_model_name, args.prompt, 120.0)
    demotion_started = time.perf_counter()
    demotion = post_json(
        f"{args.base_url}/collective_rpc",
        {"method": "demote_weight_cpu_backup_to_disk"},
        300.0,
    )
    demotion_latency_s = time.perf_counter() - demotion_started
    results = demotion.get("results", []) if isinstance(demotion, dict) else []
    if not results or not all(
        int(result.get("released_bytes_total", 0)) > 0
        and int(result.get("remaining_cpu_backup_bytes", 0)) == 0
        and int(result.get("pending_release_bytes", 0)) == 0
        for result in results
        if isinstance(result, dict)
    ):
        raise RuntimeError(f"disk demotion did not complete: {demotion!r}")

    cycles: list[dict[str, Any]] = []
    after = before
    for cycle_index in range(args.cycles):
        sleep_started = time.perf_counter()
        post_json(f"{args.base_url}/sleep?level=1", {}, 300.0)
        sleep_latency_s = time.perf_counter() - sleep_started

        wake_started = time.perf_counter()
        post_json(f"{args.base_url}/wake_up", {}, 300.0)
        wake_latency_s = time.perf_counter() - wake_started

        after = infer(args.base_url, args.served_model_name, args.prompt, 120.0)
        output_equal = before == after
        cycles.append(
            {
                "cycle_index": cycle_index,
                "sleep_latency_s": sleep_latency_s,
                "wake_latency_s": wake_latency_s,
                "output_equal": output_equal,
            }
        )
        if not output_equal:
            raise RuntimeError(
                f"deterministic output changed after exact disk restore cycle {cycle_index}"
            )

    observation = {
        "schema_version": 1,
        "before": before,
        "after": after,
        "demotion": demotion,
        "demotion_latency_s": demotion_latency_s,
        "cycles": cycles,
    }
    output_path.write_text(
        json.dumps(observation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(observation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
