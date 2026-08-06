#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def gpu_used_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return sum(int(line.strip()) for line in output.splitlines() if line.strip())


def request(base_url: str, model: str) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
    }
    started = time.perf_counter()
    text = ""
    done = False
    with requests.post(
        f"{base_url}/v1/chat/completions",
        json=body,
        stream=True,
        timeout=300,
    ) as response:
        status = response.status_code
        for line in response.iter_lines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                continue
            payload = json.loads(data)
            if payload.get("error"):
                raise RuntimeError(payload["error"])
            choice = (payload.get("choices") or [{}])[0]
            text += str((choice.get("delta") or {}).get("content") or choice.get("text") or "")
    return {
        "status": status,
        "done": done,
        "output": text,
        "latency_s": time.perf_counter() - started,
        "ok": 200 <= status < 300 and done and bool(text.strip()),
    }


def running_models(base_url: str) -> list[dict[str, Any]]:
    response = requests.get(f"{base_url}/running", timeout=10)
    response.raise_for_status()
    payload = response.json()
    return (
        payload
        if isinstance(payload, list)
        else payload.get("running", payload.get("data", payload.get("models", [])))
    )


def wait_unloaded(
    base_url: str,
    model: str,
    baseline_gpu_mib: int,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    threshold = max(128, baseline_gpu_mib + 128)
    last_running: list[dict[str, Any]] = []
    last_gpu = gpu_used_mib()
    while time.perf_counter() - started < timeout_s:
        last_running = running_models(base_url)
        active = {
            str(row.get("model", row.get("id", "")))
            for row in last_running
            if str(row.get("state", "")).lower() != "stopped"
        }
        last_gpu = gpu_used_mib()
        if model not in active and last_gpu <= threshold:
            return {
                "ok": True,
                "latency_s": time.perf_counter() - started,
                "gpu_used_mib": last_gpu,
                "idle_threshold_mib": threshold,
                "running": last_running,
            }
        time.sleep(0.05)
    return {
        "ok": False,
        "latency_s": time.perf_counter() - started,
        "gpu_used_mib": last_gpu,
        "idle_threshold_mib": threshold,
        "running": last_running,
    }


def unload(
    base_url: str,
    model: str,
    baseline_gpu_mib: int,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(f"{base_url}/api/models/unload/{model}", timeout=timeout_s)
    api_latency = time.perf_counter() - started
    wait = wait_unloaded(base_url, model, baseline_gpu_mib, timeout_s)
    return {
        "ok": response.status_code == 200 and wait["ok"],
        "status": response.status_code,
        "body": response.text[:200],
        "api_latency_s": api_latency,
        "postcondition_latency_s": wait["latency_s"],
        "latency_s": time.perf_counter() - started,
        "postcondition": wait,
    }


def git_metadata(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    return {
        "path": str(repo.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "tracked_dirty": bool(run("status", "--short", "--untracked-files=no")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure llama-swap process unload and cold wake phases separately."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18100")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--unload-timeout-s", type=float, default=60)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--lifecycle-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_gpu_mib = gpu_used_mib()
    profile_offset = 0
    rows: list[dict[str, Any]] = []
    for model in args.models:
        for cycle in range(args.cycles):
            wake = request(args.base_url, model)
            ready_gpu = gpu_used_mib()
            sleep = unload(
                args.base_url,
                model,
                baseline_gpu_mib,
                args.unload_timeout_s,
            )
            profile_events: list[dict[str, Any]] = []
            if args.lifecycle_profile:
                profile_events = [
                    json.loads(line)
                    for line in args.lifecycle_profile.read_text(encoding="utf-8").splitlines()[
                        profile_offset:
                    ]
                    if line.strip()
                ]
                profile_offset += len(profile_events)
            wake_events = [
                event
                for event in profile_events
                if event.get("model") == model
                and event.get("phase") == "wake"
                and event.get("success")
            ]
            sleep_events = [
                event
                for event in profile_events
                if event.get("model") == model
                and event.get("phase") == "sleep_process"
                and event.get("success")
            ]
            if args.lifecycle_profile:
                if len(wake_events) != 1 or len(sleep_events) != 1:
                    raise RuntimeError(
                        f"expected one successful wake and sleep event for {model}, "
                        f"got wake={len(wake_events)}, sleep={len(sleep_events)}"
                    )
                wake["state_machine_latency_s"] = float(wake_events[0]["duration_s"])
                sleep["state_machine_latency_s"] = float(sleep_events[0]["duration_s"])
            row = {
                "model": model,
                "cycle": cycle,
                "wake": wake,
                "sleep": sleep,
                "gpu_used_ready_mib": ready_gpu,
                "gpu_used_sleep_mib": sleep["postcondition"]["gpu_used_mib"],
                "ok": wake["ok"] and sleep["ok"],
                "definitions": {
                    "wake": "process state starting through health-ready state",
                    "sleep": "POST /api/models/unload/{model} through stopped process and idle-GPU post-condition",
                },
                "lifecycle_profile_events": profile_events,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if not row["ok"]:
                break
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "baseline_gpu_mib": baseline_gpu_mib,
        "llama_swap_repo": git_metadata(args.repo),
        "cycles": args.cycles,
        "rows": rows,
        "medians": {
            model: {
                "sleep_s": statistics.median(
                    row["sleep"]["latency_s"] for row in rows if row["model"] == model and row["ok"]
                ),
                "wake_s": statistics.median(
                    row["wake"].get("state_machine_latency_s", row["wake"]["latency_s"])
                    for row in rows
                    if row["model"] == model and row["ok"]
                ),
            }
            for model in args.models
            if any(row["model"] == model and row["ok"] for row in rows)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output)
    return 0 if all(row["ok"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
