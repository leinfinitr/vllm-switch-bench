from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bench_serverless_llm import build_register_payload, infer


def gpu_used_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return sum(int(line.strip()) for line in output.splitlines() if line.strip())


def model_absent(base_url: str, name: str) -> bool:
    response = requests.get(f"{base_url}/v1/models", timeout=30)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("models", payload.get("data", []))
    return all(str(row.get("id", row.get("model", ""))) != name for row in rows)


def wait_deleted(base_url: str, name: str, threshold_mib: int, timeout_s: float) -> dict:
    started = time.perf_counter()
    while True:
        absent = model_absent(base_url, name)
        gpu = gpu_used_mib()
        elapsed = time.perf_counter() - started
        if absent and gpu <= threshold_mib:
            return {"ok": True, "latency_s": elapsed, "gpu_used_mib": gpu, "model_absent": absent}
        if elapsed >= timeout_s:
            return {"ok": False, "latency_s": elapsed, "gpu_used_mib": gpu, "model_absent": absent}
        time.sleep(0.25)


def register(base_url: str, payload: dict) -> requests.Response:
    return requests.post(f"{base_url}/register", json=payload, timeout=300)


def delete(base_url: str, name: str) -> requests.Response:
    return requests.post(f"{base_url}/delete", json={"model": name}, timeout=300)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8343")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--idle-threshold-mib", type=int, default=538)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_register_payload(
        args.model,
        args.name,
        args.name,
        args.max_model_len,
        args.gpu_memory_utilization,
    )
    rows = []
    if not model_absent(args.base_url, args.name):
        response = delete(args.base_url, args.name)
        response.raise_for_status()
        state = wait_deleted(args.base_url, args.name, args.idle_threshold_mib, 120)
        if not state["ok"]:
            raise RuntimeError(f"initial delete did not reclaim: {state}")

    initial = register(args.base_url, payload)
    initial.raise_for_status()
    warm = infer(args.base_url, args.name, "short_short", timeout_s=900)
    if not warm.get("ok"):
        raise RuntimeError(f"initial inference failed: {warm}")

    for cycle in range(args.cycles):
        sleep_started = time.perf_counter()
        delete_response = delete(args.base_url, args.name)
        delete_response.raise_for_status()
        deleted = wait_deleted(args.base_url, args.name, args.idle_threshold_mib, 120)
        sleep_s = time.perf_counter() - sleep_started
        if not deleted["ok"]:
            rows.append({"cycle": cycle, "ok": False, "sleep_s": sleep_s, "delete_state": deleted})
            break

        wake_started = time.perf_counter()
        register_response = register(args.base_url, payload)
        register_response.raise_for_status()
        restored = infer(args.base_url, args.name, "short_short", timeout_s=900)
        wake_s = time.perf_counter() - wake_started
        row = {
            "cycle": cycle,
            "model": str(args.model),
            "registered_model_name": args.name,
            "sleep_s": sleep_s,
            "wake_s": wake_s,
            "switch_s": sleep_s + wake_s,
            "delete_state": deleted,
            "restore_inference": restored,
            "gpu_used_awake_mib": gpu_used_mib(),
            "ok": bool(restored.get("ok")),
            "definition": "delete through model-absent+GPU-idle, plus register through first successful inference",
        }
        rows.append(row)
        if not row["ok"]:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(args.output)
    return 0 if len(rows) == args.cycles and all(row["ok"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
