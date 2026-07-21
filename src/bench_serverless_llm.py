from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from benchlib.resources import query_gpu_memory_used_mib
from benchlib.schema import PROMPTS, write_summary_csv

SYSTEM_NAME = "serverless_llm"


def build_register_payload(
    model_path: Path,
    prompt_model_name: str,
    registered_model_name: str,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    return {
        "model": registered_model_name,
        "backend": "vllm",
        "num_gpus": 1,
        "auto_scaling_config": {
            "metric": "concurrency",
            "target": 1,
            "min_instances": 0,
            "max_instances": 1,
            "keep_alive": 0,
        },
        "backend_config": {
            "pretrained_model_name_or_path": str(model_path),
            "torch_dtype": "float16",
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": True,
            "load_format": "auto",
        },
        "benchmark_metadata": {
            "prompt_model_name": prompt_model_name,
            "source_model_path": str(model_path),
        },
    }


def infer(base_url: str, model_name: str, prompt_name: str, timeout_s: float = 900.0) -> dict[str, Any]:
    prompt = PROMPTS[prompt_name]
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt["prompt"]}],
        "max_tokens": int(prompt["max_tokens"]),
        "temperature": 0,
    }
    started_at = time.perf_counter()
    last_error = ""
    while True:
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s
            )
        except requests.exceptions.ReadTimeout as exc:
            total = time.perf_counter() - started_at
            return {
                "ok": False,
                "status": None,
                "error": f"request timed out after {timeout_s}s: {exc}",
                "client_latency_s": total,
            }
        total = time.perf_counter() - started_at
        if response.status_code != 200:
            return {
                "ok": False,
                "status": response.status_code,
                "error": response.text[:500],
                "client_latency_s": total,
            }
        try:
            data = response.json()
        except Exception as exc:
            return {
                "ok": False,
                "status": response.status_code,
                "error": f"invalid json: {exc}; {response.text[:300]}",
                "client_latency_s": total,
            }
        if data.get("choices"):
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
            completion_tokens = usage.get("completion_tokens") or max(
                1, len(content.split())
            )
            return {
                "ok": True,
                "status": response.status_code,
                "ttft_s": None,
                "client_latency_s": total,
                "approx_output_tokens": completion_tokens,
                "approx_tokens_per_s": completion_tokens / total
                if total > 0
                else None,
                "output_prefix": content[:120],
            }
        last_error = json.dumps(data, ensure_ascii=False)[:500]
        if total >= 300:
            return {
                "ok": False,
                "status": response.status_code,
                "error": f"no choices in response after retries: {last_error}",
                "client_latency_s": total,
            }
        time.sleep(1)


def _request_ok(
    client, method: str, url: str, payload: dict[str, Any] | None = None
) -> requests.Response:
    return client.request(method, url, json=payload, timeout=300)


def _source_model(payload: dict[str, Any]) -> str:
    return str(
        payload.get("benchmark_metadata", {}).get(
            "source_model_path", payload["model"]
        )
    )


def query_gpu_used_mib() -> int | None:
    return query_gpu_memory_used_mib()


def wait_for_scale_to_zero(
    base_url: str,
    client,
    model_name: str,
    baseline_gpu_used_mib: int | None,
    timeout_s: float,
    poll_interval_s: float,
    idle_gpu_buffer_mib: int,
    idle_gpu_threshold_mib_override: int | None = None,
) -> dict[str, Any]:
    threshold_mib: int | None = idle_gpu_threshold_mib_override
    if threshold_mib is None and baseline_gpu_used_mib is not None:
        threshold_mib = max(500, baseline_gpu_used_mib + idle_gpu_buffer_mib)

    started_at = time.perf_counter()
    last_gpu_used_mib = query_gpu_used_mib()
    last_models_error = None
    model_absent = False
    while True:
        elapsed_s = time.perf_counter() - started_at
        models_response = _request_ok(client, "GET", f"{base_url}/v1/models")
        if models_response.status_code != 200:
            last_models_error = (
                f"models failed: {models_response.status_code} "
                f"{models_response.text[:200]}"
            )
        else:
            last_models_error = None
            try:
                models_payload = models_response.json()
                model_rows = models_payload.get(
                    "models", models_payload.get("data", [])
                )
                model_absent = all(
                    str(row.get("id", row.get("model", ""))) != model_name
                    for row in model_rows
                )
            except Exception as exc:
                last_models_error = f"invalid models response: {exc}"
                model_absent = False

        last_gpu_used_mib = query_gpu_used_mib()
        if threshold_mib is not None and last_gpu_used_mib is not None:
            if model_absent and last_gpu_used_mib <= threshold_mib:
                return {
                    "ok": True,
                    "status": 200,
                    "latency_s": elapsed_s,
                    "baseline_gpu_used_mib": baseline_gpu_used_mib,
                    "gpu_used_mib": last_gpu_used_mib,
                    "idle_gpu_threshold_mib": threshold_mib,
                    "model_absent": True,
                    "body": f"verified scale-to-zero for {model_name}",
                }

        if elapsed_s >= timeout_s:
            error = (
                last_models_error
                or f"timed out waiting for GPU memory to return to idle; "
                f"baseline={baseline_gpu_used_mib}, last={last_gpu_used_mib}, "
                f"threshold={threshold_mib}"
            )
            return {
                "ok": False,
                "status": 408,
                "latency_s": elapsed_s,
                "baseline_gpu_used_mib": baseline_gpu_used_mib,
                "gpu_used_mib": last_gpu_used_mib,
                "idle_gpu_threshold_mib": threshold_mib,
                "model_absent": model_absent,
                "body": error,
            }
        time.sleep(poll_interval_s)


def run_delete_register(
    base_url: str,
    client,
    payload: dict[str, Any],
    prompt_name: str,
    repeat_index: int,
    scale_zero_timeout_s: float,
    scale_zero_poll_interval_s: float,
    idle_gpu_buffer_mib: int,
    request_timeout_s: float = 900.0,
) -> dict[str, Any]:
    model_name = payload["model"]
    row: dict[str, Any] = {
        "system": SYSTEM_NAME,
        "method": "delete_register",
        "model": _source_model(payload),
        "registered_model_name": model_name,
        "prompt_name": prompt_name,
        "repeat_index": repeat_index,
        "startup_latency_s": None,
    }
    health = _request_ok(client, "GET", f"{base_url}/health")
    if health.status_code != 200:
        row["ok"] = False
        row["error"] = (
            f"health check failed: {health.status_code} {health.text[:200]}"
        )
        return row

    register_response = _request_ok(client, "POST", f"{base_url}/register", payload)
    if register_response.status_code != 200:
        row["ok"] = False
        row["error"] = (
            f"register failed: {register_response.status_code} "
            f"{register_response.text[:200]}"
        )
        return row

    warmup_before = infer(base_url, model_name, prompt_name, timeout_s=request_timeout_s)
    if not warmup_before.get("ok"):
        row["ok"] = False
        row["error"] = warmup_before.get("error", "initial warmup failed")
        return row
    row["memory_gpu_used_ready_mib"] = query_gpu_used_mib()
    row["memory_cpu_used_ready_mib"] = None
    row["infer_before"] = infer(base_url, model_name, prompt_name, timeout_s=request_timeout_s)

    evict_start = time.perf_counter()
    delete_response = _request_ok(
        client, "POST", f"{base_url}/delete", {"model": model_name}
    )
    if delete_response.status_code != 200:
        row["evict"] = {
            "ok": False,
            "status": delete_response.status_code,
            "latency_s": time.perf_counter() - evict_start,
            "body": delete_response.text[:200],
        }
        row["ok"] = False
        row["error"] = (
            f"delete failed: {delete_response.status_code} {delete_response.text[:200]}"
        )
        return row

    idle_after_delete = wait_for_scale_to_zero(
        base_url=base_url,
        client=client,
        model_name=model_name,
        baseline_gpu_used_mib=None,
        timeout_s=scale_zero_timeout_s,
        poll_interval_s=scale_zero_poll_interval_s,
        idle_gpu_buffer_mib=idle_gpu_buffer_mib,
        idle_gpu_threshold_mib_override=538,
    )
    delete_elapsed_s = time.perf_counter() - evict_start
    idle_wait_s = idle_after_delete.get("latency_s")
    if idle_wait_s is not None:
        delete_elapsed_s += float(idle_wait_s)
    row["evict"] = {
        "ok": bool(idle_after_delete.get("ok")),
        "status": delete_response.status_code,
        "latency_s": delete_elapsed_s,
        "body": delete_response.text[:200],
        "idle_check": idle_after_delete,
    }
    if not idle_after_delete.get("ok"):
        row["ok"] = False
        row["error"] = idle_after_delete.get("body", "delete did not reach idle GPU state")
        return row

    row["memory_gpu_used_evict_mib"] = idle_after_delete.get("gpu_used_mib", idle_after_delete.get("idle_gpu_threshold_mib"))
    row["memory_cpu_used_evict_mib"] = None

    restore_start = time.perf_counter()
    restore_response = _request_ok(client, "POST", f"{base_url}/register", payload)
    if restore_response.status_code != 200:
        row["restore"] = {
            "ok": False,
            "status": restore_response.status_code,
            "latency_s": time.perf_counter() - restore_start,
            "body": restore_response.text[:200],
        }
        row["ok"] = False
        row["error"] = (
            f"restore register failed: {restore_response.status_code} "
            f"{restore_response.text[:200]}"
        )
        return row

    restore_warmup = infer(base_url, model_name, prompt_name, timeout_s=request_timeout_s)
    infer_after = infer(base_url, model_name, prompt_name, timeout_s=request_timeout_s)
    row["infer_after"] = infer_after
    restore_latency_s = None
    first_latency = restore_warmup.get("client_latency_s")
    second_latency = infer_after.get("client_latency_s")
    if first_latency is not None and second_latency is not None:
        restore_latency_s = max(0.0, float(first_latency) - float(second_latency))
    row["restore"] = {
        "ok": bool(restore_warmup.get("ok")) and bool(infer_after.get("ok")),
        "status": restore_response.status_code,
        "latency_s": restore_latency_s,
        "body": "register plus warm request minus second active request",
    }
    row["restore_latency_estimated"] = True
    row["ttft_available"] = False
    row["tpot_available"] = False
    row["stage_breakdown"] = {
        "initial_warm_request_s": warmup_before.get("client_latency_s"),
        "restore_warm_request_s": first_latency,
        "second_active_request_s": second_latency,
        "delete_idle_wait_s": idle_after_delete.get("latency_s"),
    }
    row["ok"] = bool(row["infer_before"].get("ok")) and bool(restore_warmup.get("ok")) and bool(
        row["infer_after"].get("ok")
    )
    return row


def run_scale_to_zero_restore(
    base_url: str,
    client,
    payload: dict[str, Any],
    prompt_name: str,
    repeat_index: int,
    scale_zero_timeout_s: float,
    scale_zero_poll_interval_s: float,
    idle_gpu_buffer_mib: int,
    request_timeout_s: float = 900.0,
) -> dict[str, Any]:
    model_name = payload["model"]
    row: dict[str, Any] = {
        "system": SYSTEM_NAME,
        "method": "scale_to_zero_restore",
        "model": _source_model(payload),
        "registered_model_name": model_name,
        "prompt_name": prompt_name,
        "repeat_index": repeat_index,
        "startup_latency_s": None,
    }
    health = _request_ok(client, "GET", f"{base_url}/health")
    if health.status_code != 200:
        row["ok"] = False
        row["error"] = (
            f"health check failed: {health.status_code} {health.text[:200]}"
        )
        return row

    baseline_idle = wait_for_scale_to_zero(
        base_url=base_url,
        client=client,
        model_name=model_name,
        baseline_gpu_used_mib=None,
        timeout_s=scale_zero_timeout_s,
        poll_interval_s=scale_zero_poll_interval_s,
        idle_gpu_buffer_mib=idle_gpu_buffer_mib,
        idle_gpu_threshold_mib_override=538,
    )
    if not baseline_idle.get("ok"):
        row["ok"] = False
        row["error"] = baseline_idle.get("body", "failed to reach baseline idle GPU state")
        return row
    baseline_gpu_used_mib = query_gpu_used_mib()

    register_response = _request_ok(client, "POST", f"{base_url}/register", payload)
    if register_response.status_code != 200:
        row["ok"] = False
        row["error"] = (
            f"register failed: {register_response.status_code} "
            f"{register_response.text[:200]}"
        )
        return row

    warmup_before = infer(base_url, model_name, prompt_name, timeout_s=request_timeout_s)
    if not warmup_before.get("ok"):
        row["ok"] = False
        row["error"] = warmup_before.get("error", "initial warmup failed")
        return row
    row["memory_gpu_used_ready_mib"] = query_gpu_used_mib()
    row["memory_cpu_used_ready_mib"] = None
    row["infer_before"] = infer(base_url, model_name, prompt_name, timeout_s=request_timeout_s)
    if not row["infer_before"].get("ok"):
        row["ok"] = False
        row["error"] = row["infer_before"].get("error", "infer_before failed")
        return row

    row["evict"] = wait_for_scale_to_zero(
        base_url=base_url,
        client=client,
        model_name=model_name,
        baseline_gpu_used_mib=baseline_gpu_used_mib,
        timeout_s=scale_zero_timeout_s,
        poll_interval_s=scale_zero_poll_interval_s,
        idle_gpu_buffer_mib=idle_gpu_buffer_mib,
        idle_gpu_threshold_mib_override=538,
    )
    if not row["evict"].get("ok"):
        row["ok"] = False
        row["error"] = row["evict"].get("body", "scale-to-zero verification failed")
        return row

    row["memory_gpu_used_evict_mib"] = row["evict"].get("gpu_used_mib", row["evict"].get("idle_gpu_threshold_mib"))
    row["memory_cpu_used_evict_mib"] = None

    restore_warmup = infer(base_url, model_name, prompt_name, timeout_s=request_timeout_s)
    infer_after = infer(base_url, model_name, prompt_name, timeout_s=request_timeout_s)
    row["infer_after"] = infer_after
    first_latency = restore_warmup.get("client_latency_s")
    second_latency = infer_after.get("client_latency_s")
    restore_latency_s = None
    if first_latency is not None and second_latency is not None:
        restore_latency_s = max(0.0, float(first_latency) - float(second_latency))
    row["restore"] = {
        "ok": bool(restore_warmup.get("ok")) and bool(infer_after.get("ok")),
        "status": restore_warmup.get("status"),
        "latency_s": restore_latency_s,
        "body": "estimated from restore warm request minus second active request",
    }
    row["restore_latency_estimated"] = True
    row["ttft_available"] = False
    row["tpot_available"] = False
    row["stage_breakdown"] = {
        "baseline_idle_wait_s": baseline_idle.get("latency_s"),
        "scale_to_zero_wait_s": row["evict"].get("latency_s"),
        "initial_warm_request_s": warmup_before.get("client_latency_s"),
        "restore_warm_request_s": first_latency,
        "second_active_request_s": second_latency,
        "baseline_gpu_used_mib": baseline_gpu_used_mib,
        "idle_gpu_threshold_mib": row["evict"].get("idle_gpu_threshold_mib"),
    }
    row["ok"] = bool(row["infer_before"].get("ok")) and bool(restore_warmup.get("ok")) and bool(
        row["infer_after"].get("ok")
    )
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--registered-model-name", default="qwen2p5-0p5b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8343")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["delete_register"],
        choices=["delete_register", "scale_to_zero_restore"],
    )
    parser.add_argument("--prompts", nargs="+", default=["short_short"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--scale-zero-timeout", type=float, default=120.0)
    parser.add_argument("--scale-zero-poll-interval", type=float, default=0.001)
    parser.add_argument("--idle-gpu-buffer-mib", type=int, default=300)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--out-dir", default="results/tmp/serverless_llm")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    payload = build_register_payload(
        Path(args.model),
        Path(args.model).name,
        args.registered_model_name,
        args.max_model_len,
        args.gpu_memory_utilization,
    )
    client = requests.Session()
    for method in args.methods:
        for prompt_name in args.prompts:
            for repeat_index in range(args.repeats):
                if method == "delete_register":
                    row = run_delete_register(
                        args.base_url,
                        client,
                        payload,
                        prompt_name,
                        repeat_index,
                        args.scale_zero_timeout,
                        args.scale_zero_poll_interval,
                        args.idle_gpu_buffer_mib,
                        args.request_timeout,
                    )
                elif method == "scale_to_zero_restore":
                    row = run_scale_to_zero_restore(
                        args.base_url,
                        client,
                        payload,
                        prompt_name,
                        repeat_index,
                        args.scale_zero_timeout,
                        args.scale_zero_poll_interval,
                        args.idle_gpu_buffer_mib,
                        args.request_timeout,
                    )
                else:
                    raise ValueError(f"unsupported method: {method}")
                rows.append(row)
    (out_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary_csv(out_dir / "summary.csv", rows)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "system": SYSTEM_NAME,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "repo": args.repo,
                "base_url": args.base_url,
                "methods": args.methods,
                "registered_model_name": args.registered_model_name,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(out_dir)
    return 0 if all(row.get("ok") for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
