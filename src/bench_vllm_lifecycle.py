#!/usr/bin/env python3
"""Benchmark model lifecycle transitions for local vLLM sleep/cold-reload tests.

The harness is intentionally self-contained and conservative for the shared IPADS
server: it does not drop Linux page cache, does not change system-wide settings,
and records enough metadata for later manual inspection.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from benchlib.http import parse_openai_stream_response
from benchlib.sampling import Sampler, make_event, run_cmd
from benchlib.schema import JsonlLogger, PROMPTS, write_summary_csv

try:
    import requests
except Exception as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("requests is required to run this benchmark") from exc


SYSTEM_NAME = "vllm"


def wait_process_http_ok(proc: subprocess.Popen[str], url: str, timeout_s: float, log_path: Path) -> float:
    sess = requests.Session()
    sess.trust_env = False
    start = time.perf_counter()
    last = None
    while time.perf_counter() - start < timeout_s:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            except Exception:
                pass
            raise RuntimeError(f"vLLM process exited early with code {proc.returncode}; last={last}; log_tail=\n{tail}")
        try:
            response = sess.get(url, timeout=2)
            if response.status_code == 200:
                return time.perf_counter() - start
            last = f"HTTP {response.status_code}: {response.text[:120]}"
        except Exception as exc:
            last = repr(exc)
        time.sleep(0.5)
    raise TimeoutError(f"{url} not ready after {timeout_s}s; last={last}")



def start_vllm(args: argparse.Namespace, log_path: Path) -> subprocess.Popen[str]:
    cmd = [
        args.python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
    ]
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.load_format:
        cmd.extend(["--load-format", args.load_format])
    if args.quantization:
        cmd.extend(["--quantization", args.quantization])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.enable_sleep_mode:
        cmd.append("--enable-sleep-mode")

    env = os.environ.copy()
    python_bin_dir = str(Path(args.python).resolve().parent)
    env["PATH"] = python_bin_dir + os.pathsep + env.get("PATH", "")
    env.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    env.setdefault("VLLM_USE_V1", args.vllm_use_v1)
    if args.cuda_home:
        env["CUDA_HOME"] = args.cuda_home
        env["PATH"] = str(Path(args.cuda_home) / "bin") + os.pathsep + env["PATH"]
    if args.enable_server_dev_mode:
        env["VLLM_SERVER_DEV_MODE"] = "1"

    log_fh = log_path.open("w", encoding="utf-8", buffering=1)
    return subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env, cwd=args.workdir)



def stop_process(proc: subprocess.Popen[str] | None, timeout_s: float = 15) -> float:
    start = time.perf_counter()
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout_s)
    return time.perf_counter() - start



def post_json(url: str, payload: dict[str, Any], timeout_s: int = 300) -> requests.Response:
    sess = requests.Session()
    sess.trust_env = False
    return sess.post(url, json=payload, timeout=timeout_s)



def infer(args: argparse.Namespace, prompt_name: str) -> dict[str, Any]:
    item = PROMPTS[prompt_name]
    if args.endpoint == "completion":
        payload = {
            "model": args.served_model_name,
            "prompt": item["prompt"],
            "max_tokens": int(item["max_tokens"]),
            "temperature": 0,
            "stream": True,
            "guided_decoding_backend": "no-guide",
        }
        path = "/v1/completions"
    else:
        payload = {
            "model": args.served_model_name,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": int(item["max_tokens"]),
            "temperature": 0,
            "stream": True,
            "guided_decoding_backend": "no-guide",
        }
        path = "/v1/chat/completions"

    url = f"http://{args.host}:{args.port}{path}"
    sess = requests.Session()
    sess.trust_env = False
    started_at = time.perf_counter()
    status = None
    try:
        with sess.post(url, json=payload, stream=True, timeout=300) as response:
            status = response.status_code
            if response.status_code != 200:
                return {
                    "ok": False,
                    "status": status,
                    "error": response.text[:500],
                    "client_latency_s": time.perf_counter() - started_at,
                }
            parsed = parse_openai_stream_response(response, started_at=started_at, now_fn=time.perf_counter)
    except Exception as exc:
        total = time.perf_counter() - started_at
        return {"ok": False, "status": status, "error": repr(exc), "client_latency_s": total}

    total = time.perf_counter() - started_at
    output = parsed["output_text"]
    completion_tokens = parsed["completion_tokens"]
    approx_tokens = completion_tokens or max(1, len(output.split()))
    return {
        "ok": True,
        "status": status,
        "error": None,
        "ttft_s": parsed["ttft_s"],
        "client_latency_s": total,
        "approx_output_tokens": approx_tokens,
        "approx_tokens_per_s": (approx_tokens / total) if total > 0 else None,
        "output_prefix": output[:120],
    }



def call_sleep(args: argparse.Namespace, level: int) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/sleep?level={level}"
    started_at = time.perf_counter()
    try:
        response = post_json(url, {}, timeout_s=300)
        return {
            "ok": response.status_code == 200,
            "status": response.status_code,
            "latency_s": time.perf_counter() - started_at,
            "body": response.text[:500],
        }
    except Exception as exc:
        return {"ok": False, "latency_s": time.perf_counter() - started_at, "error": repr(exc)}



def call_wake(args: argparse.Namespace, tags: list[str] | None = None) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/wake_up"
    if tags:
        url += "?" + "&".join(f"tags={tag}" for tag in tags)
    started_at = time.perf_counter()
    try:
        response = post_json(url, {}, timeout_s=300)
        return {
            "ok": response.status_code == 200,
            "status": response.status_code,
            "latency_s": time.perf_counter() - started_at,
            "body": response.text[:500],
        }
    except Exception as exc:
        return {"ok": False, "latency_s": time.perf_counter() - started_at, "error": repr(exc)}



def call_rpc(args: argparse.Namespace, method: str, kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/collective_rpc"
    payload: dict[str, Any] = {"method": method}
    if kwargs:
        payload["kwargs"] = kwargs
    started_at = time.perf_counter()
    try:
        response = post_json(url, payload, timeout_s=300)
        return {
            "ok": response.status_code == 200,
            "status": response.status_code,
            "latency_s": time.perf_counter() - started_at,
            "body": response.text[:500],
        }
    except Exception as exc:
        return {"ok": False, "latency_s": time.perf_counter() - started_at, "error": repr(exc)}



def combine_restore_steps(*steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": all(bool(step.get("ok")) for step in steps),
        "status": "+".join(str(step.get("status", "error")) for step in steps),
        "latency_s": sum(float(step.get("latency_s") or 0.0) for step in steps),
    }



def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])



def run_one(args: argparse.Namespace, method: str, prompt_name: str, repeat_index: int, out_dir: Path) -> dict[str, Any]:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{method}_{prompt_name}_{repeat_index}_{uuid.uuid4().hex[:8]}"
    ctx = {
        "system": SYSTEM_NAME,
        "run_id": run_id,
        "method": method,
        "model": args.model,
        "prompt_name": prompt_name,
        "repeat_index": repeat_index,
    }
    event_log = JsonlLogger(out_dir / f"{run_id}.events.jsonl")
    server_log = out_dir / f"{run_id}.server.log"
    start_ts = time.time()
    proc: subprocess.Popen[str] | None = None
    summary: dict[str, Any] = {**ctx, "server_log": str(server_log), "event_log": str(event_log.path)}
    original_port: int | None = None
    try:
        original_port = args.port
        if args.port == 0:
            args.port = find_free_port(args.host)
        dynamic_port_mode = original_port == 0
        summary["port"] = args.port
        args.enable_sleep_mode = method.startswith("sleep_l")
        event_log.write(make_event(ctx, "run_start", start_ts, None))
        event_log.write(make_event(ctx, "process_start_begin", start_ts, None))
        proc = start_vllm(args, server_log)
        with Sampler(event_log, ctx, start_ts, lambda: proc.pid if proc else None, args.sample_interval_s):
            ready_s = wait_process_http_ok(proc, f"http://{args.host}:{args.port}/health", args.ready_timeout_s, server_log)
            summary["startup_to_health_s"] = ready_s
            event_log.write(make_event(ctx, "api_ready", start_ts, proc.pid, extra={"startup_to_health_s": ready_s}))
            before = infer(args, prompt_name)
            summary["infer_before"] = before
            event_log.write(make_event(ctx, "infer_before_end", start_ts, proc.pid, extra=before))

            if method == "cold_reload":
                event_log.write(make_event(ctx, "evict_begin", start_ts, proc.pid))
                summary["evict"] = {"ok": True, "latency_s": stop_process(proc)}
                proc = None
                event_log.write(make_event(ctx, "evict_end", start_ts, None, extra=summary["evict"]))
                time.sleep(args.idle_s)
                event_log.write(make_event(ctx, "restore_begin", start_ts, None))
                if dynamic_port_mode:
                    args.port = find_free_port(args.host)
                    summary["reload_port"] = args.port
                reload_log = server_log.with_suffix(".reload.log")
                proc = start_vllm(args, reload_log)
                restore_s = wait_process_http_ok(proc, f"http://{args.host}:{args.port}/health", args.ready_timeout_s, reload_log)
                summary["restore"] = {"ok": True, "latency_s": restore_s}
                event_log.write(make_event(ctx, "restore_end", start_ts, proc.pid, extra=summary["restore"]))
            elif method in {"sleep_l1", "sleep_l2"}:
                level = 1 if method == "sleep_l1" else 2
                event_log.write(make_event(ctx, "evict_begin", start_ts, proc.pid))
                sleep_result = call_sleep(args, level)
                summary["evict"] = sleep_result
                event_log.write(make_event(ctx, "evict_end", start_ts, proc.pid, extra=sleep_result))
                time.sleep(args.idle_s)
                event_log.write(make_event(ctx, "restore_begin", start_ts, proc.pid))
                if method == "sleep_l2":
                    wake_weights = call_wake(args, tags=["weights"])
                    event_log.write(make_event(ctx, "wake_weights_end", start_ts, proc.pid, extra=wake_weights))
                    reload_weights = call_rpc(args, "reload_weights")
                    event_log.write(make_event(ctx, "reload_weights_end", start_ts, proc.pid, extra=reload_weights))
                    wake_kv = call_wake(args, tags=["kv_cache"])
                    summary["restore"] = combine_restore_steps(wake_weights, reload_weights, wake_kv)
                    summary["restore"]["steps"] = {
                        "wake_weights": wake_weights,
                        "reload_weights": reload_weights,
                        "wake_kv_cache": wake_kv,
                    }
                else:
                    wake_result = call_wake(args)
                    summary["restore"] = wake_result
                event_log.write(make_event(ctx, "restore_end", start_ts, proc.pid, extra=summary["restore"]))
            else:
                raise ValueError(f"unknown method: {method}")

            after = infer(args, prompt_name)
            summary["infer_after"] = after
            event_log.write(make_event(ctx, "infer_after_end", start_ts, proc.pid if proc else None, extra=after))
            summary["ok"] = bool(before.get("ok")) and bool(after.get("ok")) and bool(summary.get("restore", {}).get("ok", True))
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = repr(exc)
        event_log.write(make_event(ctx, "run_error", start_ts, proc.pid if proc else None, note=repr(exc)))
    finally:
        stop_s = stop_process(proc)
        if original_port is not None:
            args.port = original_port
        summary["cleanup_s"] = stop_s
        event_log.write(make_event(ctx, "run_end", start_ts, None, extra={"cleanup_s": stop_s}))
        event_log.close()
    return summary



def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="bench-model")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workdir", default=str(Path.cwd()))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--vllm-use-v1", default="1")
    parser.add_argument("--cuda-home", default="/home/ljl/cuda-13.0")
    parser.add_argument("--enable-server-dev-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--load-format", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--endpoint", choices=["completion", "chat"], default="completion")
    parser.add_argument("--methods", nargs="+", default=["cold_reload", "sleep_l1", "sleep_l2"])
    parser.add_argument("--prompts", nargs="+", default=["short_short", "long_short", "short_long"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ready-timeout-s", type=float, default=240)
    parser.add_argument("--idle-s", type=float, default=2)
    parser.add_argument("--sample-interval-s", type=float, default=0.5)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    unknown_prompts = sorted(set(args.prompts) - set(PROMPTS))
    if unknown_prompts:
        raise SystemExit(f"unknown prompts: {unknown_prompts}; available={sorted(PROMPTS)}")
    return args



def read_gpu_metadata() -> str:
    try:
        return run_cmd(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
            timeout=10,
        ).stdout
    except Exception as exc:
        return f"unavailable: {exc!r}"



def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "system": SYSTEM_NAME,
        "argv": list(argv) if argv is not None else sys.argv,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "methods": args.methods,
        "prompts": args.prompts,
        "repeats": args.repeats,
        "gpu": read_gpu_metadata(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dry_run:
        print(out_dir)
        return 0

    rows: list[dict[str, Any]] = []
    for method in args.methods:
        for prompt_name in args.prompts:
            for repeat_index in range(args.repeats):
                row = run_one(args, method, prompt_name, repeat_index, out_dir)
                rows.append(row)
                (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
                write_summary_csv(out_dir / "summary.csv", rows)

    print(out_dir)
    return 0 if all(row.get("ok") for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
