#!/usr/bin/env python3
"""Profile model lifecycle transitions for local vLLM and vllm-switch tests.

The harness is intentionally self-contained and conservative for the shared IPADS
server: it does not drop Linux page cache, does not change system-wide settings,
and records enough metadata for later manual inspection.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from vllm_switch_bench.common.http import parse_openai_stream_response
from vllm_switch_bench.common.provenance import repository_root
from vllm_switch_bench.common.resources import process_tree_rss_mib, query_gpu_memory_used_mib
from vllm_switch_bench.common.sampling import Sampler, make_event, run_cmd
from vllm_switch_bench.common.schema import JsonlLogger, PROMPTS, write_summary_csv

try:
    import requests
except Exception as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("requests is required to run this benchmark") from exc


SYSTEM_NAME = "vllm"


def git_metadata(path: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    status = command("status", "--porcelain")
    tracked_status = command("status", "--short", "--untracked-files=no")
    return {
        "path": str(path.resolve()),
        "commit": command("rev-parse", "HEAD"),
        "tree": command("rev-parse", "HEAD^{tree}"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(status),
        "status_porcelain": status,
        "tracked_dirty": bool(tracked_status),
    }


def _engine_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    workdir = str(Path(args.workdir).resolve())
    benchmark_root = repository_root()
    benchmark_src = str(benchmark_root / "src")
    env["VLLM_SWITCH_BENCH_ROOT"] = str(benchmark_root)
    pythonpath = [workdir, benchmark_src]
    pythonpath.extend(
        entry
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and entry not in pythonpath
    )
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    python_bin_dir = str(Path(args.python).absolute().parent)
    env["PATH"] = python_bin_dir + os.pathsep + env.get("PATH", "")
    env.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    env.setdefault("VLLM_USE_V1", args.vllm_use_v1)
    if args.cuda_home:
        env["CUDA_HOME"] = args.cuda_home
        env["PATH"] = str(Path(args.cuda_home) / "bin") + os.pathsep + env["PATH"]
    if args.enable_server_dev_mode:
        env["VLLM_SERVER_DEV_MODE"] = "1"
    if getattr(args, "sleep_profile_path", None):
        env["VLLM_SLEEP_PROFILE_PATH"] = str(args.sleep_profile_path)
    if args.sleep_cpu_backup_pin_memory != "auto":
        env["VLLM_SLEEP_CPU_BACKUP_PIN_MEMORY"] = args.sleep_cpu_backup_pin_memory
    return env


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def engine_runtime_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Query the selected engine interpreter instead of this package environment."""

    python_path = Path(args.python).absolute()
    probe = """
import json
import platform
import torch
import vllm

print(json.dumps({
    "python_version": platform.python_version(),
    "vllm_import_path": str(vllm.__file__),
    "vllm_version": getattr(vllm, "__version__", None),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
}))
"""
    try:
        completed = subprocess.run(
            [str(python_path), "-c", probe],
            cwd=args.workdir,
            env=_engine_environment(args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        imported = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        imported = {"probe_error": repr(exc)}
    return {
        "python_path": str(python_path),
        "python_resolved_path": str(python_path.resolve()),
        "python_sha256": _sha256(python_path.resolve()),
        **imported,
    }


def model_metadata(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model).expanduser()
    config_path = model_path / "config.json"
    observed_digest = _sha256(config_path) if config_path.is_file() else None
    return {
        "identity": args.model,
        "resolved_path": str(model_path.resolve()) if model_path.exists() else None,
        "revision": args.model_revision,
        "config_path": str(config_path.resolve()) if config_path.is_file() else None,
        "config_sha256": args.model_config_sha256 or observed_digest,
        "config_sha256_observed": observed_digest,
    }


def behavior_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: getattr(args, key)
        for key in (
            "model",
            "model_revision",
            "model_config_sha256",
            "served_model_name",
            "host",
            "port",
            "cuda_visible_devices",
            "vllm_use_v1",
            "cuda_home",
            "enable_server_dev_mode",
            "max_model_len",
            "gpu_memory_utilization",
            "dtype",
            "load_format",
            "quantization",
            "enforce_eager",
            "extra_vllm_arg",
            "endpoint",
            "methods",
            "prompts",
            "repeats",
            "cycles_per_process",
            "ready_timeout_s",
            "idle_s",
            "sample_interval_s",
            "sleep_cpu_backup_pin_memory",
            "cold_max_resident_ratio",
            "cold_min_read_ratio",
            "warm_min_resident_ratio",
            "warm_max_read_ratio",
        )
    }


def wait_process_http_ok(
    proc: subprocess.Popen[str], url: str, timeout_s: float, log_path: Path
) -> float:
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
            raise RuntimeError(
                f"vLLM process exited early with code {proc.returncode}; last={last}; log_tail=\n{tail}"
            )
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
    python_path = Path(args.python).absolute()
    cmd = [
        str(python_path),
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
    for extra in args.extra_vllm_arg:
        cmd.extend(extra.split())

    env = _engine_environment(args)

    log_fh = log_path.open("w", encoding="utf-8", buffering=1)
    return subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env, cwd=args.workdir
    )


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
            parsed = parse_openai_stream_response(
                response, started_at=started_at, now_fn=time.perf_counter
            )
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
        "completion_tokens": completion_tokens,
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


def call_rpc(
    args: argparse.Namespace, method: str, kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def collect_sleep_profile_window(
    path: Path,
    operation: str,
    started_monotonic_s: float,
    ended_monotonic_s: float,
) -> dict[str, Any]:
    events = [
        event
        for event in _read_jsonl(path)
        if started_monotonic_s <= float(event.get("monotonic_s", -1.0)) <= ended_monotonic_s
    ]
    return {
        "operation": operation,
        "started_monotonic_s": started_monotonic_s,
        "ended_monotonic_s": ended_monotonic_s,
        "event_count": len(events),
        "phase_latency_s": {
            str(event.get("phase")): event.get("latency_s")
            for event in events
            if event.get("latency_s") is not None
        },
        "events": events,
    }


def call_with_sleep_profile(args: argparse.Namespace, operation: str, fn) -> dict[str, Any]:
    started_monotonic_s = time.perf_counter()
    result = fn()
    ended_monotonic_s = time.perf_counter()
    profile_path = getattr(args, "sleep_profile_path", "")
    if profile_path:
        result["sleep_profile"] = collect_sleep_profile_window(
            Path(profile_path), operation, started_monotonic_s, ended_monotonic_s
        )
    return result


def combine_restore_steps(
    *steps: dict[str, Any],
    started_monotonic_s: float | None = None,
    ended_monotonic_s: float | None = None,
) -> dict[str, Any]:
    active_latency_s = sum(float(step.get("latency_s") or 0.0) for step in steps)
    envelope_latency_s = (
        ended_monotonic_s - started_monotonic_s
        if started_monotonic_s is not None and ended_monotonic_s is not None
        else active_latency_s
    )
    combined = {
        "ok": all(bool(step.get("ok")) for step in steps),
        "status": "+".join(str(step.get("status", "error")) for step in steps),
        "latency_s": envelope_latency_s,
        "active_latency_s": active_latency_s,
        "inter_step_gap_s": max(0.0, envelope_latency_s - active_latency_s),
    }
    profiles = [step.get("sleep_profile") for step in steps if step.get("sleep_profile")]
    if profiles:
        combined["sleep_profile"] = {
            "operation": "restore",
            "event_count": sum(int((profile or {}).get("event_count", 0)) for profile in profiles),
            "steps": profiles,
        }
    return combined


def flatten_sleep_profile_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []

    def append_profile(row: dict[str, Any], parent_operation: str, profile: dict[str, Any]) -> None:
        for event in profile.get("events", []):
            flat_rows.append(
                {
                    "system": row.get("system"),
                    "run_id": row.get("run_id"),
                    "method": row.get("method"),
                    "model": row.get("model"),
                    "prompt_name": row.get("prompt_name"),
                    "repeat_index": row.get("repeat_index"),
                    "operation": profile.get("operation", parent_operation),
                    "phase": event.get("phase"),
                    "pid": event.get("pid"),
                    "latency_s": event.get("latency_s"),
                    "copy_d2h_s": event.get("copy_d2h_s"),
                    "copy_h2d_s": event.get("copy_h2d_s"),
                    "create_map_s": event.get("create_map_s"),
                    "unmap_release_s": event.get("unmap_release_s"),
                    "metadata_s": event.get("metadata_s"),
                    "cpu_backup_alloc_s": event.get("cpu_backup_alloc_s"),
                    "cpu_backup_pool_hit_count": event.get("cpu_backup_pool_hit_count"),
                    "cpu_backup_pool_miss_count": event.get("cpu_backup_pool_miss_count"),
                    "cpu_backup_pool_reserved_bytes": event.get("cpu_backup_pool_reserved_bytes"),
                    "cpu_backup_pool_free_bytes": event.get("cpu_backup_pool_free_bytes"),
                    "cpu_backup_data_ptr_s": event.get("cpu_backup_data_ptr_s"),
                    "assign_backup_s": event.get("assign_backup_s"),
                    "discard_accounting_s": event.get("discard_accounting_s"),
                    "loop_s": event.get("loop_s"),
                    "loop_accounted_s": event.get("loop_accounted_s"),
                    "loop_unaccounted_s": event.get("loop_unaccounted_s"),
                    "logger_s": event.get("logger_s"),
                    "gc_s": event.get("gc_s"),
                    "empty_cache_s": event.get("empty_cache_s"),
                    "accounted_s": event.get("accounted_s"),
                    "unaccounted_s": event.get("unaccounted_s"),
                    "allocator_sleep_s": event.get("allocator_sleep_s"),
                    "allocator_wake_up_s": event.get("allocator_wake_up_s"),
                    "buffer_backup_s": event.get("buffer_backup_s"),
                    "restore_buffers_s": event.get("restore_buffers_s"),
                    "post_kv_cache_wake_up_s": event.get("post_kv_cache_wake_up_s"),
                    "get_iterator_s": event.get("get_iterator_s"),
                    "iterator_first_yield_s": event.get("iterator_first_yield_s"),
                    "iterator_total_s": event.get("iterator_total_s"),
                    "iterator_gap_s": event.get("iterator_gap_s"),
                    "iterator_consumer_s": event.get("iterator_consumer_s"),
                    "initialize_layerwise_reload_s": event.get("initialize_layerwise_reload_s"),
                    "model_load_weights_s": event.get("model_load_weights_s"),
                    "finalize_layerwise_reload_s": event.get("finalize_layerwise_reload_s"),
                    "is_checkpoint_format": event.get("is_checkpoint_format"),
                    "weights_from_disk": event.get("weights_from_disk"),
                    "weights_path": event.get("weights_path"),
                    "iterator_tensor_count": event.get("iterator_tensor_count"),
                    "iterator_tensor_bytes": event.get("iterator_tensor_bytes"),
                    "cpu_backup_pin_memory": event.get("cpu_backup_pin_memory"),
                    "allocation_count": event.get("allocation_count"),
                    "total_bytes": event.get("total_bytes"),
                    "backup_bytes": event.get("backup_bytes"),
                    "discard_bytes": event.get("discard_bytes"),
                    "bytes": event.get("bytes"),
                    "bytes_by_tag": json.dumps(event.get("bytes_by_tag", {}), sort_keys=True),
                    "backup_bytes_by_tag": json.dumps(
                        event.get("backup_bytes_by_tag", {}), sort_keys=True
                    ),
                    "discard_bytes_by_tag": json.dumps(
                        event.get("discard_bytes_by_tag", {}), sort_keys=True
                    ),
                    "restored_bytes_by_tag": json.dumps(
                        event.get("restored_bytes_by_tag", {}), sort_keys=True
                    ),
                    "remapped_without_backup_bytes_by_tag": json.dumps(
                        event.get("remapped_without_backup_bytes_by_tag", {}),
                        sort_keys=True,
                    ),
                }
            )

    for row in rows:
        for operation in ("evict", "restore"):
            profile = row.get(operation, {}).get("sleep_profile")
            if not isinstance(profile, dict):
                continue
            if "steps" in profile:
                for step_profile in profile["steps"]:
                    append_profile(row, operation, step_profile)
            else:
                append_profile(row, operation, profile)
    return flat_rows


def write_sleep_profile_summary_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    flat_rows = flatten_sleep_profile_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()) if flat_rows else [])
        if flat_rows:
            writer.writeheader()
            writer.writerows(flat_rows)


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def run_one(
    args: argparse.Namespace, method: str, prompt_name: str, repeat_index: int, out_dir: Path
) -> dict[str, Any]:
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
    sleep_profile_path = out_dir / f"{run_id}.sleep_profile.jsonl"
    start_ts = time.time()
    proc: subprocess.Popen[str] | None = None
    summary: dict[str, Any] = {
        **ctx,
        "server_log": str(server_log),
        "event_log": str(event_log.path),
    }
    original_port: int | None = None
    try:
        original_port = args.port
        if args.port == 0:
            args.port = find_free_port(args.host)
        dynamic_port_mode = original_port == 0
        summary["port"] = args.port
        args.enable_sleep_mode = method.startswith("sleep_l") or method == "cpu_backup"
        args.sleep_profile_path = (
            str(sleep_profile_path.resolve()) if args.enable_sleep_mode else ""
        )
        if args.sleep_profile_path:
            sleep_profile_path.unlink(missing_ok=True)
            summary["sleep_profile_log"] = args.sleep_profile_path
        event_log.write(make_event(ctx, "run_start", start_ts, None))
        event_log.write(make_event(ctx, "process_start_begin", start_ts, None))
        proc = start_vllm(args, server_log)
        with Sampler(
            event_log, ctx, start_ts, lambda: proc.pid if proc else None, args.sample_interval_s
        ):
            ready_s = wait_process_http_ok(
                proc, f"http://{args.host}:{args.port}/health", args.ready_timeout_s, server_log
            )
            summary["startup_latency_s"] = ready_s
            summary["memory_gpu_used_ready_mib"] = query_gpu_memory_used_mib()
            summary["memory_cpu_used_ready_mib"] = process_tree_rss_mib(proc.pid if proc else None)
            event_log.write(
                make_event(
                    ctx, "api_ready", start_ts, proc.pid, extra={"startup_latency_s": ready_s}
                )
            )
            before = infer(args, prompt_name)
            summary["infer_before"] = before
            event_log.write(make_event(ctx, "infer_before_end", start_ts, proc.pid, extra=before))

            if method == "cold_reload":
                event_log.write(make_event(ctx, "evict_begin", start_ts, proc.pid))
                summary["evict"] = {"ok": True, "latency_s": stop_process(proc)}
                summary["memory_gpu_used_evict_mib"] = query_gpu_memory_used_mib()
                summary["memory_cpu_used_evict_mib"] = None
                proc = None
                event_log.write(
                    make_event(ctx, "evict_end", start_ts, None, extra=summary["evict"])
                )
                time.sleep(args.idle_s)
                event_log.write(make_event(ctx, "restore_begin", start_ts, None))
                if dynamic_port_mode:
                    args.port = find_free_port(args.host)
                    summary["reload_port"] = args.port
                reload_log = server_log.with_suffix(".reload.log")
                proc = start_vllm(args, reload_log)
                restore_s = wait_process_http_ok(
                    proc, f"http://{args.host}:{args.port}/health", args.ready_timeout_s, reload_log
                )
                summary["restore"] = {"ok": True, "latency_s": restore_s}
                event_log.write(
                    make_event(ctx, "restore_end", start_ts, proc.pid, extra=summary["restore"])
                )
            elif method in {"sleep_l1", "sleep_l2"}:
                level = 1 if method == "sleep_l1" else 2
                event_log.write(make_event(ctx, "evict_begin", start_ts, proc.pid))
                sleep_result = call_with_sleep_profile(
                    args, "sleep", lambda: call_sleep(args, level)
                )
                summary["evict"] = sleep_result
                summary["memory_gpu_used_evict_mib"] = query_gpu_memory_used_mib()
                summary["memory_cpu_used_evict_mib"] = process_tree_rss_mib(
                    proc.pid if proc else None
                )
                event_log.write(
                    make_event(ctx, "evict_end", start_ts, proc.pid, extra=sleep_result)
                )
                time.sleep(args.idle_s)
                event_log.write(make_event(ctx, "restore_begin", start_ts, proc.pid))
                if method == "sleep_l2":
                    wake_weights = call_with_sleep_profile(
                        args, "wake_weights", lambda: call_wake(args, tags=["weights"])
                    )
                    event_log.write(
                        make_event(ctx, "wake_weights_end", start_ts, proc.pid, extra=wake_weights)
                    )
                    reload_weights = call_with_sleep_profile(
                        args,
                        "reload_weights",
                        lambda: call_rpc(args, "reload_weights"),
                    )
                    event_log.write(
                        make_event(
                            ctx, "reload_weights_end", start_ts, proc.pid, extra=reload_weights
                        )
                    )
                    wake_kv = call_with_sleep_profile(
                        args, "wake_kv_cache", lambda: call_wake(args, tags=["kv_cache"])
                    )
                    summary["restore"] = combine_restore_steps(
                        wake_weights, reload_weights, wake_kv
                    )
                    summary["restore"]["steps"] = {
                        "wake_weights": wake_weights,
                        "reload_weights": reload_weights,
                        "wake_kv_cache": wake_kv,
                    }
                else:
                    wake_result = call_with_sleep_profile(args, "wake", lambda: call_wake(args))
                    summary["restore"] = wake_result
                event_log.write(
                    make_event(ctx, "restore_end", start_ts, proc.pid, extra=summary["restore"])
                )
            else:
                raise ValueError(f"unknown method: {method}")

            after = infer(args, prompt_name)
            summary["infer_after"] = after
            event_log.write(
                make_event(
                    ctx, "infer_after_end", start_ts, proc.pid if proc else None, extra=after
                )
            )
            summary["ok"] = (
                bool(before.get("ok"))
                and bool(after.get("ok"))
                and bool(summary.get("restore", {}).get("ok", True))
            )
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = repr(exc)
        event_log.write(
            make_event(ctx, "run_error", start_ts, proc.pid if proc else None, note=repr(exc))
        )
    finally:
        stop_s = stop_process(proc)
        if original_port is not None:
            args.port = original_port
        summary["cleanup_s"] = stop_s
        event_log.write(make_event(ctx, "run_end", start_ts, None, extra={"cleanup_s": stop_s}))
        event_log.close()
    return summary


def _run_block_driver(
    args: argparse.Namespace, method: str, block_index: int, out_dir: Path
) -> None:
    block_dir = out_dir / method / f"block-{block_index}"
    command = [
        str(Path(args.python).absolute()),
        "-m",
        "vllm_switch_bench.experiments.vllm_profiling.block_driver",
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--method",
        method,
        "--process-block",
        str(block_index),
        "--cycles",
        str(args.cycles_per_process),
        "--prompt",
        args.prompts[0],
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--dtype",
        args.dtype,
        "--model-revision",
        args.model_revision or "",
        "--load-format",
        args.load_format,
        "--quantization",
        args.quantization,
        "--idle-s",
        str(args.idle_s),
        "--out-dir",
        str(block_dir),
        "--cold-max-resident-ratio",
        str(args.cold_max_resident_ratio),
        "--cold-min-read-ratio",
        str(args.cold_min_read_ratio),
        "--warm-min-resident-ratio",
        str(args.warm_min_resident_ratio),
        "--warm-max-read-ratio",
        str(args.warm_max_read_ratio),
    ]
    for extra in args.extra_vllm_arg:
        command.extend(["--extra-vllm-arg", extra])
    if args.enforce_eager:
        command.append("--enforce-eager")
    process = subprocess.Popen(
        command,
        cwd=args.workdir,
        env=_engine_environment(args),
        text=True,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=args.ready_timeout_s + 900)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        raise
    if return_code != 0:
        raise RuntimeError(
            f"{method} process block {block_index} failed with code {return_code}; "
            f"summary={block_dir / 'block-summary.json'}"
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-revision", default=os.environ.get("VLLM_SWITCH_BENCH_MODEL_REVISION")
    )
    parser.add_argument(
        "--model-config-sha256",
        default=os.environ.get("VLLM_SWITCH_BENCH_MODEL_CONFIG_SHA256"),
    )
    parser.add_argument("--served-model-name", default="bench-model")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workdir", default=str(Path.cwd()))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--vllm-use-v1", default="1")
    parser.add_argument("--cuda-home", default=os.environ.get("CUDA_HOME"))
    parser.add_argument(
        "--enable-server-dev-mode", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--load-format", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--extra-vllm-arg",
        action="append",
        default=[],
        help="Additional raw argument(s) appended to the vLLM API server command. Repeatable; split on whitespace.",
    )
    parser.add_argument("--endpoint", choices=["completion", "chat"], default="completion")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["cold_reload", "sleep_l1", "sleep_l2"],
        choices=["cold_reload", "sleep_l1", "sleep_l2", "cpu_backup", "exact_disk"],
    )
    parser.add_argument("--prompts", nargs="+", default=["short_short"])
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Independent process blocks per method and prompt.",
    )
    parser.add_argument("--cycles-per-process", type=int, default=3)
    parser.add_argument("--ready-timeout-s", type=float, default=240)
    parser.add_argument("--idle-s", type=float, default=2)
    parser.add_argument("--sample-interval-s", type=float, default=0.5)
    parser.add_argument("--cold-max-resident-ratio", type=float, default=0.05)
    parser.add_argument("--cold-min-read-ratio", type=float, default=0.90)
    parser.add_argument("--warm-min-resident-ratio", type=float, default=0.90)
    parser.add_argument("--warm-max-read-ratio", type=float, default=0.10)
    parser.add_argument(
        "--sleep-cpu-backup-pin-memory",
        choices=["auto", "true", "false", "1", "0"],
        default="auto",
        help="Forwarded to VLLM_SLEEP_CPU_BACKUP_PIN_MEMORY for sleep_l1 backup allocation experiments.",
    )
    parser.add_argument("--out-dir", default="results/tmp/vllm-profiling")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.sleep_profile_path = ""
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.cycles_per_process != 3:
        parser.error("--cycles-per-process must be exactly three for first/steady profiling")
    if any(
        value < 0 or value > 1
        for value in (
            args.cold_max_resident_ratio,
            args.cold_min_read_ratio,
            args.warm_min_resident_ratio,
            args.warm_max_read_ratio,
        )
    ):
        parser.error("page-cache validation ratios must be between zero and one")
    unknown_prompts = sorted(set(args.prompts) - set(PROMPTS))
    if unknown_prompts:
        raise SystemExit(f"unknown prompts: {unknown_prompts}; available={sorted(PROMPTS)}")
    return args


def read_gpu_metadata() -> str:
    try:
        return run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            timeout=10,
        ).stdout
    except Exception as exc:
        return f"unavailable: {exc!r}"


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    block_methods = {"sleep_l1", "sleep_l2", "cpu_backup", "exact_disk"}
    if any(method in block_methods for method in args.methods) and len(args.prompts) != 1:
        raise ValueError("repeated sleep/wake profiling requires exactly one prompt")
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = behavior_config(args)
    meta = {
        "schema_version": 1,
        "experiment": "vllm-profiling",
        "system": SYSTEM_NAME,
        "argv": list(argv) if argv is not None else sys.argv,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "methods": args.methods,
        "prompts": args.prompts,
        "repeats": args.repeats,
        "sleep_cpu_backup_pin_memory": args.sleep_cpu_backup_pin_memory,
        "gpu": read_gpu_metadata(),
        "benchmark_git": git_metadata(repository_root()),
        "engine_git": git_metadata(Path(args.workdir)),
        "python": str(Path(args.python).absolute()),
        "engine_runtime": engine_runtime_metadata(args),
        "model_identity": model_metadata(args),
        "behavior_config": config,
        "behavior_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.dry_run:
        print(out_dir)
        return 0

    rows: list[dict[str, Any]] = []
    for method in args.methods:
        if method in block_methods:
            for block_index in range(args.repeats):
                _run_block_driver(args, method, block_index, out_dir)
            continue
        for prompt_name in args.prompts:
            for repeat_index in range(args.repeats):
                row = run_one(args, method, prompt_name, repeat_index, out_dir)
                rows.append(row)
                (out_dir / "summary.json").write_text(
                    json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                write_summary_csv(out_dir / "summary.csv", rows)
                write_sleep_profile_summary_csv(out_dir / "sleep_profile_summary.csv", rows)

    print(out_dir)
    if not rows:
        block_summaries = list(out_dir.glob("*/block-*/block-summary.json"))
        if not block_summaries:
            raise RuntimeError("profiling campaign produced no summary artifacts")
        block_rows = [json.loads(path.read_text(encoding="utf-8")) for path in block_summaries]
        return 0 if all(row.get("ok") is True for row in block_rows) else 2
    return 0 if all(row.get("ok") is True for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
