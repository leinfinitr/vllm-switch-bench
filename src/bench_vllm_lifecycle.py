#!/usr/bin/env python3
"""Benchmark model lifecycle transitions for local vLLM sleep/cold-reload tests.

The harness is intentionally self-contained and conservative for the shared IPADS
server: it does not drop Linux page cache, does not change system-wide settings,
and records enough metadata for later manual inspection.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import psutil
except Exception:  # pragma: no cover - handled at runtime
    psutil = None  # type: ignore[assignment]

try:
    import requests
except Exception as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("requests is required to run this benchmark") from exc


PROMPTS: dict[str, dict[str, Any]] = {
    "short_short": {
        "prompt": "Give one concise sentence about why GPU memory matters for LLM serving.",
        "max_tokens": 32,
    },
    "long_short": {
        "prompt": "\n".join([
            "You are analyzing an LLM serving system. Summarize the main bottleneck in one sentence.",
            *(f"Context line {i}: weights, KV cache, CUDA graphs, CPU RAM, and storage affect switching." for i in range(1, 45)),
        ]),
        "max_tokens": 24,
    },
    "short_long": {
        "prompt": "List practical measurements for evaluating LLM model switching.",
        "max_tokens": 160,
    },
}


@dataclass
class Event:
    run_id: str
    method: str
    model: str
    prompt_name: str
    repeat_index: int
    event: str
    ts: float
    elapsed_s: float
    gpu_used_mib: int | None = None
    gpu_free_mib: int | None = None
    gpu_util_pct: int | None = None
    cpu_used_mib: int | None = None
    cpu_available_mib: int | None = None
    proc_pid: int | None = None
    proc_rss_mib: int | None = None
    proc_uss_mib: int | None = None
    note: str | None = None
    extra: dict[str, Any] | None = None


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, event: Event) -> None:
        self._fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def run_cmd(cmd: list[str], timeout: int = 30, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=check)


def query_gpu() -> dict[str, int | None]:
    try:
        cp = run_cmd([
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], timeout=10)
        parts = [p.strip() for p in cp.stdout.strip().splitlines()[0].split(",")]
        return {"gpu_used_mib": int(parts[0]), "gpu_free_mib": int(parts[1]), "gpu_util_pct": int(parts[2])}
    except Exception:
        return {"gpu_used_mib": None, "gpu_free_mib": None, "gpu_util_pct": None}


def query_cpu(pid: int | None) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "cpu_used_mib": None,
        "cpu_available_mib": None,
        "proc_rss_mib": None,
        "proc_uss_mib": None,
    }
    if psutil is None:
        return result
    try:
        vm = psutil.virtual_memory()
        result["cpu_used_mib"] = int(vm.used / 2**20)
        result["cpu_available_mib"] = int(vm.available / 2**20)
    except Exception:
        pass
    if pid:
        try:
            p = psutil.Process(pid)
            info = p.memory_full_info()
            result["proc_rss_mib"] = int(info.rss / 2**20)
            result["proc_uss_mib"] = int(getattr(info, "uss", 0) / 2**20)
        except Exception:
            pass
    return result


def make_event(ctx: dict[str, Any], event: str, start_ts: float, pid: int | None = None, note: str | None = None, extra: dict[str, Any] | None = None) -> Event:
    now = time.time()
    metrics = {}
    metrics.update(query_gpu())
    metrics.update(query_cpu(pid))
    return Event(
        run_id=ctx["run_id"],
        method=ctx["method"],
        model=ctx["model"],
        prompt_name=ctx["prompt_name"],
        repeat_index=ctx["repeat_index"],
        event=event,
        ts=now,
        elapsed_s=now - start_ts,
        proc_pid=pid,
        note=note,
        extra=extra,
        **metrics,
    )


class Sampler:
    def __init__(self, logger: JsonlLogger, ctx: dict[str, Any], start_ts: float, get_pid, interval_s: float = 0.5):
        self.logger = logger
        self.ctx = ctx
        self.start_ts = start_ts
        self.get_pid = get_pid
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self):
        while not self._stop.is_set():
            self.logger.write(make_event(self.ctx, "sample", self.start_ts, self.get_pid()))
            time.sleep(self.interval_s)


def wait_http_ok(url: str, timeout_s: float) -> float:
    sess = requests.Session()
    sess.trust_env = False
    start = time.perf_counter()
    last = None
    while time.perf_counter() - start < timeout_s:
        try:
            r = sess.get(url, timeout=2)
            if r.status_code == 200:
                return time.perf_counter() - start
            last = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as exc:
            last = repr(exc)
        time.sleep(0.5)
    raise TimeoutError(f"{url} not ready after {timeout_s}s; last={last}")


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
            r = sess.get(url, timeout=2)
            if r.status_code == 200:
                return time.perf_counter() - start
            last = f"HTTP {r.status_code}: {r.text[:120]}"
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
    if args.compat_sitecustomize:
        compat_path = str(Path(args.compat_sitecustomize).resolve().parent)
        env["PYTHONPATH"] = compat_path + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONSTARTUP"] = str(Path(args.compat_sitecustomize).resolve())
        env["VLLM_BENCH_COMPAT_SITECUSTOMIZE"] = str(Path(args.compat_sitecustomize).resolve())
    log_f = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True, env=env, cwd=args.workdir)
    return proc


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
    t0 = time.perf_counter()
    first_chunk_s = None
    chunks = []
    status = None
    error = None
    try:
        with sess.post(url, json=payload, stream=True, timeout=300) as r:
            status = r.status_code
            if r.status_code != 200:
                text = r.text[:500]
                return {"ok": False, "status": status, "error": text, "client_latency_s": time.perf_counter() - t0}
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
                if first_chunk_s is None:
                    first_chunk_s = time.perf_counter() - t0
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    chunks.append(data)
    except Exception as exc:
        error = repr(exc)
    total = time.perf_counter() - t0
    text_parts = []
    completion_tokens = None
    for c in chunks:
        try:
            obj = json.loads(c)
            choice = obj.get("choices", [{}])[0]
            delta = choice.get("delta") or {}
            if "content" in delta and delta["content"]:
                text_parts.append(delta["content"])
            if "text" in choice and choice["text"]:
                text_parts.append(choice["text"])
            usage = obj.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
        except Exception:
            pass
    output = "".join(text_parts)
    approx_tokens = completion_tokens or max(1, len(output.split()))
    return {
        "ok": error is None,
        "status": status,
        "error": error,
        "ttft_s": first_chunk_s,
        "client_latency_s": total,
        "approx_output_tokens": approx_tokens,
        "approx_tokens_per_s": (approx_tokens / total) if total > 0 else None,
        "output_prefix": output[:120],
    }


def call_sleep(args: argparse.Namespace, level: int) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/sleep?level={level}"
    t0 = time.perf_counter()
    try:
        r = post_json(url, {}, timeout_s=300)
        return {"ok": r.status_code == 200, "status": r.status_code, "latency_s": time.perf_counter() - t0, "body": r.text[:500]}
    except Exception as exc:
        return {"ok": False, "latency_s": time.perf_counter() - t0, "error": repr(exc)}


def call_wake(args: argparse.Namespace, tags: list[str] | None = None) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/wake_up"
    if tags:
        url += "?" + "&".join(f"tags={tag}" for tag in tags)
    t0 = time.perf_counter()
    try:
        r = post_json(url, {}, timeout_s=300)
        return {"ok": r.status_code == 200, "status": r.status_code, "latency_s": time.perf_counter() - t0, "body": r.text[:500]}
    except Exception as exc:
        return {"ok": False, "latency_s": time.perf_counter() - t0, "error": repr(exc)}


def call_rpc(args: argparse.Namespace, method: str, kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/collective_rpc"
    payload: dict[str, Any] = {"method": method}
    if kwargs:
        payload["kwargs"] = kwargs
    t0 = time.perf_counter()
    try:
        r = post_json(url, payload, timeout_s=300)
        return {"ok": r.status_code == 200, "status": r.status_code, "latency_s": time.perf_counter() - t0, "body": r.text[:500]}
    except Exception as exc:
        return {"ok": False, "latency_s": time.perf_counter() - t0, "error": repr(exc)}


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
    ctx = {"run_id": run_id, "method": method, "model": args.model, "prompt_name": prompt_name, "repeat_index": repeat_index}
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
        enable_sleep = method.startswith("sleep_l")
        args.enable_sleep_mode = enable_sleep
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
                summary["evict"] = {"latency_s": stop_process(proc)}
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
                summary["restore"] = {"latency_s": restore_s}
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
                    summary["restore"]["steps"] = {"wake_weights": wake_weights, "reload_weights": reload_weights, "wake_kv_cache": wake_kv}
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


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    for r in rows:
        flat_rows.append({
            "run_id": r.get("run_id"),
            "method": r.get("method"),
            "model": r.get("model"),
            "prompt_name": r.get("prompt_name"),
            "repeat_index": r.get("repeat_index"),
            "ok": r.get("ok"),
            "startup_to_health_s": r.get("startup_to_health_s"),
            "evict_latency_s": (r.get("evict") or {}).get("latency_s"),
            "restore_latency_s": (r.get("restore") or {}).get("latency_s"),
            "ttft_before_s": (r.get("infer_before") or {}).get("ttft_s"),
            "ttft_after_s": (r.get("infer_after") or {}).get("ttft_s"),
            "latency_before_s": (r.get("infer_before") or {}).get("client_latency_s"),
            "latency_after_s": (r.get("infer_after") or {}).get("client_latency_s"),
            "tokens_per_s_before": (r.get("infer_before") or {}).get("approx_tokens_per_s"),
            "tokens_per_s_after": (r.get("infer_after") or {}).get("approx_tokens_per_s"),
            "error": r.get("error"),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()) if flat_rows else [])
        if flat_rows:
            writer.writeheader()
            writer.writerows(flat_rows)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--served-model-name", default="bench-model")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--workdir", default=str(Path.cwd()))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18000)
    p.add_argument("--cuda-visible-devices", default="0")
    p.add_argument("--vllm-use-v1", default="1")
    p.add_argument("--cuda-home", default="/home/ljl/cuda-13.0")
    p.add_argument("--enable-server-dev-mode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--load-format", default="")
    p.add_argument("--quantization", default="")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--compat-sitecustomize", default="")
    p.add_argument("--endpoint", choices=["completion", "chat"], default="completion")
    p.add_argument("--methods", nargs="+", default=["cold_reload", "sleep_l1", "sleep_l2"])
    p.add_argument("--prompts", nargs="+", default=["short_short", "long_short", "short_long"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--ready-timeout-s", type=float, default=240)
    p.add_argument("--idle-s", type=float, default=2)
    p.add_argument("--sample-interval-s", type=float, default=0.5)
    p.add_argument("--out-dir", default="benchmark/model-switching/results")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    unknown_prompts = sorted(set(args.prompts) - set(PROMPTS))
    if unknown_prompts:
        raise SystemExit(f"unknown prompts: {unknown_prompts}; available={sorted(PROMPTS)}")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "argv": sys.argv,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "methods": args.methods,
        "prompts": args.prompts,
        "repeats": args.repeats,
        "gpu": run_cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"], timeout=10).stdout,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dry_run:
        print(out_dir)
        return 0
    rows: list[dict[str, Any]] = []
    for method in args.methods:
        for prompt_name in args.prompts:
            for i in range(args.repeats):
                row = run_one(args, method, prompt_name, i, out_dir)
                rows.append(row)
                (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
                write_summary_csv(out_dir / "summary.csv", rows)
    print(out_dir)
    return 0 if all(r.get("ok") for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
