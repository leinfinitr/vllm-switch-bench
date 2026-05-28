#!/usr/bin/env python3
"""Offline vLLM lifecycle benchmark using the Python LLM API.

This backend is preferred for Sleep Mode because recent vLLM exposes
`LLM(..., enable_sleep_mode=True)`, `llm.sleep()`, and `llm.wake_up()` directly.
It avoids relying on development-only OpenAI server endpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

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
    proc_rss_mib: int | None = None
    proc_uss_mib: int | None = None
    extra: dict[str, Any] | None = None
    note: str | None = None


def run_cmd(cmd: list[str], timeout: int = 10) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)


def query_gpu() -> dict[str, int | None]:
    try:
        cp = run_cmd([
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        parts = [p.strip() for p in cp.stdout.strip().splitlines()[0].split(",")]
        return {"gpu_used_mib": int(parts[0]), "gpu_free_mib": int(parts[1]), "gpu_util_pct": int(parts[2])}
    except Exception:
        return {"gpu_used_mib": None, "gpu_free_mib": None, "gpu_util_pct": None}


def query_cpu() -> dict[str, int | None]:
    result: dict[str, int | None] = {"cpu_used_mib": None, "cpu_available_mib": None, "proc_rss_mib": None, "proc_uss_mib": None}
    if psutil is None:
        return result
    try:
        vm = psutil.virtual_memory()
        result["cpu_used_mib"] = int(vm.used / 2**20)
        result["cpu_available_mib"] = int(vm.available / 2**20)
        proc = psutil.Process(os.getpid())
        mem = proc.memory_full_info()
        result["proc_rss_mib"] = int(mem.rss / 2**20)
        result["proc_uss_mib"] = int(getattr(mem, "uss", 0) / 2**20)
    except Exception:
        pass
    return result


def make_event(ctx: dict[str, Any], event: str, start_ts: float, extra: dict[str, Any] | None = None, note: str | None = None) -> Event:
    now = time.time()
    data: dict[str, Any] = {}
    data.update(query_gpu())
    data.update(query_cpu())
    return Event(
        run_id=ctx["run_id"], method=ctx["method"], model=ctx["model"], prompt_name=ctx["prompt_name"], repeat_index=ctx["repeat_index"],
        event=event, ts=now, elapsed_s=now - start_ts, extra=extra, note=note, **data
    )


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("w", encoding="utf-8")

    def write(self, event: Event) -> None:
        self.fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def timed_infer(llm: Any, sampling_params: Any, prompt_name: str) -> dict[str, Any]:
    prompt = PROMPTS[prompt_name]["prompt"]
    t0 = time.perf_counter()
    outputs = llm.generate(prompt, sampling_params)
    latency = time.perf_counter() - t0
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    token_ids = getattr(outputs[0].outputs[0], "token_ids", None) if outputs and outputs[0].outputs else None
    token_count = len(token_ids) if token_ids is not None else max(1, len(text.split()))
    return {
        "ok": True,
        "client_latency_s": latency,
        "approx_output_tokens": token_count,
        "approx_tokens_per_s": token_count / latency if latency > 0 else None,
        "output_prefix": text[:120],
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat: list[dict[str, Any]] = []
    for r in rows:
        flat.append({
            "run_id": r.get("run_id"), "method": r.get("method"), "model": r.get("model"), "prompt_name": r.get("prompt_name"),
            "repeat_index": r.get("repeat_index"), "ok": r.get("ok"), "startup_to_ready_s": r.get("startup_to_ready_s"),
            "evict_latency_s": (r.get("evict") or {}).get("latency_s"), "restore_latency_s": (r.get("restore") or {}).get("latency_s"),
            "latency_before_s": (r.get("infer_before") or {}).get("client_latency_s"), "latency_after_s": (r.get("infer_after") or {}).get("client_latency_s"),
            "tokens_per_s_before": (r.get("infer_before") or {}).get("approx_tokens_per_s"),
            "tokens_per_s_after": (r.get("infer_after") or {}).get("approx_tokens_per_s"), "error": r.get("error"),
        })
    if not flat:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        writer.writeheader(); writer.writerows(flat)


def run_one(args: argparse.Namespace, method: str, prompt_name: str, repeat_index: int, out_dir: Path) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{method}_{prompt_name}_{repeat_index}_{uuid.uuid4().hex[:8]}"
    ctx = {"run_id": run_id, "method": method, "model": args.model, "prompt_name": prompt_name, "repeat_index": repeat_index}
    logger = JsonlLogger(out_dir / f"{run_id}.events.jsonl")
    llm: Any | None = None
    start_ts = time.time()
    row: dict[str, Any] = {**ctx, "event_log": str(logger.path)}
    try:
        logger.write(make_event(ctx, "run_start", start_ts))
        t0 = time.perf_counter()
        enable_sleep = method.startswith("sleep_l")
        llm = LLM(
            model=args.model,
            enable_sleep_mode=enable_sleep,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=True,
            enforce_eager=args.enforce_eager,
        )
        row["startup_to_ready_s"] = time.perf_counter() - t0
        logger.write(make_event(ctx, "llm_ready", start_ts, {"startup_to_ready_s": row["startup_to_ready_s"]}))
        sampling_params = SamplingParams(temperature=0, max_tokens=int(PROMPTS[prompt_name]["max_tokens"]))
        row["infer_before"] = timed_infer(llm, sampling_params, prompt_name)
        logger.write(make_event(ctx, "infer_before_end", start_ts, row["infer_before"]))

        logger.write(make_event(ctx, "evict_begin", start_ts))
        t0 = time.perf_counter()
        if method == "cold_reload":
            del llm
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        elif method == "sleep_l1":
            assert llm is not None
            llm.sleep(level=1)
        elif method == "sleep_l2":
            assert llm is not None
            llm.sleep(level=2)
        else:
            raise ValueError(method)
        row["evict"] = {"latency_s": time.perf_counter() - t0}
        logger.write(make_event(ctx, "evict_end", start_ts, row["evict"]))
        time.sleep(args.idle_s)

        logger.write(make_event(ctx, "restore_begin", start_ts))
        t0 = time.perf_counter()
        if method == "cold_reload":
            llm = LLM(
                model=args.model,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype=args.dtype,
                trust_remote_code=True,
                enforce_eager=args.enforce_eager,
            )
        elif method == "sleep_l1":
            assert llm is not None
            llm.wake_up()
        elif method == "sleep_l2":
            assert llm is not None
            llm.wake_up(tags=["weights"])
            llm.collective_rpc("reload_weights")
            llm.wake_up(tags=["kv_cache"])
        row["restore"] = {"latency_s": time.perf_counter() - t0}
        logger.write(make_event(ctx, "restore_end", start_ts, row["restore"]))
        assert llm is not None
        row["infer_after"] = timed_infer(llm, sampling_params, prompt_name)
        logger.write(make_event(ctx, "infer_after_end", start_ts, row["infer_after"]))
        row["ok"] = True
        llm = None
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:
        row["ok"] = False
        row["error"] = repr(exc)
        logger.write(make_event(ctx, "run_error", start_ts, note=repr(exc)))
    finally:
        logger.write(make_event(ctx, "run_end", start_ts))
        logger.close()
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--methods", nargs="+", default=["cold_reload", "sleep_l1", "sleep_l2"])
    p.add_argument("--prompts", nargs="+", default=["short_short", "long_short", "short_long"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--idle-s", type=float, default=2.0)
    p.add_argument("--out-dir", default="results/offline")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps({
        "argv": sys.argv, "created_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "methods": args.methods, "prompts": args.prompts, "repeats": args.repeats,
        "gpu": run_cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]).stdout,
    }, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for method in args.methods:
        for prompt in args.prompts:
            for i in range(args.repeats):
                rows.append(run_one(args, method, prompt, i, out_dir))
                (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
                write_summary_csv(out_dir / "summary.csv", rows)
    print(out_dir)
    return 0 if all(r.get("ok") for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
