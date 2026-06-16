#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from benchlib.schema import PROMPTS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure repeated offline sleep_l1 to validate vLLM CPU backup pool reuse."
    )
    parser.add_argument("--model", default="/home/ljl/models/hf/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out-dir", default="results/profiling/phase1_pinned_pool")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="short_short")
    parser.add_argument("--cycles", type=int, default=2)
    return parser.parse_args(argv)


def load_profile_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["VLLM_USE_V1"] = "1"
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"phase1_offline_repeated_sleep_l1_{int(time.time())}"
    profile_path = out_dir / f"{run_id}.sleep_profile.jsonl"
    os.environ["VLLM_SLEEP_PROFILE_PATH"] = str(profile_path.resolve())

    summary: dict[str, Any] = {
        "model": args.model,
        "cycles": args.cycles,
        "profile_path": str(profile_path),
    }
    try:
        from vllm import LLM, SamplingParams

        prompt_spec = PROMPTS[args.prompt]
        prompt = prompt_spec["prompt"]
        sampling_params = SamplingParams(
            max_tokens=prompt_spec["max_tokens"], temperature=0.0, seed=0
        )
        started = time.perf_counter()
        llm = LLM(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            enable_sleep_mode=True,
        )
        summary["startup_latency_s"] = time.perf_counter() - started
        before = llm.generate(prompt, sampling_params, use_tqdm=False)
        before_text = before[0].outputs[0].text
        summary["before_text"] = before_text

        cycles = []
        for cycle in range(args.cycles):
            sleep_started = time.perf_counter()
            llm.sleep(level=1)
            sleep_latency_s = time.perf_counter() - sleep_started
            wake_started = time.perf_counter()
            llm.wake_up()
            wake_latency_s = time.perf_counter() - wake_started
            after = llm.generate(prompt, sampling_params, use_tqdm=False)
            after_text = after[0].outputs[0].text
            cycles.append(
                {
                    "cycle": cycle,
                    "sleep_latency_s": sleep_latency_s,
                    "wake_latency_s": wake_latency_s,
                    "after_text": after_text,
                    "matches_before": after_text == before_text,
                }
            )
            if after_text != before_text:
                raise RuntimeError(f"output changed after cycle {cycle}")
        summary["cycles"] = cycles

        events = load_profile_events(profile_path)
        sleep_events = [event for event in events if event.get("phase") == "allocator_sleep"]
        wake_events = [event for event in events if event.get("phase") == "allocator_wake_up"]
        summary["sleep_events"] = sleep_events
        summary["wake_events"] = wake_events
        summary["sleep_latencies_s"] = [event.get("latency_s") for event in sleep_events]
        summary["cpu_backup_alloc_s"] = [event.get("cpu_backup_alloc_s") for event in sleep_events]
        summary["pool_hit_counts"] = [event.get("cpu_backup_pool_hit_count") for event in sleep_events]
        summary["pool_miss_counts"] = [event.get("cpu_backup_pool_miss_count") for event in sleep_events]
        if len(sleep_events) >= 2:
            first = sleep_events[0]
            later = sleep_events[1:]
            summary["first_cpu_backup_alloc_s"] = first.get("cpu_backup_alloc_s")
            summary["later_cpu_backup_alloc_s_mean"] = statistics.mean(
                float(event.get("cpu_backup_alloc_s") or 0.0) for event in later
            )
            summary["later_pool_hit_count_total"] = sum(
                int(event.get("cpu_backup_pool_hit_count") or 0) for event in later
            )
            summary["later_pool_miss_count_total"] = sum(
                int(event.get("cpu_backup_pool_miss_count") or 0) for event in later
            )
        summary["ok"] = True
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = repr(exc)
    finally:
        summary_path = out_dir / "phase1_repeated_sleep_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(summary_path)
    return 0 if summary.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
