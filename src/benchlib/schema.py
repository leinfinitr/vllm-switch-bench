from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable

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
    system: str
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


SummaryWriter = Callable[[Path, list[dict[str, Any]]], None]


def _nested(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _output_tokens(infer: dict[str, Any]) -> int | None:
    value = _first_present(infer.get("completion_tokens"), infer.get("output_tokens"), infer.get("approx_output_tokens"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tpot(latency_s: Any, ttft_s: Any, output_tokens: int | None) -> float | None:
    if latency_s is None or ttft_s is None or output_tokens is None:
        return None
    try:
        return (float(latency_s) - float(ttft_s)) / max(output_tokens - 1, 1)
    except (TypeError, ValueError):
        return None


def flatten_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    before = _nested(row, "infer_before")
    after = _nested(row, "infer_after")
    output_tokens_before = _output_tokens(before)
    output_tokens_after = _output_tokens(after)
    latency_before = before.get("client_latency_s")
    latency_after = after.get("client_latency_s")
    ttft_before = before.get("ttft_s")
    ttft_after = after.get("ttft_s")
    tpot_before = _tpot(latency_before, ttft_before, output_tokens_before)
    tpot_after = _tpot(latency_after, ttft_after, output_tokens_after)
    ttft_available = ttft_before is not None and ttft_after is not None
    tpot_available = tpot_before is not None and tpot_after is not None

    return {
        "system": row.get("system"),
        "run_id": row.get("run_id"),
        "method": row.get("method"),
        "model": row.get("model"),
        "prompt_name": row.get("prompt_name"),
        "repeat_index": row.get("repeat_index"),
        "ok": row.get("ok"),
        "startup_latency_s": _first_present(row.get("startup_latency_s"), row.get("startup_to_health_s")),
        "memory_gpu_used_ready_mib": row.get("memory_gpu_used_ready_mib"),
        "memory_cpu_used_ready_mib": row.get("memory_cpu_used_ready_mib"),
        "memory_gpu_used_evict_mib": row.get("memory_gpu_used_evict_mib"),
        "memory_cpu_used_evict_mib": row.get("memory_cpu_used_evict_mib"),
        "evict_latency_s": _nested(row, "evict").get("latency_s"),
        "restore_latency_s": _nested(row, "restore").get("latency_s"),
        "restore_latency_estimated": bool(row.get("restore_latency_estimated", False)),
        "ttft_before_s": ttft_before,
        "ttft_after_s": ttft_after,
        "latency_before_s": latency_before,
        "latency_after_s": latency_after,
        "output_tokens_before": output_tokens_before,
        "output_tokens_after": output_tokens_after,
        "tpot_before_s": tpot_before,
        "tpot_after_s": tpot_after,
        "ttft_available": bool(row.get("ttft_available", ttft_available)),
        "tpot_available": bool(row.get("tpot_available", tpot_available)),
        "error": row.get("error"),
    }


def write_summary_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    flat_rows = [flatten_summary_row(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()) if flat_rows else [])
        if flat_rows:
            writer.writeheader()
            writer.writerows(flat_rows)
