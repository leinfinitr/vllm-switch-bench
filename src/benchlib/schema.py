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


def flatten_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": row.get("system"),
        "run_id": row.get("run_id"),
        "method": row.get("method"),
        "model": row.get("model"),
        "prompt_name": row.get("prompt_name"),
        "repeat_index": row.get("repeat_index"),
        "ok": row.get("ok"),
        "startup_to_health_s": row.get("startup_to_health_s"),
        "evict_latency_s": (row.get("evict") or {}).get("latency_s"),
        "restore_latency_s": (row.get("restore") or {}).get("latency_s"),
        "ttft_before_s": (row.get("infer_before") or {}).get("ttft_s"),
        "ttft_after_s": (row.get("infer_after") or {}).get("ttft_s"),
        "latency_before_s": (row.get("infer_before") or {}).get("client_latency_s"),
        "latency_after_s": (row.get("infer_after") or {}).get("client_latency_s"),
        "tokens_per_s_before": (row.get("infer_before") or {}).get("approx_tokens_per_s"),
        "tokens_per_s_after": (row.get("infer_after") or {}).get("approx_tokens_per_s"),
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
