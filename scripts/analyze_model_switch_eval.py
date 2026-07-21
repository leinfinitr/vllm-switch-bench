#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/model_switch_eval"
RAW = RESULTS / "raw"
LATEST = RESULTS / "latest"


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p95": pct(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def latest(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def repeated(system: str, model: str) -> dict:
    path = latest(
        f"results/model_switch_eval/raw/{system}/{model}/*/repeated_sleep_l1_summary.json"
    )
    data = json.loads(path.read_text())
    if not data["ok"]:
        raise ValueError(f"failed lifecycle: {path}")
    rows = data["steps"][1:]
    return {
        "source": str(path.relative_to(ROOT)),
        "success": sum(bool(r["ok"]) for r in rows),
        "attempted": len(rows),
        "sleep_s": stats([r["sleep_latency_s"] for r in rows]),
        "wake_s": stats([r["wake_latency_s"] for r in rows]),
        "switch_s": stats(
            [r["sleep_latency_s"] + r["wake_latency_s"] for r in rows]
        ),
    }


def swapserve(model: str) -> dict:
    path = latest(
        f"results/model_switch_eval/raw/swapserve/{model}/runs/*/summary.json"
    )
    rows = json.loads(path.read_text())
    return {
        "source": str(path.relative_to(ROOT)),
        "success": sum(bool(r["ok"]) for r in rows),
        "attempted": len(rows),
        "sleep_s": stats([r["evict"]["latency_s"] for r in rows]),
        "wake_s": stats([r["restore"]["latency_s"] for r in rows]),
        "switch_s": stats(
            [r["evict"]["latency_s"] + r["restore"]["latency_s"] for r in rows]
        ),
    }


def serverless() -> dict:
    path = RAW / "serverlessllm/qwen-1.5b/steady-cycles.json"
    rows = json.loads(path.read_text())
    return {
        "source": str(path.relative_to(ROOT)),
        "success": sum(bool(r["ok"]) for r in rows),
        "attempted": len(rows),
        "sleep_s": stats([r["sleep_s"] for r in rows]),
        "wake_s": stats([r["wake_s"] for r in rows]),
        "switch_s": stats([r["switch_s"] for r in rows]),
    }


def llama_lifecycle(model: str) -> dict:
    path = RAW / "llama-swap/lifecycle/switches.json"
    rows = json.loads(path.read_text())
    values = [
        r["switch_time_s"]
        for r in rows
        if r["target_model"] == model and r["switch_time_s"] is not None
    ]
    return {
        "source": str(path.relative_to(ROOT)),
        "success": len(values),
        "attempted": len(values),
        "sleep_s": None,
        "wake_s": None,
        "switch_s": stats(values),
        "definition_note": "request-visible terminate-current + start-target + first complete streamed inference; upstream API does not expose separate sleep/wake phases",
    }


def trace(system: str, files: list[Path]) -> dict:
    runs = []
    for path in sorted(files):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        failed = [
            r
            for r in rows
            if r.get("error")
            or r.get("stream_done") is not True
            or not str(r.get("output_text") or "").strip()
        ]
        if failed:
            raise ValueError(f"strict failures in retained trace {path}: {len(failed)}")
        elapsed = max(
            r["client_dispatch_offset_s"] + r["completion_latency_ms"] / 1000
            for r in rows
        )
        runs.append(
            {
                "source": str(path.relative_to(ROOT)),
                "requests": len(rows),
                "elapsed_s": elapsed,
                "semantic_ttft_median_ms": statistics.median(
                    r["semantic_ttft_ms"] for r in rows
                ),
                "completion_median_ms": statistics.median(
                    r["completion_latency_ms"] for r in rows
                ),
            }
        )
    return {
        "system": system,
        "runs": runs,
        "success_requests": sum(r["requests"] for r in runs),
        "attempted_requests": sum(r["requests"] for r in runs),
        "elapsed_s": stats([r["elapsed_s"] for r in runs]),
        "semantic_ttft_run_median_ms": stats(
            [r["semantic_ttft_median_ms"] for r in runs]
        ),
    }


def main() -> None:
    lifecycle = {
        "proposed": {
            "qwen-1.5b": repeated("proposed", "qwen-1.5b"),
            "qwen-3b": repeated("proposed", "qwen-3b"),
        },
        "vllm-stock-l1": {
            "qwen-1.5b": repeated("vllm-stock", "qwen-1.5b"),
            "qwen-3b": repeated("vllm-stock", "qwen-3b"),
        },
        "swapserve": {
            "qwen-1.5b": swapserve("qwen-1.5b"),
            "qwen-3b": swapserve("qwen-3b"),
        },
        "serverlessllm": {"qwen-1.5b": serverless()},
        "llama-swap": {
            "qwen-1.5b": llama_lifecycle("qwen-1.5b"),
            "qwen-3b": llama_lifecycle("qwen-3b"),
        },
    }
    proposed_files = list((RAW / "proposed/e2e-published").glob("*.jsonl"))
    llama_files = list((RAW / "llama-swap/e2e").glob("*.jsonl"))
    e2e = {}
    for workload in ("alternating", "burst"):
        e2e[workload] = {
            "proposed": trace(
                "proposed", [p for p in proposed_files if workload in p.name]
            ),
            "llama-swap": trace(
                "llama-swap", [p for p in llama_files if workload in p.name]
            ),
        }
    summary = {
        "scope": "exploratory single-host RTX 3080 comparison",
        "switch_definition": "sleep/evict/swap-out through post-condition + wake/restore/swap-in through post-condition",
        "lifecycle": lifecycle,
        "e2e": e2e,
        "caveats": [
            "Lifecycle methods preserve different amounts of process/CUDA/CPU state, so the common sum is operational rather than mechanism-equivalent.",
            "Proposed and vLLM lifecycle medians exclude each model's first D2H backup-population step and use five steady-state cycles.",
            "llama-swap lacks explicit lifecycle APIs; its lifecycle row is request-visible transition time and is not decomposed.",
            "ServerlessLLM 3B failed after a startup error left the scheduler GPU reservation at zero; no 3B latency is reported.",
            "E2E retains only strict-success completed runs. Proposed has two alternating runs and one burst run; a later burst run hit a lifecycle hang and is retained outside the curated set as a failed diagnostic.",
        ],
    }
    LATEST.mkdir(parents=True, exist_ok=True)
    (LATEST / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    excluded_names = {"checksums.json", "feishu-report.md"}
    files = sorted(
        p
        for p in RESULTS.rglob("*")
        if p.is_file()
        and p.name not in excluded_names
        and not p.relative_to(RESULTS).as_posix().startswith("tmp/")
    )
    manifest = []
    for path in files:
        data = path.read_bytes()
        manifest.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (LATEST / "checksums.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(LATEST / "summary.json")


if __name__ == "__main__":
    main()
