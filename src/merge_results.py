#!/usr/bin/env python3
"""Merge benchmark result directories produced by the same harness."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", type=Path)
    p.add_argument("inputs", nargs="+", type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for inp in args.inputs:
        rows.extend(load_json(inp / "summary.json"))
        meta_path = inp / "metadata.json"
        metadata.append(json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"input": str(inp)})

    # Prefer later replacement rows when rerunning the same method/prompt/repeat.
    dedup: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["method"], row["prompt_name"], int(row["repeat_index"]))
        dedup[key] = row
    merged = [dedup[k] for k in sorted(dedup)]

    (args.out_dir / "summary.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "metadata.json").write_text(json.dumps({"inputs": [str(p) for p in args.inputs], "input_metadata": metadata}, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "run_id", "method", "model", "prompt_name", "repeat_index", "ok", "startup_to_health_s",
        "evict_latency_s", "restore_latency_s", "ttft_before_s", "ttft_after_s",
        "latency_before_s", "latency_after_s", "tokens_per_s_before", "tokens_per_s_after", "error",
    ]
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in merged:
            writer.writerow({
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
    print(args.out_dir)
    print(f"merged_rows={len(merged)}")
    return 0 if all(row.get("ok") for row in merged) else 2


if __name__ == "__main__":
    raise SystemExit(main())
