from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import requests


def gpu_used() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return sum(int(line.strip()) for line in out.splitlines() if line.strip())


def request(base: str, model: str) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
    }
    started = time.perf_counter()
    text = ""
    done = False
    with requests.post(
        f"{base}/v1/chat/completions", json=body, stream=True, timeout=300
    ) as response:
        status = response.status_code
        for line in response.iter_lines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                continue
            obj = json.loads(data)
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            choice = (obj.get("choices") or [{}])[0]
            text += str((choice.get("delta") or {}).get("content") or choice.get("text") or "")
    return {
        "model": model,
        "latency_s": time.perf_counter() - started,
        "status": status,
        "done": done,
        "output": text,
        "ok": 200 <= status < 300 and done and bool(text.strip()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18100")
    parser.add_argument("--models", nargs=2, required=True)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    current = None
    for cycle in range(args.cycles):
        for target in args.models:
            # llama-swap has no explicit sleep endpoint. Its request-visible model
            # transition is terminate current + start target + first inference.
            result = request(args.base_url, target)
            result.update(
                {
                    "cycle": cycle,
                    "source_model": current,
                    "target_model": target,
                    "switch_time_s": result["latency_s"] if current is not None else None,
                    "gpu_used_mib": gpu_used(),
                    "definition": "terminate current + start target + first streamed inference",
                }
            )
            rows.append(result)
            if not result["ok"]:
                break
            current = target
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "rows": len(rows),
        "ok": sum(bool(row["ok"]) for row in rows),
        "switch_median_s": statistics.median(
            row["switch_time_s"] for row in rows if row["switch_time_s"] is not None
        ),
    }))
    return 0 if all(row["ok"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
