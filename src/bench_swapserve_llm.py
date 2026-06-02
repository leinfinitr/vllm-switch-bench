from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from benchlib.schema import PROMPTS, write_summary_csv

SYSTEM_NAME = "swapserve_llm"
STAGE_PATTERNS = {
    "swapout.get_gpu_pids_s": re.compile(r"SwapOut Stage\] Get GPU PIDs took ([0-9.]+)(ms|s)"),
    "swapout.unload_model_s": re.compile(r"SwapOut Stage\] Unload model took ([0-9.]+)(ms|s)"),
    "swapout.checkpoint_gpu_threads_s": re.compile(r"SwapOut Stage\] Checkpoint GPU threads took ([0-9.]+)(ms|s)"),
    "swapout.pause_container_s": re.compile(r"SwapOut Stage\] Pause container took ([0-9.]+)(ms|s)"),
    "swapin.resume_container_s": re.compile(r"SwapIn Stage\] resumeContainer completed in ([0-9.]+)(ms|s)"),
    "swapin.cuda_restore_s": re.compile(r"SwapIn Stage\] cuda\.RestorePID.*?([0-9.]+)(ms|s)"),
    "swapin.wait_for_server_s": re.compile(r"SwapIn Stage\] WaitForServer completed in ([0-9.]+)(ms|s)"),
    "swapin.load_model_s": re.compile(r"SwapIn Stage\] LoadModel completed in ([0-9.]+)(ms|s)"),
}


def parse_duration(value: str, unit: str) -> float:
    amount = float(value)
    return amount / 1000.0 if unit == "ms" else amount



def parse_swapserve_stage_logs(text: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, pattern in STAGE_PATTERNS.items():
        match = pattern.search(text)
        if match:
            parsed[key] = parse_duration(match.group(1), match.group(2))
    return parsed



def auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}



def infer(base_url: str, model_name: str, prompt_name: str, api_key: str | None = None) -> dict[str, Any]:
    prompt = PROMPTS[prompt_name]
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt["prompt"]}],
        "max_tokens": int(prompt["max_tokens"]),
        "temperature": 0,
    }
    started_at = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers=auth_headers(api_key) or None,
        timeout=300,
    )
    total = time.perf_counter() - started_at
    if response.status_code != 200:
        return {
            "ok": False,
            "status": response.status_code,
            "error": response.text[:500],
            "client_latency_s": total,
        }
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or max(1, len(content.split()))
    return {
        "ok": True,
        "status": response.status_code,
        "ttft_s": None,
        "client_latency_s": total,
        "approx_output_tokens": completion_tokens,
        "approx_tokens_per_s": completion_tokens / total if total > 0 else None,
        "output_prefix": content[:120],
    }



def _request(
    client,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
):
    kwargs: dict[str, Any] = {"timeout": 300}
    if payload is not None:
        kwargs["json"] = payload
    if api_key:
        kwargs["headers"] = auth_headers(api_key)
    return client.request(method, url, **kwargs)



def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0



def _read_from(path: Path, offset: int) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        return handle.read()



def run_swapout_swapin(
    base_url: str,
    client,
    model_name: str,
    prompt_name: str,
    repeat_index: int,
    stage_log_text: str = "",
    log_dir: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "system": SYSTEM_NAME,
        "method": "swapout_swapin",
        "model": model_name,
        "prompt_name": prompt_name,
        "repeat_index": repeat_index,
    }

    swapout_path = Path(log_dir) / "swapout.log" if log_dir else None
    swapin_path = Path(log_dir) / "swapin.log" if log_dir else None
    swapout_offset = _file_size(swapout_path) if swapout_path else 0
    swapin_offset = _file_size(swapin_path) if swapin_path else 0

    models_response = _request(client, "GET", f"{base_url}/v1/models", api_key=api_key)
    if models_response.status_code != 200:
        row["ok"] = False
        row["error"] = f"models failed: {models_response.status_code} {models_response.text[:200]}"
        return row

    row["infer_before"] = infer(base_url, model_name, prompt_name, api_key=api_key)

    evict_start = time.perf_counter()
    swapout_response = _request(client, "POST", f"{base_url}/api/swapout", {"model": model_name}, api_key=api_key)
    row["evict"] = {
        "ok": swapout_response.status_code == 200,
        "status": swapout_response.status_code,
        "latency_s": time.perf_counter() - evict_start,
        "body": swapout_response.text[:200],
    }
    if swapout_response.status_code != 200:
        row["ok"] = False
        row["error"] = f"swapout failed: {swapout_response.status_code} {swapout_response.text[:200]}"
        return row

    restore_start = time.perf_counter()
    swapin_response = _request(client, "POST", f"{base_url}/api/swapin", {"model": model_name}, api_key=api_key)
    row["restore"] = {
        "ok": swapin_response.status_code == 200,
        "status": swapin_response.status_code,
        "latency_s": time.perf_counter() - restore_start,
        "body": swapin_response.text[:200],
    }
    if swapin_response.status_code != 200:
        row["ok"] = False
        row["error"] = f"swapin failed: {swapin_response.status_code} {swapin_response.text[:200]}"
        return row

    row["infer_after"] = infer(base_url, model_name, prompt_name, api_key=api_key)

    if stage_log_text:
        row["stage_breakdown"] = parse_swapserve_stage_logs(stage_log_text)
    elif log_dir:
        appended = "\n".join(
            part
            for part in (
                _read_from(swapout_path, swapout_offset) if swapout_path else "",
                _read_from(swapin_path, swapin_offset) if swapin_path else "",
            )
            if part
        )
        if appended:
            row["stage_breakdown"] = parse_swapserve_stage_logs(appended)

    row["ok"] = bool(row["infer_before"].get("ok")) and bool(row["infer_after"].get("ok"))
    return row



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--log-dir")
    parser.add_argument("--methods", nargs="+", default=["swapout_swapin"])
    parser.add_argument("--prompts", nargs="+", default=["short_short"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-dir", default="results/tmp/swapserve_llm")
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    client = requests.Session()
    rows: list[dict[str, Any]] = []
    for prompt_name in args.prompts:
        for repeat_index in range(args.repeats):
            rows.append(
                run_swapout_swapin(
                    args.base_url,
                    client,
                    args.model,
                    prompt_name,
                    repeat_index,
                    log_dir=args.log_dir,
                    api_key=args.api_key,
                )
            )
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(out_dir / "summary.csv", rows)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "system": SYSTEM_NAME,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "repo": args.repo,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out_dir)
    return 0 if all(row.get("ok") for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
