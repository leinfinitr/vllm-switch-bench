from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from benchlib.config import collect_repo_metadata, load_baseline3_config
from benchlib.schema import write_summary_csv


def build_serverless_cmd(
    repo: str,
    model: str,
    registered_model_name: str,
    base_url: str,
    prompts: list[str],
    repeats: int,
    out_dir: str,
    methods: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "src/bench_serverless_llm.py",
        "--repo",
        repo,
        "--model",
        model,
        "--registered-model-name",
        registered_model_name,
        "--base-url",
        base_url,
        "--prompts",
        *prompts,
        "--repeats",
        str(repeats),
        "--out-dir",
        out_dir,
    ]
    if methods:
        cmd.extend(["--methods", *methods])
    return cmd


def build_swapserve_cmd(
    repo: str,
    model_name: str,
    base_url: str,
    prompts: list[str],
    repeats: int,
    out_dir: str,
) -> list[str]:
    return [
        sys.executable,
        "src/bench_swapserve_llm.py",
        "--repo",
        repo,
        "--model",
        model_name,
        "--base-url",
        base_url,
        "--prompts",
        *prompts,
        "--repeats",
        str(repeats),
        "--out-dir",
        out_dir,
    ]


def normalize_rows(rows: list[dict[str, Any]], default_system: str) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        item = dict(row)
        item.setdefault("system", default_system)
        normalized.append(item)
    return normalized


def read_summary_rows(run_dir: Path) -> list[dict[str, Any]]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def make_blocker_row(
    system: str,
    method: str,
    model: str,
    prompt_name: str,
    repeat_index: int,
    error: str,
) -> dict[str, Any]:
    return {
        "system": system,
        "method": method,
        "model": model,
        "prompt_name": prompt_name,
        "repeat_index": repeat_index,
        "ok": False,
        "unsupported": True,
        "error": error,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["vllm", "serverless_llm", "swapserve_llm"],
    )
    parser.add_argument("--prompts", nargs="+")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def _parse_run_dir_from_stdout(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _run_adapter(cmd: list[str], workdir: Path) -> tuple[int, str, str, Path | None]:
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        text=True,
        capture_output=True,
        check=False,
    )
    run_dir = _parse_run_dir_from_stdout(proc.stdout)
    return proc.returncode, proc.stdout, proc.stderr, run_dir


def _serverless_model_path(config: dict[str, Any]) -> str:
    system_cfg = config["systems"]["serverless_llm"]
    explicit = system_cfg.get("container_model_path")
    if explicit:
        return str(explicit)
    model = str(config["model"])
    prefix = "/home/ljl/models/"
    if model.startswith(prefix):
        return "/host-models/" + model[len(prefix) :]
    return model


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_baseline3_config(Path(args.config))
    prompts = args.prompts or config.get("prompts", ["short_short"])
    repeats = args.repeats or int(config.get("repeats", 1))
    out_root = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "model": str(config["model"]),
        "systems": args.systems,
        "repos": {},
        "artifacts": {},
    }

    if "vllm" in args.systems:
        latest_vllm = Path("results/qwen2p5_0p5b_clean_hbm_main/20260601_185457")
        if latest_vllm.exists():
            rows.extend(normalize_rows(read_summary_rows(latest_vllm), "vllm"))
            metadata["repos"]["vllm"] = {"source": str(latest_vllm)}
            metadata["artifacts"]["vllm"] = {"run_dir": str(latest_vllm)}
        else:
            rows.append(
                make_blocker_row(
                    "vllm",
                    "import_existing",
                    str(config["model"]),
                    prompts[0],
                    0,
                    "missing existing vllm results",
                )
            )

    if "serverless_llm" in args.systems:
        repo = Path(config["systems"]["serverless_llm"]["repo"])
        metadata["repos"]["serverless_llm"] = collect_repo_metadata(repo)
        base_url = (
            f"http://{config['systems']['serverless_llm']['host']}:"
            f"{config['systems']['serverless_llm']['port']}"
        )
        try:
            health = requests.get(f"{base_url}/health", timeout=5)
            healthy = health.status_code == 200
        except Exception:
            healthy = False
        if healthy:
            rows.append(
                make_blocker_row(
                    "serverless_llm",
                    "delete_register",
                    str(config["model"]),
                    prompts[0],
                    0,
                    "controller delete only removes metadata; router.shutdown remains commented in ServerlessLLM/sllm/controller.py, so delete_register does not reliably free GPU for the next register",
                )
            )
            serverless_out = out_root / "serverless_llm"
            cmd = build_serverless_cmd(
                repo=str(repo),
                model=_serverless_model_path(config),
                registered_model_name=str(
                    config["systems"]["serverless_llm"].get(
                        "registered_model_name", "qwen2p5-0p5b"
                    )
                ),
                base_url=base_url,
                prompts=list(prompts),
                repeats=int(repeats),
                out_dir=str(serverless_out),
                methods=["scale_to_zero_restore"],
            )
            code, stdout, stderr, run_dir = _run_adapter(cmd, Path.cwd())
            metadata["artifacts"]["serverless_llm"] = {
                "command": cmd,
                "returncode": code,
                "stdout": stdout,
                "stderr": stderr,
                "run_dir": str(run_dir) if run_dir else None,
            }
            if run_dir and (run_dir / "summary.json").exists():
                rows.extend(read_summary_rows(run_dir))
            else:
                rows.append(
                    make_blocker_row(
                        "serverless_llm",
                        "scale_to_zero_restore",
                        str(config["model"]),
                        prompts[0],
                        0,
                        f"adapter failed before writing summary (exit {code})",
                    )
                )
        else:
            rows.append(
                make_blocker_row(
                    "serverless_llm",
                    "scale_to_zero_restore",
                    str(config["model"]),
                    prompts[0],
                    0,
                    "ServerlessLLM health endpoint unavailable at configured base URL",
                )
            )

    if "swapserve_llm" in args.systems:
        repo = Path(config["systems"]["swapserve_llm"]["repo"])
        metadata["repos"]["swapserve_llm"] = collect_repo_metadata(repo)
        base_url = (
            f"http://{config['systems']['swapserve_llm']['host']}:"
            f"{config['systems']['swapserve_llm']['port']}"
        )
        podman = shutil.which("podman")
        if podman is None:
            rows.append(
                make_blocker_row(
                    "swapserve_llm",
                    "swapout_swapin",
                    str(config["model"]),
                    prompts[0],
                    0,
                    "podman missing on host; SwapServeLLM router not started",
                )
            )
        else:
            try:
                models = requests.get(f"{base_url}/v1/models", timeout=5)
                if models.status_code == 200:
                    rows.append(
                        make_blocker_row(
                            "swapserve_llm",
                            "swapout_swapin",
                            str(config["model"]),
                            prompts[0],
                            0,
                            "SwapServeLLM runtime reachable but orchestrator start path not automated in this run",
                        )
                    )
                else:
                    rows.append(
                        make_blocker_row(
                            "swapserve_llm",
                            "swapout_swapin",
                            str(config["model"]),
                            prompts[0],
                            0,
                            f"SwapServeLLM router unavailable: HTTP {models.status_code}",
                        )
                    )
            except Exception:
                rows.append(
                    make_blocker_row(
                        "swapserve_llm",
                        "swapout_swapin",
                        str(config["model"]),
                        prompts[0],
                        0,
                        "SwapServeLLM router unavailable and podman-backed runtime not started",
                    )
                )

    (out_root / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary_csv(out_root / "summary.csv", rows)
    (out_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
