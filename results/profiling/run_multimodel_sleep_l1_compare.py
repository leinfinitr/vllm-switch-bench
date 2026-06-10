#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/ljl/research-systems/llm-switch-bench')
PYTHON = ROOT / '.venv/bin/python'
BENCH = ROOT / 'src/bench_vllm_lifecycle.py'
OUT_BASE = ROOT / 'results/profiling_smoke/multimodel_sleep_l1_pin_compare'
MODELS = [
    ('qwen2p5_0p5b', '/home/ljl/models/hf/Qwen2.5-0.5B-Instruct'),
    ('qwen2p5_1p5b', '/home/ljl/models/hf/Qwen2.5-1.5B-Instruct'),
    ('qwen2p5_3b', '/home/ljl/models/hf/Qwen2.5-3B-Instruct'),
]
PIN_MODES = ['true', 'false']


def run_one(model_name: str, model_path: str, pin_mode: str) -> dict:
    out_dir = OUT_BASE / model_name / f'pin_{pin_mode}'
    cmd = [
        sys.executable,
        str(BENCH),
        '--model', model_path,
        '--python', str(PYTHON),
        '--workdir', str(ROOT),
        '--methods', 'sleep_l1',
        '--prompts', 'short_short',
        '--repeats', '3',
        '--port', '0',
        '--idle-s', '0.2',
        '--sample-interval-s', '1',
        '--ready-timeout-s', '300',
        '--sleep-cpu-backup-pin-memory', pin_mode,
        '--out-dir', str(out_dir),
    ]
    env = os.environ.copy()
    env['CUDA_HOME'] = '/home/ljl/cuda-13.0'
    env['PATH'] = f"{ROOT / '.venv/bin'}:/home/ljl/cuda-13.0/bin:" + env.get('PATH', '')
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
    )
    output = proc.stdout.strip()
    return {
        'model_name': model_name,
        'model_path': model_path,
        'pin_mode': pin_mode,
        'returncode': proc.returncode,
        'duration_s': time.time() - started,
        'output': output,
        'result_dir': output.splitlines()[-1] if output else None,
    }


def main() -> int:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_BASE / f'manifest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    results = []
    with manifest_path.open('w', encoding='utf-8') as handle:
        for model_name, model_path in MODELS:
            for pin_mode in PIN_MODES:
                print(f'RUN {model_name} pin={pin_mode}', flush=True)
                try:
                    result = run_one(model_name, model_path, pin_mode)
                except Exception as exc:
                    result = {
                        'model_name': model_name,
                        'model_path': model_path,
                        'pin_mode': pin_mode,
                        'returncode': -1,
                        'duration_s': None,
                        'output': repr(exc),
                        'result_dir': None,
                    }
                result['created_at'] = datetime.now(timezone.utc).isoformat()
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                handle.flush()
                print(json.dumps(result, ensure_ascii=False), flush=True)
    print(str(manifest_path))
    return 0 if all(result['returncode'] == 0 for result in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
