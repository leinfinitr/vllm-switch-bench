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
OUT_BASE = ROOT / 'results/profiling_smoke/multimodel_sleep_l1_pin_compare/qwen2p5_3b_retry_gpu085'
MODEL = '/home/ljl/models/hf/Qwen2.5-3B-Instruct'


def run(pin_mode: str) -> dict:
    out_dir = OUT_BASE / f'pin_{pin_mode}'
    cmd = [
        sys.executable,
        str(BENCH),
        '--model', MODEL,
        '--python', str(PYTHON),
        '--workdir', str(ROOT),
        '--methods', 'sleep_l1',
        '--prompts', 'short_short',
        '--repeats', '3',
        '--port', '0',
        '--idle-s', '0.2',
        '--sample-interval-s', '1',
        '--ready-timeout-s', '360',
        '--gpu-memory-utilization', '0.85',
        '--sleep-cpu-backup-pin-memory', pin_mode,
        '--out-dir', str(out_dir),
    ]
    env = os.environ.copy()
    env['CUDA_HOME'] = '/home/ljl/cuda-13.0'
    env['PATH'] = f"{ROOT / '.venv/bin'}:/home/ljl/cuda-13.0/bin:" + env.get('PATH', '')
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
    output = proc.stdout.strip()
    return {
        'model_name': 'qwen2p5_3b',
        'model_path': MODEL,
        'pin_mode': pin_mode,
        'gpu_memory_utilization': 0.85,
        'returncode': proc.returncode,
        'duration_s': time.time() - started,
        'output': output,
        'result_dir': output.splitlines()[-1] if output else None,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    manifest = OUT_BASE / f'manifest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    results = []
    with manifest.open('w', encoding='utf-8') as f:
        for pin_mode in ['true', 'false']:
            print(f'RUN 3B pin={pin_mode}', flush=True)
            result = run(pin_mode)
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(manifest)
    return 0 if all(r['returncode'] == 0 for r in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
