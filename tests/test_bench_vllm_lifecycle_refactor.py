from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
MODULE_PATH = SRC / "bench_vllm_lifecycle.py"
spec = importlib.util.spec_from_file_location("bench_vllm_lifecycle", MODULE_PATH)
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


def test_vllm_harness_uses_benchlib_prompt_catalog():
    from benchlib.schema import PROMPTS

    assert bench.PROMPTS is PROMPTS


def test_vllm_harness_uses_benchlib_summary_writer(tmp_path):
    out = tmp_path / "summary.csv"
    bench.write_summary_csv(
        out,
        [
            {
                "system": "vllm",
                "run_id": "r1",
                "method": "sleep_l1",
                "model": "m",
                "prompt_name": "short_short",
                "repeat_index": 0,
                "ok": True,
                "infer_before": {"ttft_s": 0.1, "client_latency_s": 0.2, "approx_tokens_per_s": 3.0},
                "infer_after": {"ttft_s": 0.2, "client_latency_s": 0.3, "approx_tokens_per_s": 4.0},
            }
        ],
    )
    text = out.read_text()
    assert text.splitlines()[0].startswith("system,")
    assert ",vllm," in text or text.endswith(",vllm") or "vllm" in text


def test_vllm_main_dry_run_metadata_records_system(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "run_cmd", lambda *args, **kwargs: type("CP", (), {"stdout": "0,FakeGPU,10240,999.1\n"})())
    rc = bench.main(["--model", "dummy", "--out-dir", str(tmp_path), "--dry-run"])
    assert rc == 0
    created = [p for p in tmp_path.iterdir() if p.is_dir()]
    metadata = (created[0] / "metadata.json").read_text()
    assert '"system": "vllm"' in metadata


def test_parse_args_rejects_unknown_prompt():
    with pytest.raises(SystemExit) as exc:
        bench.parse_args(["--model", "dummy", "--prompts", "missing_prompt"])
    assert "unknown prompts" in str(exc.value)
