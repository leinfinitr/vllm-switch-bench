from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_swapserve_llm import auth_headers, parse_swapserve_stage_logs, run_swapout_swapin


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="OK"):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data


class FakeClient:
    def __init__(self):
        self.calls = []
        self.responses = {
            ("GET", "/v1/models"): [FakeResponse(200, {"data": [{"id": "qwen"}]})],
            ("POST", "/api/swapout"): [FakeResponse(200, text="swapout ok")],
            ("POST", "/api/swapin"): [FakeResponse(200, text="swapin ok")],
        }

    def request(self, method, url, json=None, timeout=None, headers=None):
        path = url.split("http://127.0.0.1:8000", 1)[1]
        self.calls.append((method, path, json, headers))
        return self.responses[(method, path)].pop(0)



def test_swapserve_api_paths():
    fake = FakeClient()
    fake.request("GET", "http://127.0.0.1:8000/v1/models")
    fake.request("POST", "http://127.0.0.1:8000/api/swapout", {"model": "qwen"})
    fake.request("POST", "http://127.0.0.1:8000/api/swapin", {"model": "qwen"})
    assert [call[:2] for call in fake.calls] == [
        ("GET", "/v1/models"),
        ("POST", "/api/swapout"),
        ("POST", "/api/swapin"),
    ]



def test_auth_headers_empty_when_no_key():
    assert auth_headers(None) == {}



def test_auth_headers_bearer_when_key_present():
    assert auth_headers("dummy") == {"Authorization": "Bearer dummy"}



def test_swapout_swapin_sequence_records_evict_restore(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(
        "bench_swapserve_llm.infer",
        lambda *args, **kwargs: {"ok": True, "ttft_s": 0.1, "client_latency_s": 0.2, "approx_tokens_per_s": 11.0},
    )
    monkeypatch.setattr("bench_swapserve_llm.time.perf_counter", iter([1.0, 1.4, 2.0, 2.6]).__next__)
    row = run_swapout_swapin(
        base_url="http://127.0.0.1:8000",
        client=fake,
        model_name="qwen",
        prompt_name="short_short",
        repeat_index=0,
        api_key="dummy",
    )
    assert row["system"] == "swapserve_llm"
    assert row["evict"]["latency_s"] == pytest.approx(0.4)
    assert row["restore"]["latency_s"] == pytest.approx(0.6)
    # GET models + swapout + swapin all carry auth now.
    assert fake.calls[0][3] == {"Authorization": "Bearer dummy"}
    assert fake.calls[1][3] == {"Authorization": "Bearer dummy"}
    assert fake.calls[2][3] == {"Authorization": "Bearer dummy"}



def test_non_200_swapin_marks_failure(monkeypatch):
    fake = FakeClient()
    fake.responses[("POST", "/api/swapin")] = [FakeResponse(500, text="nope")]
    monkeypatch.setattr("bench_swapserve_llm.infer", lambda *args, **kwargs: {"ok": True})
    row = run_swapout_swapin(
        base_url="http://127.0.0.1:8000",
        client=fake,
        model_name="qwen",
        prompt_name="short_short",
        repeat_index=0,
        api_key="dummy",
    )
    assert row["ok"] is False
    assert "nope" in row["error"]



def test_summary_row_has_system_swapserve_llm(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(
        "bench_swapserve_llm.infer",
        lambda *args, **kwargs: {"ok": True, "ttft_s": 0.1, "client_latency_s": 0.2, "approx_tokens_per_s": 11.0},
    )
    monkeypatch.setattr("bench_swapserve_llm.time.perf_counter", iter([1.0, 1.4, 2.0, 2.6]).__next__)
    row = run_swapout_swapin(
        base_url="http://127.0.0.1:8000",
        client=fake,
        model_name="qwen",
        prompt_name="short_short",
        repeat_index=0,
        api_key="dummy",
    )
    assert row["system"] == "swapserve_llm"



def test_parse_swapserve_stage_logs_handles_missing_logs():
    parsed = parse_swapserve_stage_logs("[🔃 SwapOut Stage] Pause container took 12ms\n")
    assert parsed["swapout.pause_container_s"] == pytest.approx(0.012)
    assert "swapin.load_model_s" not in parsed



def test_stage_breakdown_can_be_read_from_log_dir(monkeypatch, tmp_path: Path):
    fake = FakeClient()
    (tmp_path / "swapout.log").write_text("old\n[🔃 SwapOut Stage] Pause container took 12ms\n", encoding="utf-8")
    (tmp_path / "swapin.log").write_text("old\n", encoding="utf-8")
    monkeypatch.setattr("bench_swapserve_llm.infer", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr("bench_swapserve_llm.time.perf_counter", iter([1.0, 1.4, 2.0, 2.6]).__next__)
    # Append new content after offsets are captured inside run_swapout_swapin.
    original_request = fake.request

    def wrapped_request(method, url, json=None, timeout=None, headers=None):
        if method == "POST" and url.endswith("/api/swapout"):
            with (tmp_path / "swapout.log").open("a", encoding="utf-8") as handle:
                handle.write("[🔃 SwapOut Stage] Pause container took 34ms\n")
        if method == "POST" and url.endswith("/api/swapin"):
            with (tmp_path / "swapin.log").open("a", encoding="utf-8") as handle:
                handle.write("[🔄 SwapIn Stage] LoadModel completed in 56ms\n")
        return original_request(method, url, json=json, timeout=timeout, headers=headers)

    fake.request = wrapped_request
    row = run_swapout_swapin(
        base_url="http://127.0.0.1:8000",
        client=fake,
        model_name="qwen",
        prompt_name="short_short",
        repeat_index=0,
        log_dir=str(tmp_path),
        api_key="dummy",
    )
    assert row["stage_breakdown"]["swapout.pause_container_s"] == pytest.approx(0.034)
    assert row["stage_breakdown"]["swapin.load_model_s"] == pytest.approx(0.056)
