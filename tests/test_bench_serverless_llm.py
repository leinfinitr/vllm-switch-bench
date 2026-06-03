from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_serverless_llm import (
    build_register_payload,
    run_delete_register,
    run_scale_to_zero_restore,
)


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
            ("GET", "/health"): [FakeResponse(200, {"status": "ok"})],
            ("POST", "/register"): [
                FakeResponse(200, {"status": "ok"}),
                FakeResponse(200, {"status": "ok"}),
            ],
            ("POST", "/delete"): [FakeResponse(200, {"status": "deleted"})],
            ("GET", "/v1/models"): [FakeResponse(200, {"object": "list", "models": []})],
        }

    def request(self, method, url, json=None, timeout=None):
        path = url.split("http://127.0.0.1:8343", 1)[1]
        self.calls.append((method, path, json))
        return self.responses[(method, path)].pop(0)



def test_build_register_payload_uses_vllm_backend_and_local_model():
    payload = build_register_payload(
        model_path=Path("/home/ljl/models/hf/Qwen2.5-0.5B-Instruct"),
        prompt_model_name="Qwen2.5-0.5B-Instruct",
        registered_model_name="qwen2p5-0p5b",
        max_model_len=1024,
        gpu_memory_utilization=0.45,
    )
    assert payload["model"] == "qwen2p5-0p5b"
    assert payload["backend"] == "vllm"
    assert payload["num_gpus"] == 1
    assert payload["auto_scaling_config"]["min_instances"] == 0
    assert (
        payload["backend_config"]["pretrained_model_name_or_path"]
        == "/home/ljl/models/hf/Qwen2.5-0.5B-Instruct"
    )



def test_serverless_delete_register_sequence_records_evict_restore(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(
        "bench_serverless_llm.infer",
        lambda *args, **kwargs: {
            "ok": True,
            "ttft_s": 0.1,
            "client_latency_s": 0.2,
            "approx_tokens_per_s": 10.0,
        },
    )
    monkeypatch.setattr(
        "bench_serverless_llm.time.perf_counter",
        iter([1.0, 1.3, 2.0, 2.4, 3.0, 3.1]).__next__,
    )
    row = run_delete_register(
        base_url="http://127.0.0.1:8343",
        client=fake,
        payload={
            "model": "qwen2p5-0p5b",
            "benchmark_metadata": {
                "source_model_path": "/home/ljl/models/hf/Qwen2.5-0.5B-Instruct"
            },
        },
        prompt_name="short_short",
        repeat_index=0,
    )
    assert row["system"] == "serverless_llm"
    assert row["model"] == "/home/ljl/models/hf/Qwen2.5-0.5B-Instruct"
    assert row["registered_model_name"] == "qwen2p5-0p5b"
    assert row["evict"]["latency_s"] == pytest.approx(0.3)
    assert row["restore"]["latency_s"] == pytest.approx(0.4)
    assert [call[:2] for call in fake.calls] == [
        ("GET", "/health"),
        ("POST", "/register"),
        ("POST", "/delete"),
        ("POST", "/register"),
    ]



def test_non_200_register_is_failed_with_body_snippet(monkeypatch):
    fake = FakeClient()
    fake.responses[("POST", "/register")] = [FakeResponse(500, text="boom")]
    monkeypatch.setattr("bench_serverless_llm.infer", lambda *args, **kwargs: {"ok": True})
    row = run_delete_register(
        base_url="http://127.0.0.1:8343",
        client=fake,
        payload={
            "model": "qwen2p5-0p5b",
            "benchmark_metadata": {
                "source_model_path": "/home/ljl/models/hf/Qwen2.5-0.5B-Instruct"
            },
        },
        prompt_name="short_short",
        repeat_index=0,
    )
    assert row["ok"] is False
    assert "boom" in row["error"]



def test_summary_row_has_system_serverless_llm(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(
        "bench_serverless_llm.infer",
        lambda *args, **kwargs: {
            "ok": True,
            "ttft_s": 0.1,
            "client_latency_s": 0.2,
            "approx_tokens_per_s": 10.0,
        },
    )
    monkeypatch.setattr(
        "bench_serverless_llm.time.perf_counter",
        iter([1.0, 1.3, 2.0, 2.4, 3.0, 3.1]).__next__,
    )
    row = run_delete_register(
        base_url="http://127.0.0.1:8343",
        client=fake,
        payload={
            "model": "qwen2p5-0p5b",
            "benchmark_metadata": {
                "source_model_path": "/home/ljl/models/hf/Qwen2.5-0.5B-Instruct"
            },
        },
        prompt_name="short_short",
        repeat_index=0,
    )
    assert row["system"] == "serverless_llm"



def test_scale_to_zero_restore_sequence_records_wait_and_restore(monkeypatch):
    fake = FakeClient()
    fake.responses[("GET", "/health")] = [FakeResponse(200, {"status": "ok"})]
    fake.responses[("POST", "/register")] = [FakeResponse(200, {"status": "ok"})]
    infer_calls = iter(
        [
            {
                "ok": True,
                "status": 200,
                "client_latency_s": 12.5,
                "approx_tokens_per_s": 2.0,
                "output_prefix": "before",
            },
            {
                "ok": True,
                "status": 200,
                "client_latency_s": 11.2,
                "approx_tokens_per_s": 2.2,
                "output_prefix": "first post evict",
            },
            {
                "ok": True,
                "status": 200,
                "client_latency_s": 1.2,
                "approx_tokens_per_s": 3.2,
                "output_prefix": "second active",
            },
        ]
    )
    wait_calls = iter(
        [
            {"ok": True, "latency_s": 0.6, "body": "baseline idle", "idle_gpu_threshold_mib": 538},
            {"ok": True, "latency_s": 2.5, "body": "scaled zero", "idle_gpu_threshold_mib": 538},
        ]
    )
    monkeypatch.setattr("bench_serverless_llm.infer", lambda *args, **kwargs: next(infer_calls))
    monkeypatch.setattr("bench_serverless_llm.wait_for_scale_to_zero", lambda *args, **kwargs: next(wait_calls))
    monkeypatch.setattr("bench_serverless_llm.query_gpu_used_mib", lambda: 238)
    row = run_scale_to_zero_restore(
        base_url="http://127.0.0.1:8343",
        client=fake,
        payload={
            "model": "qwen2p5-0p5b",
            "benchmark_metadata": {
                "source_model_path": "/home/ljl/models/hf/Qwen2.5-0.5B-Instruct"
            },
        },
        prompt_name="short_short",
        repeat_index=0,
        scale_zero_timeout_s=120.0,
        scale_zero_poll_interval_s=1.0,
        idle_gpu_buffer_mib=300,
    )
    assert row["ok"] is True
    assert row["method"] == "scale_to_zero_restore"
    assert row["startup_latency_s"] == pytest.approx(0.6)
    assert row["evict"]["ok"] is True
    assert row["evict"]["latency_s"] == pytest.approx(2.5)
    assert row["restore"]["latency_s"] == pytest.approx(10.0)
    assert row["infer_after"]["client_latency_s"] == pytest.approx(1.2)
    assert row["restore_latency_estimated"] is True
    assert row["ttft_available"] is False
    assert row["tpot_available"] is False
    assert row["memory_gpu_used_ready_mib"] == 238
    assert row["memory_gpu_used_evict_mib"] == 538
    assert row["stage_breakdown"]["baseline_gpu_used_mib"] == 238
    assert row["stage_breakdown"]["idle_gpu_threshold_mib"] == 538
    assert [call[:2] for call in fake.calls] == [
        ("GET", "/health"),
        ("POST", "/register"),
    ]


def test_serverless_docker_compose_mounts_host_models():
    compose = (Path(__file__).resolve().parents[1].parent / "ServerlessLLM" / "examples" / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    assert "${HOST_MODEL_FOLDER:-/home/ljl/models}:/host-models" in compose
