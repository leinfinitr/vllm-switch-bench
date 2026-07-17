from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_serverless_llm import (
    build_register_payload,
    infer,
    parse_args,
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


def infer_result(latency: float, tokens: int = 10, prefix: str = "ok") -> dict:
    return {
        "ok": True,
        "status": 200,
        "ttft_s": None,
        "client_latency_s": latency,
        "approx_output_tokens": tokens,
        "approx_tokens_per_s": tokens / latency,
        "output_prefix": prefix,
    }


def test_build_register_payload_uses_vllm_backend_and_local_model():
    payload = build_register_payload(
        model_path=Path("/models/example"),
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
        == "/models/example"
    )


def test_serverless_delete_register_warms_backend_and_measures_ready_latencies(monkeypatch):
    fake = FakeClient()
    infer_calls: list[str] = []
    infer_results = iter([
        infer_result(8.0, prefix="initial warmup"),
        infer_result(0.8, prefix="before measured"),
        infer_result(9.0, prefix="restore warmup"),
        infer_result(0.9, prefix="after measured"),
    ])

    def fake_infer(base_url, model_name, prompt_name, **kwargs):
        infer_calls.append(prompt_name)
        return next(infer_results)

    wait_calls = iter([
        {"ok": True, "latency_s": 0.5, "gpu_used_mib": 240, "idle_gpu_threshold_mib": 538},
    ])
    gpu_values = iter([2600, 240])
    monkeypatch.setattr("bench_serverless_llm.infer", fake_infer)
    monkeypatch.setattr("bench_serverless_llm.wait_for_scale_to_zero", lambda *args, **kwargs: next(wait_calls))
    monkeypatch.setattr("bench_serverless_llm.query_gpu_used_mib", lambda: next(gpu_values))
    monkeypatch.setattr(
        "bench_serverless_llm.time.perf_counter",
        iter([1.0, 1.2, 2.0, 2.3]).__next__,
    )

    row = run_delete_register(
        base_url="http://127.0.0.1:8343",
        client=fake,
        payload={
            "model": "qwen2p5-0p5b",
            "benchmark_metadata": {
                "source_model_path": "/models/example"
            },
        },
        prompt_name="long_short",
        repeat_index=0,
        scale_zero_timeout_s=120.0,
        scale_zero_poll_interval_s=0.001,
        idle_gpu_buffer_mib=300,
    )

    assert row["system"] == "serverless_llm"
    assert row["startup_latency_s"] is None
    assert row["memory_gpu_used_ready_mib"] == 2600
    assert row["memory_gpu_used_evict_mib"] == 240
    assert row["infer_before"]["client_latency_s"] == pytest.approx(0.8)
    assert row["infer_after"]["client_latency_s"] == pytest.approx(0.9)
    assert row["evict"]["latency_s"] == pytest.approx(0.7)
    assert row["restore"]["latency_s"] == pytest.approx(8.1)
    assert row["ttft_available"] is False
    assert row["tpot_available"] is False
    assert infer_calls == ["long_short", "long_short", "long_short", "long_short"]
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
                "source_model_path": "/models/example"
            },
        },
        prompt_name="short_short",
        repeat_index=0,
        scale_zero_timeout_s=120.0,
        scale_zero_poll_interval_s=0.001,
        idle_gpu_buffer_mib=300,
    )
    assert row["ok"] is False
    assert "boom" in row["error"]


def test_summary_row_has_system_serverless_llm(monkeypatch):
    fake = FakeClient()
    infer_results = iter([
        infer_result(1.0),
        infer_result(0.2),
        infer_result(1.0),
        infer_result(0.2),
    ])
    monkeypatch.setattr("bench_serverless_llm.infer", lambda *args, **kwargs: next(infer_results))
    monkeypatch.setattr(
        "bench_serverless_llm.wait_for_scale_to_zero",
        lambda *args, **kwargs: {"ok": True, "latency_s": 0.1, "gpu_used_mib": 238, "idle_gpu_threshold_mib": 538},
    )
    monkeypatch.setattr("bench_serverless_llm.query_gpu_used_mib", lambda: 238)
    monkeypatch.setattr(
        "bench_serverless_llm.time.perf_counter",
        iter([1.0, 1.1, 2.0, 2.1]).__next__,
    )
    row = run_delete_register(
        base_url="http://127.0.0.1:8343",
        client=fake,
        payload={
            "model": "qwen2p5-0p5b",
            "benchmark_metadata": {
                "source_model_path": "/models/example"
            },
        },
        prompt_name="short_short",
        repeat_index=0,
        scale_zero_timeout_s=120.0,
        scale_zero_poll_interval_s=0.001,
        idle_gpu_buffer_mib=300,
    )
    assert row["system"] == "serverless_llm"


def test_scale_to_zero_restore_warms_backend_and_subtracts_ready_request(monkeypatch):
    fake = FakeClient()
    fake.responses[("GET", "/health")] = [FakeResponse(200, {"status": "ok"})]
    fake.responses[("POST", "/register")] = [FakeResponse(200, {"status": "ok"})]
    infer_calls: list[str] = []
    infer_results = iter(
        [
            infer_result(12.5, prefix="initial warmup"),
            infer_result(1.5, prefix="before measured"),
            infer_result(11.2, prefix="restore warmup"),
            infer_result(1.2, prefix="after measured"),
        ]
    )

    def fake_infer(base_url, model_name, prompt_name, **kwargs):
        infer_calls.append(prompt_name)
        return next(infer_results)

    wait_calls = iter(
        [
            {"ok": True, "latency_s": 0.6, "body": "baseline idle", "idle_gpu_threshold_mib": 538},
            {"ok": True, "latency_s": 2.5, "body": "scaled zero", "gpu_used_mib": 238, "idle_gpu_threshold_mib": 538},
        ]
    )
    monkeypatch.setattr("bench_serverless_llm.infer", fake_infer)
    monkeypatch.setattr("bench_serverless_llm.wait_for_scale_to_zero", lambda *args, **kwargs: next(wait_calls))
    monkeypatch.setattr("bench_serverless_llm.query_gpu_used_mib", lambda: 2600)
    row = run_scale_to_zero_restore(
        base_url="http://127.0.0.1:8343",
        client=fake,
        payload={
            "model": "qwen2p5-0p5b",
            "benchmark_metadata": {
                "source_model_path": "/models/example"
            },
        },
        prompt_name="short_long",
        repeat_index=0,
        scale_zero_timeout_s=120.0,
        scale_zero_poll_interval_s=0.001,
        idle_gpu_buffer_mib=300,
    )
    assert row["ok"] is True
    assert row["method"] == "scale_to_zero_restore"
    assert row["startup_latency_s"] is None
    assert row["memory_gpu_used_ready_mib"] == 2600
    assert row["evict"]["ok"] is True
    assert row["evict"]["latency_s"] == pytest.approx(2.5)
    assert row["restore"]["latency_s"] == pytest.approx(10.0)
    assert row["infer_before"]["client_latency_s"] == pytest.approx(1.5)
    assert row["infer_after"]["client_latency_s"] == pytest.approx(1.2)
    assert row["restore_latency_estimated"] is True
    assert row["ttft_available"] is False
    assert row["tpot_available"] is False
    assert row["memory_gpu_used_evict_mib"] == 238
    assert row["stage_breakdown"]["baseline_idle_wait_s"] == 0.6
    assert row["stage_breakdown"]["initial_warm_request_s"] == 12.5
    assert row["stage_breakdown"]["restore_warm_request_s"] == 11.2
    assert row["stage_breakdown"]["second_active_request_s"] == 1.2
    assert infer_calls == ["short_long", "short_long", "short_long", "short_long"]
    assert [call[:2] for call in fake.calls] == [
        ("GET", "/health"),
        ("POST", "/register"),
    ]


def test_scale_to_zero_poll_interval_defaults_to_one_millisecond():
    args = parse_args([
        "--repo",
        "/repo/ServerlessLLM",
        "--model",
        "/host-models/hf/Qwen2.5-0.5B-Instruct",
    ])
    assert args.scale_zero_poll_interval == pytest.approx(0.001)


def test_infer_returns_structured_timeout_error(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise __import__("requests").exceptions.ReadTimeout("slow request")

    monkeypatch.setattr("bench_serverless_llm.requests.post", raise_timeout)
    row = infer("http://127.0.0.1:8343", "qwen", "short_short", timeout_s=0.01)
    assert row["ok"] is False
    assert row["status"] is None
    assert "timed out" in row["error"]
    assert row["client_latency_s"] >= 0
