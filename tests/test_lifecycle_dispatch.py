from __future__ import annotations

from llm_switch_bench.experiments.lifecycle_latency import run


def test_dispatch_forwards_adapter_arguments(monkeypatch) -> None:
    captured: list[str] = []

    def adapter(argv):
        captured.extend(argv)
        return 7

    monkeypatch.setitem(run.SYSTEMS, "vllm", adapter)

    assert run.main(["vllm", "--cycles", "5"]) == 7
    assert captured == ["--cycles", "5"]
