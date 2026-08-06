import json

from llm_switch_bench.common import analyze_results


def test_build_report_uses_simplified_startup_latency_field(tmp_path):
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "summary.csv").write_text(
        "system,run_id,method,model,prompt_name,repeat_index,ok,startup_latency_s,evict_latency_s,restore_latency_s,ttft_before_s,ttft_after_s,latency_before_s,latency_after_s\n"
        "vllm,r1,sleep_l1,m,short_short,0,True,1.25,0.2,0.3,0.01,0.02,0.4,0.5\n",
        encoding="utf-8",
    )
    (result_dir / "summary.json").write_text(
        json.dumps([{"run_id": "r1", "event_log": "missing.jsonl"}]),
        encoding="utf-8",
    )

    report = analyze_results.build_report(result_dir, tmp_path)

    assert "| sleep_l1 | short_short | True | 1 | 1.2500 |" in report
    assert "## Ready / evicted memory" in report
    assert "| sleep_l1 | short_short | n/a | n/a | n/a | n/a |" in report
