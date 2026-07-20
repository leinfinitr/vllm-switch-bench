import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/tool"))

from build_request_switch_artifact import build_summary  # noqa: E402


def test_committed_curated_summary_is_rebuildable():
    latest = ROOT / "results/request_switch/latest"
    rebuilt = build_summary(latest, latest / "provenance.json")
    committed = json.loads((latest / "summary.json").read_text())
    assert rebuilt == committed
