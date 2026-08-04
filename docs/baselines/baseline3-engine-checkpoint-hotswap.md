# Baseline3 checkpoint/hotswap harness

**Status:** historical compatibility harness, superseded for v0.1 publication by [`../release-artifact.md`](../release-artifact.md).

Baseline3 aggregates existing vLLM results and can invoke legacy ServerlessLLM and SwapServeLLM adapters. It remains useful for debugging old artifacts but does not enforce the complete v0.1 run-time provenance or final matrix contract.

```bash
cp configs/baseline3.example.yaml configs/baseline3.local.yaml
$EDITOR configs/baseline3.local.yaml
scripts/run_baseline3.sh
```

The configuration must explicitly provide the vLLM result directory, external repository paths, service addresses, and model mount paths. The harness does not infer a latest run or host/container mount mapping.

Do not promote Baseline3 output into `results/release-v0.1/` without passing the current release policy and rebuilding the entire canonical bundle.
