# llama-swap

**Status:** current external baseline.

The canonical project name is **llama-swap**. It exposes an OpenAI-compatible proxy and automatically switches model processes based on the request body's `model` field. Operators do not manually call sleep and wake for the normal request-trace benchmark.

## Automatic switching semantics

When a request targets a model that is not ready, llama-swap may stop the current model process, start the target process, wait for its health endpoint, queue requests behind the transition, and coalesce requests for the same target. Consequently:

- request E2E/TTFT includes queueing, automatic process management, proxying, and inference;
- one nominal A/B request transition does not necessarily equal one independent process start;
- request latency must never be labeled lifecycle wake latency.

llama-swap has no public phase API matching vLLM L1/L2. For lifecycle figures, apply the retained benchmark-only profiler patch and measure source state intervals:

- sleep: `ready -> stopping -> stopped`, plus an external idle-GPU post-condition;
- wake: `stopped -> starting -> ready`, ending at successful health readiness;
- correctness: a separate complete streamed inference.

Use `uv run python -m llm_switch_bench.adapters.llama_swap` for the instrumented lifecycle and [`../../scripts/request-driven-switch.sh`](../../scripts/request-driven-switch.sh) for automatic routing. Freeze the upstream commit, patch checksum, binary checksum, configuration, model revisions, and child-process environment at run start.
