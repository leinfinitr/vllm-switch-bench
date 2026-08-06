# Request-driven switch experiment

## Question

When a frozen open-loop trace alternates model names at an OpenAI-compatible endpoint, what
request completion latency and failure behavior are observed for Proposed and llama-swap?

## Metric

The primary metric is per-request completion latency in seconds, measured from client
dispatch through complete semantic streamed output. The summary reports request count,
failed count, minimum, median, and maximum completion latency for each system.

A retained request is successful only when it has no recorded error, HTTP status 200, and a
complete SSE stream (`stream_done=true`). Semantic TTFT and TPOT are retained in request
rows, but the current family figure plots completion latency against the trace's scheduled
offset. Dispatch lag remains observable and is not silently subtracted.

## Method

Both systems replayed the same 20 immutable request identities (`w1-000` through `w1-019`),
model sequence, and absolute scheduled offsets from the alternating trace. Requests are
streaming, deterministic-temperature chat completions with bounded output length. The
builder recomputes descriptive latency summaries from retained JSON rows and plots each
request on the scheduled timeline.

The validator requires exactly 20 unique requests per system, the supported identity
sequence, identical `(request_id, model, scheduled_offset)` tuples across systems, strict
success for every retained row, finite positive completion latency, exactly the two expected
systems, and exact raw-to-summary recomputation.

## Retained result

All 20 retained requests per system satisfy the current success predicate. Proposed has a
median completion latency of about 0.859 s (0.278–1.081 s observed range); llama-swap has a
median of about 12.868 s (0.595–25.138 s observed range). The timeline shows the variation
under the alternating local trace.

This is descriptive retained evidence, not a new measurement or a canonical rerun.

- [Request timeline (PNG)](../../../results/request-driven-switch/figures/request-timeline.png)
- [Request timeline (PDF)](../../../results/request-driven-switch/figures/request-timeline.pdf)
- [Machine-readable summary](../../../results/request-driven-switch/summary.json)
- [Result-family notes](../../../results/request-driven-switch/README.md)

## Threats to validity

- The family contains one 20-request alternating trace in one local single-GPU setting.
- A request-visible metric includes routing, queueing, process/model switching, first-token
  delay, and token generation; it does not isolate lifecycle phase cost.
- Open-loop overlap means an earlier switch can affect later request latency.
- The systems may differ in process lifetime, cache state, scheduler behavior, and transport
  implementation despite sharing request identities.
- Minimum/median/maximum over 20 requests are descriptive and provide no confidence interval
  or throughput characterization.

## Limitations

The v0.1 E2E producer did not runtime-bind the engine and controller commits, actually
imported path, or behavior-affecting configuration hash. The retained values are therefore a
**historical local observation**, not an exact fresh-checkout runtime reproduction. No new
data was generated in this refactor, and a canonical GPU rerun is not complete.

The family does not cover burst/steady workloads, multiple arrival rates, concurrency
scaling, failures, ServerlessLLM, SwapServeLLM, or cluster deployments. The deterministic
rebuild validates retained rows and interpretation; it cannot supply missing runtime
provenance.

## Reproduce

### Deterministic CPU rebuild and validation

```bash
uv sync --frozen --group dev
scripts/build_all.sh
uv run python -m llm_switch_bench.validation.request_driven_switch.validate
scripts/validate_all.sh
git diff --exit-code -- results/request-driven-switch
```

Repeat the build/validation/diff sequence once more. This regenerates summary/figures from
tracked evidence and does not contact a serving endpoint or generate measurements.

### Live single-trace measurement (runtime/GPU; not run in this refactor)

Start a compatible endpoint with runtime identity capture, then run:

```bash
scripts/request-driven-switch.sh \
  --base-url http://127.0.0.1:9000 \
  --manifest configs/traces/request-switch-alternating.jsonl \
  --output results/tmp/request-driven-switch/alternating.jsonl
```

For the three repository traces repeated against one endpoint:

```bash
scripts/request-driven-switch-matrix.sh \
  --base-url http://127.0.0.1:9000 \
  --repeats 3 \
  --out-dir results/tmp/request-driven-switch/matrix
```

A publishable rerun must freeze absolute arrivals and bind benchmark, engine, controller,
imported-path/configuration, model, executable/image, and hardware identities at runtime.
