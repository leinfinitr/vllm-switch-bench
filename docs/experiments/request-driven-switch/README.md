# Request-driven switch

## Question

What request completion latencies were observed when one endpoint received a frozen 20-request alternating-model schedule and the serving system performed the required switches?

## Metric boundary

Completion latency is measured from client dispatch to completed streaming response; `dispatch_lag_ms` separately records lateness relative to each absolute scheduled arrival. A retained row is successful only when it has HTTP 2xx status, no recorded error, a complete stream marker, a semantic first-token timestamp, and non-empty semantic output. This curve includes queueing, switching, and generation; it is not lifecycle wake latency.

## Method

The migrated evidence contains one Proposed and one llama-swap 20-request array plus retained JSONL source rows. The validator binds every row to the current frozen trace's complete dispatch identity (`request_id`, endpoint, model, prompt name, generation fields, stream mode, and scheduled offset), requires 20 unique IDs and the canonical strict-success predicate, validates non-negative finite timing fields, and recomputes both aggregate rows from the arrays. The JSONL copies are retained source evidence but are not duplicate builder inputs.

## Result

For this historical alternating trace, the retained medians are about 0.859 seconds for Proposed and 12.868 seconds for llama-swap, with zero recorded failures in each 20-row array. These values describe the retained local observation only.

![Request completion timeline](../../../results/request-driven-switch/figures/request-timeline.png)

## Threats and limitations

- The v0.1 producer rows did **not** runtime-bind controller or engine commits, dirty states, executable/import paths, configuration hash, or model revision.
- The artifact therefore supports a historical local observation, not exact fresh-clone runtime reproduction or a current cross-system ranking.
- The rows bind all supplied dispatch fields to the retained frozen trace, but `prompt_name` does not independently authenticate the exact prompt catalog bytes used by the historical service run.
- One long-lived 20-request trace is descriptive, not an independent-run reliability estimate.
- No new data was generated, and the canonical GPU rerun is not complete.
