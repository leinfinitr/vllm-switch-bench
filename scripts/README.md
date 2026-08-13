# Shell entry points

`scripts/` contains thin executable shell wrappers only. Reusable benchmark, analysis,
plotting, provenance, artifact, and validation behavior belongs in the installed
`llm_switch_bench` package under `src/llm_switch_bench/`.

Except for the syntax enumerator, each wrapper resolves the repository root, changes to it,
and `exec`s `uv run python -m <module> "$@"`. Arguments and exit status therefore pass
through unchanged. Run the wrappers from any working directory after
`uv sync --frozen --group dev`.

## Experiment wrappers

| Wrapper | Package module | Purpose |
|---|---|---|
| `lifecycle-latency.sh` | `llm_switch_bench.experiments.lifecycle_latency.run` | Dispatch retained lifecycle measurement to a vLLM, llama-swap, SwapServeLLM, or ServerlessLLM adapter |
| `vllm-profiling.sh` | `llm_switch_bench.experiments.vllm_profiling.run` | Launch and profile cold reload or vLLM L1/L2 phases |
| `vllm-profiling-plot.sh` | `llm_switch_bench.experiments.vllm_profiling.plot` | Build a candidate vLLM profiling summary and figure from compact samples |
| `request-driven-switch.sh` | `llm_switch_bench.experiments.request_driven_switch.run` | Replay one frozen OpenAI-compatible request trace |
| `request-driven-switch-matrix.sh` | `llm_switch_bench.experiments.request_driven_switch.run_matrix` | Replay the three repository traces repeatedly against one endpoint |
| `backup-reuse-reclaim.sh` | `llm_switch_bench.experiments.backup_reuse_reclaim.run` | Run same-process repeated model sleep/wake with reuse/reclaim assertions |
| `run_profiling.sh` | `llm_switch_bench.experiments.backup_reuse_reclaim.pin_compare` | Run pinned/pageable lifecycle profiling comparisons |
| `exact-disk-run.sh` | `llm_switch_bench.experiments.exact_disk.run` | Wrap a compatible runtime command and capture exact-disk evidence |

A separate cross-system randomized request matrix is exposed by the installed
`llm-switch-trace-matrix` console command (module
`llm_switch_bench.experiments.request_driven_switch.matrix`).

These commands can launch or interact with GPU software. They are not run by CPU CI. Use
ignored `results/tmp/` output and follow the relevant protocol under
[`../docs/experiments/`](../docs/experiments/).

## Artifact and policy wrappers

| Wrapper | Package module | Contract |
|---|---|---|
| `build_all.sh` | `llm_switch_bench.build_all` | Rebuild all five current summaries, figures, and metadata from retained raw inputs |
| `vllm-profiling-build.sh` | `llm_switch_bench.artifacts vllm-profiling` | Rebuild only the retained vLLM profiling family |
| `vllm-profiling-validate.sh` | `llm_switch_bench.validation.vllm_profiling.validate` | Validate one vLLM profiling family (optional path argument) |
| `validate_all.sh` | `llm_switch_bench.validation.validate_all` | Enforce all family shape and semantic validators |
| `exact-disk-build.sh` | `llm_switch_bench.artifacts exact-disk` | Rebuild only the retained exact-disk family |
| `exact-disk-validate.sh` | `llm_switch_bench.validation.exact_disk.validate` | Validate one exact-disk family (optional path argument) |
| `docs.sh` | `llm_switch_bench.check_docs` | Check required current docs and prohibited stale references |
| `tracked-ignore.sh` | `llm_switch_bench.tracked_ignore` | Fail if any tracked file matches the ignore rules |
| `check_bash.sh` | shell built-in syntax loop | Run `bash -n` on every top-level `.sh` entry point |

## Deterministic CPU publication gate

```bash
scripts/check_bash.sh
scripts/docs.sh
scripts/build_all.sh
scripts/validate_all.sh
git diff --exit-code -- results
scripts/build_all.sh
scripts/validate_all.sh
git diff --exit-code -- results
scripts/tracked-ignore.sh
```

The two passes must be clean. This rebuild creates no measurements and needs no GPU; it
recomputes only derived artifacts from retained inputs. Semantic validators, not internal
whole-tree digest lists, establish the current family contracts. External executable
digests and exact-disk runtime payload/chunk checksums remain part of retained evidence.

## Wrapper policy

When adding or changing an entry point:

1. implement and test behavior in `src/llm_switch_bench/`;
2. keep the shell file to strict mode, root discovery, and one `exec` command;
3. preserve all CLI arguments and the module's exit status;
4. add/update a `[project.scripts]` command when installation-level access is useful;
5. update this table, the experiment protocol, tests, and CI as applicable;
6. run `scripts/check_bash.sh` and the module's `--help` smoke test.

Do not put Python files, environment-specific paths, embedded experiment matrices, result
aggregation, or publication policy logic in `scripts/`.
