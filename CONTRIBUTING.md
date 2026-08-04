# Contributing

Thank you for improving LLM Switch Bench. Contributions should preserve the distinction between reusable harness code, machine-local runs, and publishable evidence.

## Development setup

```bash
uv sync --frozen --group dev
uv run pytest tests -q
uv run ruff check src scripts tests
scripts/check_bash.sh
uv run python scripts/check_docs.py
```

Python is fixed to the supported 3.12 series in `.python-version` and `pyproject.toml`. The lockfile is authoritative for CPU-only development and artifact rebuilding. GPU frameworks, models, CUDA, and external baseline runtimes are intentionally not locked because a benchmark run must record their exact source/image identity.

## Changes

1. Open an issue for changes that alter metric definitions, success predicates, or result policy.
2. Add focused tests for behavior changes.
3. Use portable paths in code and current documentation. Put machine-specific settings in ignored `*.local.yaml` files.
4. Never rewrite checksummed raw evidence. Add a new immutable run instead.
5. Retain failed and blocked attempts as structured diagnostics; do not convert them into latency rows.
6. Run all local gates above and describe any GPU tests that were not run.

## Result contributions

A publishable bundle must include frozen inputs, request- or phase-level raw evidence, run-time source and environment identity, correctness/post-condition evidence, deterministic derived artifacts, and SHA-256 manifests. See [`results/README.md`](results/README.md) and [`docs/release-artifact.md`](docs/release-artifact.md).

## Commits and pull requests

Use focused English commits. Pull requests must state the result-policy impact, exact tests, whether raw bytes changed, and which GPU reruns remain required. Do not include model weights, credentials, local caches, or unrelated experiment output.
