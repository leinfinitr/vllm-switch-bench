# Exact-disk experiment

## Question

Can exact runtime weight bytes be spilled to a physically allocated disk payload, have their
CPU backup released, and then restore into the allocator with identical deterministic model
output?

## Metric

The retained claim-support metrics are:

- runtime payload, spill, demotion-release, and restore-read bytes;
- physical filesystem allocation (`allocated_bytes`) as well as logical size;
- the omitted payload's SHA-256 and the runtime manifest's per-chunk SHA-256 values,
  offsets, and sizes;
- zero pending release obligation after demotion;
- equality of deterministic output before and after restore.

These are runtime integrity checks, not repository-internal Git-file digest manifests. The
payload itself is intentionally omitted from Git, so its retained digest and per-chunk
manifest are essential evidence.

## Method

The retained run used Qwen2.5-0.5B-Instruct on a single RTX 3080 with an instrumented vLLM
exact-disk implementation. Before unmapping runtime weight memory, the allocator produced a
disk backup in 16 MiB chunks and recorded the runtime segment layout and chunk digests. It
then released the exact CPU backup, restored from disk, and repeated deterministic
inference. Filesystem observation records both logical and allocated bytes.

The current builder recomputes the byte totals, payload identity, and output equality from
seven retained evidence files and produces a compact figure. The semantic validator requires
the spill/demotion/sleep phases, exactly 1,048,576,000 payload bytes, segment-size closure,
nonempty 64-character chunk digests, physical allocation of at least the payload size,
settled release, equal outputs, and exact raw-to-summary recomputation.

## Retained result

The observation records 1,048,576,000 bytes spilled, released from CPU backup, and restored
from disk. The filesystem reported 1,048,580,096 allocated bytes for the
1,048,576,000-byte logical payload. The before/after completion text is identical. The
retained payload SHA-256 is
`4aec04d7b5d1a8a9ace300e239bc65381955b058f2dab0326b8a44dc3afbbdbb`.

- [Exact-disk figure (PNG)](../../../results/exact-disk/figures/exact-disk.png)
- [Exact-disk figure (PDF)](../../../results/exact-disk/figures/exact-disk.pdf)
- [Machine-readable summary](../../../results/exact-disk/summary.json)
- [Runtime segment/checksum manifest](../../../results/exact-disk/raw/exact-disk/bundle-manifest.json)
- [Result-family notes](../../../results/exact-disk/README.md)

## Threats to validity

- This is one local single-GPU observation for one model and approximately 0.98 GiB of
  runtime payload.
- Filesystem allocation does not prove storage media durability, cache coldness, or device
  read behavior; page cache and filesystem/kernel configuration can affect timings.
- Output equality covers one deterministic prompt/output observation, not arbitrary model
  behavior or every runtime tensor.
- Chunk hashes detect corruption against the retained manifest but do not establish that all
  runtime mutation paths correctly invalidate or refresh a backup.
- The exact-disk implementation is an external engine modification, not stock vLLM.

## Limitations

No new data was generated in this refactor, and a canonical GPU rerun is not complete. The
retained metadata identifies the collection/upstream engine commits and environment for this
observation, but it is not a broad performance study and the payload bytes are intentionally
not published. The wider v0.1 E2E producer did not runtime-bind engine/controller commits,
imported path, or configuration hash, so those E2E numbers remain a historical local
observation; the stronger exact-disk runtime checksum record does not retroactively repair
that separate provenance gap.

The current summary supports functional exact-byte spill/release/restore. It does not claim
steady-state latency, throughput, SSD endurance, crash consistency, multi-model capacity, or
superiority over another storage tier.

## Reproduce

### Deterministic CPU rebuild and validation

```bash
uv sync --frozen --group dev
scripts/exact-disk-build.sh
scripts/exact-disk-validate.sh
scripts/validate_all.sh
git diff --exit-code -- results/exact-disk
```

Repeat the build/validation/diff sequence once more. It validates the retained manifest and
observations but does not recreate the omitted payload or run a GPU.

### Live runtime capture (GPU; not run in this refactor)

With a compatible instrumented vLLM environment:

```bash
scripts/exact-disk-run.sh \
  --model model=/path/to/model \
  --backup-root runtime/exact-disk-backups \
  --out-dir results/tmp/exact-disk/run-001 \
  -- /path/to/python -m llm_switch_bench.experiments.exact_disk.lifecycle_driver
```

The default rejects an existing destination and nonempty backup root. Bind the actual engine
repository through run metadata/environment, record the imported path and configuration,
and retain command success, profile events, resource samples, footprint growth, runtime
checksum manifest, and output equality before considering promotion.
