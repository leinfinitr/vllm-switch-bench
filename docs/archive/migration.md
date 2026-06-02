# Migration note

The benchmark harness was initially prototyped in
`/home/ljl/research-systems/prism-research/benchmark/model-switching`, which was
incorrect because `prism-research` is a separate project.

This project is the corrected standalone location:

`/home/ljl/research-systems/llm-switch-bench`

Actions taken:

1. Created a new git repository here.
2. Copied only benchmark source, tests, and documentation from the mistaken
   prototype location.
3. Did not treat the old `prism-research` result directories as authoritative.
4. Will use a new local uv environment in this repository for vLLM Sleep Mode
   testing.

The old prototype can be cleaned from `prism-research` after this standalone
project is verified.
