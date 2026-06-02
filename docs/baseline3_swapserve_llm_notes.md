# SwapServeLLM baseline notes

Current verified facts on this machine:

- Docker works and NVIDIA Docker works.
- Podman command is missing.
- `SwapServeLLM/pkg/podman/client.go` hardcodes Podman bindings and `systemctl start podman.socket`.
- `SwapServeLLM/main.go` starts inference engines through the Podman-based launcher path.
- `SwapServeLLM/pkg/containers/vllm_launcher.go` only prepares `/root/.cache/huggingface`; it does not mount `/home/ljl/models`.
- SwapOut/SwapIn stage logs are available and parsable.

Practical implication:

- Real benchmark execution is currently blocked by missing Podman runtime on this host.
- Docker cannot be used as a drop-in replacement without code changes, because the current code imports Podman APIs directly and calls Podman CLI/systemd lifecycle.

Possible future workaround paths:

1. Install/configure Podman + podman.socket for the user session.
2. Patch SwapServeLLM to support Docker as an alternative backend runtime.
3. Patch the vLLM launcher to mount a local host model path, e.g.
   `/home/ljl/models/hf/Qwen2.5-0.5B-Instruct -> /models/Qwen2.5-0.5B-Instruct`.

For the current Baseline3 run, the honest outcome is a blocker row rather than fabricated performance numbers.
