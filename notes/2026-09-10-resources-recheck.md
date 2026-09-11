# Resource revalidation — 10 September 2026

Rechecked this session's hardware, installed models, and training access before
sizing the parameter comparison, per the README's resource section. The sibling
MLX route is a starting point, not proof; this note records the local
revalidation. All checks were local and unpriced.

## Environment

- MacBook Pro, Apple M3 Max (16 CPU cores: 12 performance + 4 efficiency),
  48 GB unified memory (`hw.memsize` 51,539,607,552 bytes), macOS 26.6.2
  (arm64). `uv` 0.10.3; Docker at `/usr/local/bin/docker`.
- Docker Model Runner is running: llama.cpp b9879-metal
  (sha256:b70706f473b4043ca3e0c32704a7fda3412b83bceef0564684187b8011230de8);
  diffusers, vllm, and the Docker MLX backend are **not** installed.
- Installed Docker models include `huggingface.co/qwen/qwen3-4b-gguf:Q4_K_M`
  (sha256:618c80458ca4012b132ef1847bcd49ec5f923c3d9df35fdc534715085108e9f3),
  `huggingface.co/qwen/qwen3-8b-gguf:Q4_K_M`,
  `huggingface.co/huggingfacetb/smollm2-1.7b-instruct-gguf:Q4_K_M`,
  `huggingface.co/ggml-org/smollm3-3b-gguf:Q4_K_M`, `llama3.2:1B-Q8_0`, and
  larger qwen3/llama/gpt-oss variants.

## MLX training route

- The sibling study's venv (Python 3.14.6) carries mlx 0.32.2 with mlx-lm
  installed from git commit `86b48c461feebf87c58788655b7e57b5574b9e6d`; the
  dist-info `direct_url.json` confirms the commit matches the sibling
  pyproject pin.
- Re-ran the sibling `scripts/mlx_smoke.py` (unmodified) on this machine with
  the sibling `models/qwen3-0.6b` weights: completed in 11.2 s, peak MLX memory
  1.83 GB. All invariants held: exact adapter reset (max logit delta 0.0),
  base weights unchanged, repeated branch bit-identical. Its competence probes
  scored 9/12 (diagnostic only, not part of this study).
- The sibling script uses only `mlx_lm.load`, `stream_generate`,
  `make_sampler`, `tuner.utils.linear_to_lora_layers`, and manual
  `nn.value_and_grad` + `optim.AdamW` steps; it does not use the tuner's
  removed/renamed high-level APIs, so the pinned commit is API-compatible with
  the intended manual training loop.

## Consequences for this study

- Parameter intervention must use native MLX (no MLX backend in Docker).
  Development diagnostics (pilot-02 onward, before the frozen comparison)
  continue on the Docker llama.cpp endpoint for continuity with pilot-01's
  model file and runtime. Docker and MLX elapsed times are never compared as a
  storage effect.
- This study now owns its own venv (Python 3.13.12) with the same pinned
  mlx-lm commit via the `adaptation` extra; `uv.lock` records the resolved
  commit. Verified working: imports and a Metal matrix multiply in
  `.venv/bin/python`.
- Scratch evidence for the smoke re-run lives in
  `tmp/mlx-route-recheck/` (gitignored; not part of the study record).
