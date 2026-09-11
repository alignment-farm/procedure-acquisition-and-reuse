# Handoff audit and decision — 11 September 2026

Decision: preserve and refactor, not reset. The dependency pin, downloaded checkpoint, MLX smoke evidence and partial Docker calls are useful. Neither pilot establishes parameter acquisition. The initial committed pilot also has weak diagnostics; reverting would not fix the study design.

## Corrections to inherited claims

The Docker attempt has 28 logged rows: 27 responses and one HTTP 500. Four scheduled requests were never attempted. Its `failure.json` is preserved as an inherited annotation, not primary evidence for the server error's cause. The response log establishes HTTP 500 but does not include a response body or establish concurrent loading as its cause. The annotation's claim of a completed retry is unsupported by an accompanying result directory. Treat shared-memory contention as a hypothesis. No timing offset is independently verified here.

The downloaded native checkpoint is present (~8 GB of weight files). It is not established to match the Docker GGUF. A change in checkpoint, precision and runtime prevents attributing any improved results solely to the prompt. The native diagnostic is a new exploratory condition.

The original MLX draft is preserved in `evidence/handoff-audit/pilot2_mlx.original.py`. Static inspection against installed libraries found `os.python_version()` does not exist, a sampler signature mismatch, chat rendering returning token IDs by default followed by another encode, and a nonexistent model-level generation interface. These were not transient network errors. It also read entire weight shards into RAM for hashes and lacked guaranteed partial summaries.

## Repair

The repaired runner uses the installed MLX-LM streaming interface, explicit string chat rendering, streaming file hashes, actual package/source metadata and the model download metadata. It validates required shards before opening an output directory, accepts explicit model/output paths, records partial summaries on Python exceptions/interruption, and preserves failed generation text. It has a ten-minute cooperative budget and a 20 GB MLX peak-allocation limit checked during generation. These are not OS-enforced limits; a blocked native call or SIGKILL cannot execute finalization. It performs no automatic retries; any new attempt gets its own directory and never overwrites old evidence.

The five-branch diagnostic adds examples with reversal, alongside the supplied reversal/uppercase rules and uppercase examples. These are the same development strings used previously, not a fresh holdout. A 7/8 supplied-rule gate is exploratory screening, not evidence of reliable deployment or internalized learning. The missing amber/urgent acquisition combination still requires an inductive assumption. No prospective claim or parameter intervention is authorized by passing this gate alone.

Reproduce with `uv sync --extra adaptation`, then `uv run --no-sync python scripts/pilot2_mlx.py --output evidence/NEW-RUN-NAME`. Default weights are the local `models/qwen3-4b-instruct` directory; hashes and cached download revision metadata are saved per run. Existing output directories are rejected.

Historical Docker scripts and evidence are retained as records, not silently rewritten to match the new runner. Before an actual acquisition comparison: complete closest-work full-text review; freeze task/evidence/targets and matched model/runtime; define prospective transfer cases and measure all retained information and acquisition costs.

## Verified repair result

`evidence/pilot-03-mlx` completed 40/40 calls: no evidence 0/8, supplied reversal rule 4/8, reversal examples 4/8, supplied uppercase rule 6/8, uppercase examples 8/8. Neither supplied-rule branch passed the declared 7/8 gate. No adapter training was started. Generation/load elapsed time was 14.26 seconds (excluding pre-run model hashing), peak MLX allocation 8,231,173,912 bytes. This is not total process/unified memory or total acquisition cost. Hash verification, unique branch/case coverage and exact-call scoring were independently checked after the run. The source snapshot matches the executed script. Existing scratch smoke artifacts were copied into `evidence/handoff-audit/mlx-route-recheck` so the resource checks no longer depend solely on gitignored scratch files.

Next research action: inspect the two supplied-uppercase-rule failures and independently test the checkpoint's instruction execution, before changing the task or interpreting example success as acquisition. No broad model or hyperparameter sweep is justified by this diagnostic.
