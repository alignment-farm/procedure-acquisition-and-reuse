# Execution diagnostic, fixed before running

The pilot-03 supplied-uppercase failures were `dax PLANTET-2` (a spelling error) and `dax AMBER-9` (transforming the mode instead of the item), both for item=planet. This does not justify saying the model cannot uppercase strings. Separate operation competence from applying a conditional instruction.

Use the same local native model, greedy sampler and token budget. Compare the original rule with an ordered-step equivalent on all four mode/priority combinations for planet, silver, harbor, cobalt, meadow and ticket (48 requests). Add six uppercase-only requests to check the string operation. Total 54 calls, no retries or prompt changes within the run. The four additional words are development inputs; neither they nor the earlier inputs become prospective evaluation material. This is one bounded diagnostic, not a search for passing examples.

Interpretation: operation-only success with routing failures locates difficulty in conditional execution rather than the operation alone. Stepwise success is prompt sensitivity, not parametric acquisition. All branch accuracies and individual errors will be reported. No inference about amortization is available here.

## Results

All 54 calls completed without errors in 19.06 seconds including model load, excluding provenance hashing. Peak MLX allocation: 8,197,471,864 bytes. Original rule: 21/24; ordered steps: 12/24; uppercase-only: 6/6. Saved evidence: `evidence/pilot-04-execution/`.

Original-rule errors were PLANTET for PLANET, HARBOUR for HARBOR, and AMBER for PLANET. Ordered steps failed all six amber/normal cases (choosing wug), five blue/normal cases (an extra space before the suffix), and amber/urgent planet (PLANT). Thus the rewrite altered both routing and formatting behavior. Strict exact-call scoring was retained, and no repair/retry was applied. Uppercasing these six isolated words is within the model's observed competence. Conditional instruction execution is sensitive to wording. The broader 21/24 result does not retroactively pass pilot-03's 7/8 gate; the original eight cases still include both failures.

Decision: stop prompt revisions at this checkpoint. Preserve both wordings and all failures. The next acquisition design should report component and whole-call success, use a fully specified acquisition domain or explicitly state its composition assumptions, and treat a supplied-rule condition as a measured diagnostic rather than selecting tasks until that condition passes. Before training, finish the closest-work review and freeze a bounded matched comparison with genuinely untouched evaluation material. No parameter acquisition or cost repayment has been tested.

## Refactor verification

Task definitions now live in `scripts/procedure_task.py`; the MLX runner handles provenance, generation, budgets and failure reporting. It requires an explicit fresh output path, snapshots both source files, and supports `--suite pilot` (the prior 40-call design) or `--suite execution` (this 54-call diagnostic). Three unit tests check independent oracle fixtures, exact agreement with pilot-03 input/target/order, and execution-suite coverage. Run `uv run --no-sync python -m unittest discover -s tests -v`.

Reproduction: `uv run --no-sync python scripts/pilot2_mlx.py --suite execution --output evidence/NEW-RUN-NAME`, after the adaptation environment and local checkpoint are installed as described in the handoff audit. Original run sources remain immutable in their evidence directories.
