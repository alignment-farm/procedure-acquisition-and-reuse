# Conditional execution: fixed component diagnostic

Protocol written before running pilot-05. Use the same six development strings and all four mode/priority combinations, same native checkpoint/runtime, greedy generation and 96-token ceiling. Six branches, 24 calls each, total 144. Rotate branch order. Ten-minute cooperative runtime budget, 20 GB MLX allocation ceiling, no retries. No new training or holdout evaluation.

Branches: routing only, conditional item transformation only, suffix selection only, formatting given correct intermediate values, complete original rule, and complete rule in a single prewritten table representation. Isolated branches retain mode/priority/item fields where applicable. Formatting receives privileged oracle intermediates and must not be mistaken for a model-generated pipeline. Repeated routing/suffix answers are low-diversity diagnostics, not independent evidence of broad competence.

Score exact responses. Also inspect full-call errors by tool, transformed item, suffix and strict format, without repairing outputs or changing the primary score. Component successes with full-call failures suggest a composition/interface difficulty; they do not identify a neural mechanism. A table improvement is a representation effect for this development set, not a claim of general reliability. Stop after this fixed diagnostic rather than searching for passing prompts. Freeze a subsequent acquisition protocol with residual execution error explicitly measured.

## Results and interpretation

All 144 calls completed without errors in 39.22 seconds including load and excluding provenance hashing. Peak MLX allocation was 8,186,723,912 bytes. Exact scores:

| Branch | Correct |
|---|---:|
| Routing only | 24/24 |
| Conditional transformation only | 24/24 |
| Suffix selection only | 24/24 |
| Formatting supplied intermediate values | 24/24 |
| Complete original rule | 21/24 |
| Complete table rule | 22/24 |

The original rule reproduced the three prior failures exactly: PLANTET for PLANET, HARBOUR for HARBOR, and AMBER for PLANET. In all 24 original-rule calls the tool, suffix and strict format were correct. The errors concern the transformed item under composition. Isolated transformation is successful on these same conditions, so the evidence supports a compositional execution difficulty rather than inability to perform the individual operations. It does not prove why the model makes those mistakes or establish general performance beyond this small reused set.

The table fixed the harbor case but produced `dax PLANTEN` and `dax PLANTEN9` for the two amber/planet conditions. It is not a general solution. The analysis script's secondary field diagnostics require a parseable full-call grammar; unparseable table responses fail all diagnostic fields conservatively, even where a visible tool token happens to be correct. The primary exact score is unaffected by this convention.

## Decision: execution characterized sufficiently to proceed

Retain the original interface for continuity, not the table selected for one extra development success. Stop prompt search. Replace the earlier pass/fail screening gate with a measured supplied-rule reference in the next acquisition comparison. This is a material methodological revision: early screening treated rule-execution failure as a reason to delay learning tests; component evidence now shows that useful individual competence coexists with residual composition errors. Perfect instruction execution is not required to compare memory representations. Report whole-call performance alongside component errors and compare costs only at comparable useful performance.

Do not remove planet/harbor or use these six words as prospective evaluation cases. Do not convert the isolated-component results into an asserted 24/24 multi-call pipeline: the formatting branch received oracle intermediate values. If a pipeline is evaluated later, feed actual generated values and count every call and failure.

Before training, the next scoped step is the closest-work full-text review and a frozen acquisition protocol: same native checkpoint/runtime across representations, disclosed checked evidence and supervision, no-evidence and supplied-rule controls, untouched evaluation words/conditions, and bounded acquisition costs. A learned lesson must be generated from acquisition evidence rather than relabeling the researcher rule. Any rule-derived training augmentation must be separately disclosed. The study question is unchanged; no parameter acquisition or amortization conclusion has been reached.

## Reproduction and validation

Run `uv run --no-sync python scripts/pilot2_mlx.py --suite components --output evidence/NEW-RUN-NAME` with the pinned adaptation environment and local model described in the handoff audit. The run snapshots both runner and task module. Read raw calls in `evidence/pilot-05-components/responses.jsonl` and the immutable checksum manifest. Generate the analysis with `uv run --no-sync python scripts/analyze_components.py evidence/pilot-05-components`; the saved output is `evidence/pilot-05-analysis/analysis.json`.

Six unit tests pass, covering the historical pilot, independent oracle fixtures, component-suite coverage, and separation of diagnostic formatting from exact scoring. Analysis verifies the run manifest, exact request order, unique identifiers and all stored success labels against raw actions. No remote calls or model downloads were used in this diagnostic.
