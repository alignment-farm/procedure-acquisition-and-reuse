# Bounded follow-up: lesson construction and transfer strata

This follow-up was approved after reviewing the initial campaign's limits. Its [protocol](../protocol/followup-v1.md) was committed as `5075482` before lesson revision and evaluation. It makes no new parameter update: the previously saved adapter is reused unchanged. The initial report remains a record of acquisition-v1 rather than being silently rewritten.

## Design and information access

Eight previously unused ordinary words and eight fresh random identifiers were matched by character length (two each at lengths 4, 6, 8, 10), then crossed with all four condition combinations. Each lexical stratum contains 24 inputs with observed acquisition combinations and eight with withheld amber/urgent. Length matching does not match tokenization or lexical frequency; these are descriptive strata, not a clean causal manipulation of lexicality. The identifiers form sixteen clusters, not 64 independent experimental units.

The lesson writer saw only the original twelve acquisition interactions, the previous candidate and all acquisition-validation outputs/expected calls. The supplied rule and follow-up inputs never entered revision or selection. Candidate 0 was the original lesson; at most two revisions were allowed. The first 12/12 acquisition-valid candidate would have been selected; otherwise highest score with earliest tie was the frozen fallback. Selection occurred before follow-up evaluation. Validation reuses acquisition inputs, so even passing would not establish transfer.

All five branches use the same native checkpoint/runtime, schema, greedy decoder, 96-token answer ceiling and fresh per-call cache. Branch order rotates. There is one pass of 320 follow-up calls, separately recorded from four checkpoint-replay checks and lesson construction/validation calls. No evaluation retries, manual edits, additional adapter training or test-based revisions were made.

## Lesson construction

The original lesson scored 0/12. Revision 1 scored 2/12. Revision 2 scored 0/12. None passed. The predefined fallback selected revision 1, explicitly **unvalidated**. It added tool names but still omitted uppercasing and contradicted itself about suffixes. Revision 2 gave a different inaccurate rule and did worse. No candidate was edited by the investigator. Thus the approved bounded attempt to obtain a validated evidence-derived lesson failed; this does not establish that such a lesson is impossible.

The selected lesson, verbatim:

> For any item, append "-2" if mode=amber, "-9" if mode=blue. Prefix with "dax" for amber, "wug" for blue. Priority (normal/urgent) only affects suffix: normal → -2, urgent → -9. Assumption: Unseen items follow the same rule; prefix is determined solely by mode.

## Fresh transfer results

| Representation | Ordinary seen /24 | Ordinary withheld /8 | Random seen /24 | Random withheld /8 | Total /64 |
|---|---:|---:|---:|---:|---:|
| No acquisition | 0 | 0 | 0 | 0 | 0 |
| All 12 examples | 24 | 7 | 19 | 1 | 51 |
| Selected **unvalidated** lesson | 4 | 0 | 4 | 0 | 8 |
| Supplied original rule | 24 | 8 | 24 | 8 | 64 |
| Unchanged adapter | 13 | 0 | 15 | 0 | 28 |

The adapter's deficits are not limited to random strings: it succeeds on 13/32 ordinary-word tasks versus examples' 31/32. It also fails all sixteen withheld-combination tasks, and scores only 28/48 on seen combinations with new items. The follow-up therefore separates two limitations: poor transfer to new items even under familiar condition combinations, and failure to compose the withheld combination on this set. It does not prove a particular internal mechanism or that a different acquisition method would fail.

Examples show a large descriptive lexical-stratum difference (31/32 ordinary versus 20/32 random), including 7/8 versus 1/8 on the withheld combination. With eight identifiers per stratum and character-length matching only, no significance or universal lexicality claim is made. The intended separable rule remains an inductive assumption that finite acquisition examples do not uniquely identify. The supplied rule is privileged additional information and provides an execution diagnostic, not an acquisition-equivalent competitor.

The selected lesson's 8/64 is not a validated-lesson benchmark. Its acquisition consistency had already failed. We report the required fallback outcome without relabeling it a successful construction method. The experiment leaves open whether a stronger evidence-derived lesson writer or validation procedure could perform well within an acceptable acquisition cost.

## Costs, resource use and verification

The run completed 362 model calls: 320 evaluation, 36 acquisition-validation, two revisions, four old-checkpoint replays. Original lesson generation cost 2.03 seconds (274 input/66 output tokens). Follow-up validation and revision added 17.09 generation/verification seconds (6,207 input/365 output tokens), for 19.12 seconds of cumulative lesson construction. All unsuccessful candidates and costs are retained.

| Representation | Follow-up inference seconds /64 uses | Input tokens | Output tokens |
|---|---:|---:|---:|
| No acquisition | 17.99 | 4,136 | 355 |
| Examples | 31.86 | 18,536 | 502 |
| Selected unvalidated lesson | 25.79 | 8,936 | 500 |
| Supplied rule | 24.26 | 7,528 | 509 |
| Unchanged adapter | 24.86 | 4,136 | 648 |

Inference timings exclude acquisition, switching and verification; the analysis records those separately. No extra adapter training occurred. Its original 13.16-second training cost (plus about 0.002 seconds target preparation) remains applicable to a fresh deployment, but was not incurred again here. The adapter's fewer prompt tokens and shorter inference time than examples do not establish repayment at comparable useful accuracy. We do not project a break-even from this lower-performing branch. No production caching, human labor, energy or dollar costs were measured.

Total run wall time, including hashing/load/invariants: 160.02 seconds. Peak MLX allocation: 9.63 GB, not total machine memory. All four prior adapter token streams were replayed exactly after loading the saved checkpoint. Saved adapter file and parameter hashes, initial adapter hash, base hash and reset probe logits matched; final checks confirmed unchanged adapter/base and restored logits. No optimizer was created and no parameter learning occurred. The follow-up records parameter replacement only to select the frozen branch.

The independent analysis verified the immutable run manifest, all 320 unique branch/input pairs, oracle scoring, the selected candidate and earliest-tie rule, acquisition-only validation coverage, and revision/replay counts. Nine tests pass, including fresh-set disjointness and every length/condition stratum. No new downloads, paid calls or external services were used.

## Scope conclusion and stop

The initial experiment and this approved follow-up are complete. Together they support a local conclusion: the fixed twelve-demonstration LoRA recipe fits training calls but transfers poorly relative to retained examples, including on new ordinary words, and does not demonstrate quality-matched acquisition-cost repayment. The bounded lesson-construction method also failed. These are valid negative outcomes for the tested methods, not an answer for all procedures, models, writers or training budgets.

Stop here under the approved bounds. We have not established when parameter storage becomes worthwhile, and have not obtained a validated compact lesson. A larger study would need a separately scoped acquisition method or stronger lesson construction, with new evaluation material and explicit additional supervision/cost. The initial PDF remains labeled as the initial experiment; this addendum and the revised README provide the current interpretation. No training sweep, correction/continual-learning extension, or external submission was performed.

## Evidence and reproduction

- [Frozen protocol](../protocol/followup-v1.md), [runner](../scripts/followup.py).
- [Candidates](../evidence/followup-v1/candidates.json), [selected lesson](../evidence/followup-v1/selected_lesson.txt), [event ledger](../evidence/followup-v1/events.jsonl).
- [Raw calls](../evidence/followup-v1/responses.jsonl), [evaluation cases](../evidence/followup-v1/evaluation.json), [checksums](../evidence/followup-v1/SHA256SUMS), [audited analysis](../evidence/followup-v1-analysis/analysis.json).

Use the pinned environment and model from the [initial reproduction note](2026-09-11-acquisition-results.md). The saved `evidence/acquisition-v1/adapter.safetensors` and initial evidence must be present and match their hashes.

```
uv run --no-sync python scripts/followup.py --output evidence/NEW-FOLLOWUP
uv run --no-sync python scripts/analyze_followup.py evidence/NEW-FOLLOWUP
uv run --no-sync python -m unittest discover -s tests -v
```

The runner rejects existing output directories. The script and dependencies are snapshotted in each run; no historical run files were changed.
