# Procedure acquisition and reuse

This study asks when repeated use makes learning from experience more useful
than repeatedly consulting that experience. Begin with one small tool-use
procedure learned from checked interactions, then evaluate its application to
new inputs. Compare retained examples, a compact explicit lesson, and a
parameter update such as an adapter.

This is an independent investigation arising from Construct-2's
[S2 acquisition and revision question](../../construct-2/sources/THEORIES_AND_QUESTIONS.md#s2-can-a-learned-procedure-survive-later-learning-and-accept-a-scoped-correction)
and [S5 retention and consolidation question](../../construct-2/sources/THEORIES_AND_QUESTIONS.md#s5-when-should-a-useful-temporary-update-be-retained-or-consolidated).

## Question

**When does repeated use justify learning a procedure into parameters rather
than retaining and consulting explicit experience?**

Repeated application might repay the cost of acquisition by improving later
performance or reducing the resources needed per successful task. Alternatively,
explicit examples or a compact lesson may remain more reliable and cheaper,
or apparent learning may be memorization of training answers. A useful procedure
must be applied to new inputs; the representation's name does not establish
what was acquired.

## Initial expectation

Begin with a focused investigation: read the closest work, identify a small
procedure and an interpretable transfer comparison, and assess a feasible local
implementation. Develop a bounded empirical comparison where feasible. The
investigator owns the exact task, methods, budgets, and protocol and may revise
the question in light of prior work or observations.

The intended first outcome is a short local research note explaining what is
already answered, what was investigated, what the evidence supports, and whether
a larger investigation is worthwhile. A failure to acquire the procedure, an
explicit-memory advantage, or no observed repayment of training cost is a valid
outcome. Do not expand the search merely to obtain a parameter-learning win.

No numerical time, compute, or spending cap has been specified. Keep the initial
work focused and bring substantial expansion or new resource requirements back
to the user. A later scoped correction is a possible extension after acquisition
and reuse are understood, not a required second campaign.

## Starting evidence and reading

The [update-source-selection study](../update-source-selection/README.md) is a
completed initial feasibility investigation, not this study's unfinished first
phase. Its [10 September 2026 report](../update-source-selection/output/pdf/update-source-selection.pdf)
documents a native MLX LoRA route on a 48 GB Apple M3 Max, exact adapter resets,
unchanged base weights, and changed generation streams. Its final full-context
comparison completed 24/24 legs with every source and with no update. It did
not establish procedural acquisition or retention without the original evidence.
Inspect its implementation for reusable components and preserve provenance;
its two-step update recipe and synthetic task are not prescribed here.

The parent's [evidence and paper map](../../construct-2/sources/THEORIES_AND_QUESTIONS.md)
provides versioned reading leads. Start with its assessments of:

- **SEAL (P4):** adaptation-data generation and evaluation after removing the
  original passage; relevant to acquisition and the training target.
- **Beyond Perplexity (P5):** separating lower loss from useful retained behavior.
- **PERK (P8):** parametric context storage and later reasoning; relevant to
  what existing methods already demonstrate.
- **ACE (P16):** explicit contextual playbooks and their adaptation/use costs.

These are starting leads from the parent's review, not a completed literature
review for this question. Check the closest procedural-learning, distillation,
and amortized-computation work before claiming novelty. A replication or a
careful explanation may be a useful first contribution. The parent's
[research perspective](../../construct-2/notes/PERSPECTIVES.md) motivates comparing
representations by how knowledge is used and revised; its
[Construct synthesis](../../construct-2/notes/RESEARCH_BRIEF.md) supplies bounded
evidence for useful explicit memory.

## Possible first comparison

Choose a small tool-use procedure whose behavior can be checked independently
and whose later inputs require applying a rule rather than copying a training
answer. For example, an unfamiliar tool could require different argument
transformations or call sequences under observable conditions. The task should
have a reason for repeated use beyond making training look economical.

Give branches the same checked acquisition evidence and distinguish what each
retains for later use:

- Examples available through a disclosed context or retrieval policy.
- A compact explicit lesson derived from that evidence.
- An adapter trained from that evidence or disclosed derived targets, evaluated
  without the original examples or lesson in its prompt.

Include a no-acquisition reference and a supplied-correct-procedure diagnostic
where useful for separating learning failures from inability to execute the
task. Use the same base model and runtime where feasible. Account for any
additional teacher knowledge, supervision, or data generation; a superior
training target must not be mistaken for an advantage of parameter storage.

Test new inputs and new combinations of the procedure's conditions. Changing
names alone is weak evidence of transfer. Keep acquisition/development material
separate from a prospective evaluation of a developed claim. Exploratory task
and interface changes should remain identifiable.

Measure acquisition cost and cumulative performance/cost over repeated uses.
Report construction, training, retrieval, input processing, generation including
reasoning, verification, and retries as applicable. Keep tokens, elapsed time,
memory, and compute measures separate unless a justified conversion combines
them. Compare costs at comparable useful performance; cheap failures do not
establish an advantage. Distinguish measured repetitions from extrapolated
break-even estimates. Repeated use may never repay acquisition in the tested
workload.

A subsequent scoped change to the procedure could test correction cost and
unaffected behavior. That extension should follow what the initial evidence
supports. These comparisons are design suggestions, not a frozen protocol.

## Resources

- Open-weight models through `docker model` are preferred.
- OpenAI models through `codex` and SpaceXAI models through `agent` are available
  resource routes; their useful roles depend on the interfaces provided.
- Use `uv` for Python and Docker/Compose for models or supporting services where
  appropriate.

Inspect this session's hardware, installed models, and training access before
sizing experiments. The sibling study's verified MLX route is a starting point,
not proof that the same resources are available here. Inference-only endpoints
can supply controls or candidate lessons but cannot perform the parameter
intervention. No paid API or remote-compute campaign is commissioned by this brief.

## Findings and publication

**The initial experiment and one approved bounded follow-up are complete (11 September 2026).** They do not settle the general question of when parameter storage is worthwhile. In the tested setup, a fixed twelve-demonstration LoRA adapter recalled the training calls but transferred poorly relative to retained examples. No acquisition-cost repayment at comparable useful accuracy was demonstrated. The bounded attempt to construct a validated compact lesson also failed.

The [follow-up results and current interpretation](notes/2026-09-11-followup-results.md) separate new ordinary words, random identifiers and the withheld condition combination:

| Representation | Ordinary words /32 | Random identifiers /32 | Withheld combination /16 |
|---|---:|---:|---:|
| Retained examples | 31 | 20 | 8 |
| Unchanged adapter | 13 | 15 | 0 |
| Selected **unvalidated** lesson | 4 | 4 | 0 |
| Supplied correct rule (privileged diagnostic) | 32 | 32 | 16 |
| No acquisition | 0 | 0 | 0 |

The adapter's failures were not confined to random strings. The generated lesson and its two allowed revisions scored 0/12, 2/12 and 0/12 on acquisition-only validation; none passed. We selected the highest-scoring candidate by the frozen fallback rule and labeled it unvalidated. This is not evidence against all compact lessons or stronger learning methods. The dataset is small, the composition rule requires an inductive assumption, and lexical strata are matched only by character length.

The current bounded investigation stops here, without a training sweep. A larger campaign would need a separately scoped acquisition or lesson-construction method and fresh evaluation material. No later-learning/correction extension or external submission has been performed.

### Reports and reproduction

- **Current addendum:** [bounded follow-up results](notes/2026-09-11-followup-results.md), [frozen protocol](protocol/followup-v1.md), [audited analysis](evidence/followup-v1-analysis/analysis.json). All 362 calls completed, including 320 fresh evaluation calls, 36 acquisition-validation calls, two lesson revisions and four saved-adapter replays. Adapter/base invariants passed.
- **Initial experiment:** [four-page PDF](output/pdf/procedure-acquisition.pdf), [LaTeX source](paper/procedure-acquisition.tex), [bibliography](paper/references.bib), [results and reproduction instructions](notes/2026-09-11-acquisition-results.md), [frozen protocol](protocol/acquisition-v1.md). On its distinct random-identifier set, the adapter scored 26/64 versus examples' 41/64 and a supplied rule's 64/64; training recall was 12/12. The initial generated lesson scored 0/64. Its PDF is preserved as the initial report; read the addendum for the current scope and conclusion.
- **Evidence:** [initial run](evidence/acquisition-v1/summary.json), [saved adapter](evidence/acquisition-v1/adapter.safetensors), [follow-up run](evidence/followup-v1/summary.json). Run directories contain raw prompts, outputs, cases, source snapshots, costs, failures where applicable, and checksums. The adapter was not retrained in the follow-up.

Use `uv sync --extra adaptation` and the exact local model revision identified in the reproduction notes. New runs require a fresh output directory. The [paper build instructions](paper/README.md) reproduce the initial PDF. Tests run with `uv run --no-sync python -m unittest discover -s tests -v`.

This study publishes locally. The [ancillary-study guidance](../../construct-2/notes/ANCILLARY_STUDY.md) describes its relationship to Construct-2, which reads these findings and updates its broader theories.

### Exploratory history

These records explain task development and implementation decisions; they are not additional prospective tests or current blockers.

- [Initial Docker pilot](notes/2026-09-10-start.md): 24 calls; examples 4/8, supplied rule 1/8, no evidence 0/8. No task adapter had been trained at that stage.
- [Resource recheck](notes/2026-09-10-resources-recheck.md) and [handoff audit](notes/2026-09-11-handoff-audit.md): preserved the interrupted Docker run, corrected unsupported failure interpretations and repaired the native MLX draft.
- [Execution diagnostic](notes/2026-09-11-execution-diagnostic.md): 54 calls; uppercase alone 6/6, original conditional rule 21/24, ordered-step rewrite 12/24.
- [Component diagnostic](notes/2026-09-11-components.md): 144 calls; routing, conditional transformation, suffix selection and supplied-intermediate formatting each 24/24; complete original/table rules 21/24 and 22/24. This justified measuring residual execution errors rather than demanding a perfect supplied-rule gate before the acquisition experiment.
- [Closest-work review](notes/2026-09-11-literature.md): versioned primary-source method comparisons and implementation provenance; this investigation is not a replication of SEAL, PMD, PERK or ACE and claims no novel learning mechanism.
