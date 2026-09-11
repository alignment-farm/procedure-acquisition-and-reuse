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

Add links here as research notes, evidence, and a manuscript become available.
Keep methods, code, configurations, saved outputs, failures, and reproduction
instructions in this project. The intended publication format is LaTeX source,
a bibliography, figures as needed, and a compiled PDF, with effort proportional
to the investigation. An early useful finding need not wait for publication
polish. External submission is a separate user decision.

The [ancillary-study guidance](../../construct-2/notes/ANCILLARY_STUDY.md) describes
the relationship: this project investigates and publishes locally; Construct-2
reads those publications and updates its broader theories.

### Initial exploration — 10 September 2026

The [initial protocol and results](notes/2026-09-10-start.md) document a completed
24-call local Docker pilot, versioned reading leads, raw evidence and reproduction
instructions. Examples scored 4/8, no evidence 0/8, and a supplied correct rule
1/8 on an unfamiliar tool-argument task. Execution competence is not yet
established; these exploratory results do not test parameter acquisition or
repayment. No adapter has been trained in this study yet.

### Handoff review — 11 September 2026

The [handoff audit](notes/2026-09-11-handoff-audit.md) selects preservation and
refactoring. It documents the partial Docker run, corrections to unsupported
failure annotations, and repairs to the unexecuted native MLX draft. Historical
outputs remain intact; the native checkpoint is a distinct exploratory condition.
The repaired [native diagnostic results](evidence/pilot-03-mlx/summary.json)
completed 40 calls: uppercase examples 8/8, supplied uppercase rule 6/8,
reversal examples and rule each 4/8, no evidence 0/8. The declared supplied-rule
gate remains unmet; no task adapter has been trained.

The subsequent [execution diagnostic](notes/2026-09-11-execution-diagnostic.md)
completed 54 calls: uppercase-only 6/6, original conditional rule 21/24,
ordered-step rewrite 12/24. This locates the observed difficulty in conditional
execution and prompt sensitivity; it does not establish parameter learning.
The refactored runner shares independently checked task definitions and saves
both source files with each run.

The [component diagnostic](notes/2026-09-11-components.md) completed 144 calls:
routing, conditional transformation, suffix selection and formatting each passed
24/24 in isolation; complete instructions scored 21/24 (original) and 22/24
(table). We retain the original interface and stop prompt search. Residual
composition errors will be measured in the acquisition comparison rather than
requiring perfect supplied-rule execution before proceeding.
