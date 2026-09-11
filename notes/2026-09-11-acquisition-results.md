# Completed acquisition comparison — 11 September 2026

The bounded experiment is complete: one fixed adapter, no tuning after evaluation, 640 evaluation calls plus one lesson-generation call and twelve training-recall calls. The [frozen protocol](../protocol/acquisition-v1.md) was committed as `0aae909` before new-input evaluation. This note supersedes earlier statements that no task adapter has been trained.

## Behavior

| Representation | New inputs /64 | Seen combinations /48 | Withheld combination /16 |
|---|---:|---:|---:|
| No acquisition | 0 | 0 | 0 |
| All 12 examples | 41 | 39 | 2 |
| Generated lesson | 0 | 0 | 0 |
| Supplied correct rule | 64 | 48 | 16 |
| Fixed LoRA adapter | 26 | 22 | 4 |

Adapter training recall: 12/12. All branches reproduced their first-pass token streams on all 64 tasks during the second pass. Repeats are not independent samples. The unit of identifier diversity is 16 strings, crossed with four conditions. The adapter shows some new-input behavior beyond the no-acquisition baseline, but does not reliably transfer the intended procedure and trails retained examples overall. Its 4/16 versus examples' 2/16 on the withheld combination is too narrow to rescue an overall advantage. The intended rule's separability is an assumption; the acquisition examples do not uniquely specify it.

The one-shot generated lesson omitted tool names and uppercasing and contradicted itself about suffixes. It was not corrected. This measures a failed lesson-construction method, not the limits of explicit rules: the supplied-rule branch was perfect on these particular new identifiers. That branch has privileged information. Earlier ordinary-word execution failures do not contradict this result; these are different strings under the same fixed wording.

## Costs and invariants

The adapter's 96 updates took 13.16 seconds including final hashing/save, processing 6,568 input positions and 616 supervised answer/EOS tokens. It is 2,624,893 bytes. Lesson construction took 2.03 seconds, 274 input tokens and 66 output tokens. No retrieval search or retries were used.

Over 128 measured uses, examples consumed 37,192 prompt and 1,176 output tokens in 69.03 generation seconds. The adapter consumed 8,392 prompt and 1,128 output tokens in 47.76 generation seconds. Adding its update cost makes approximately 60.92 seconds, but yields 52 correct uses versus examples' 82. The nominal linear time crossing near 79 uses is **not a quality-matched break-even result**. A cheaper stream with fewer successes does not establish repayment. The full analysis preserves switching, verification, preparation, cumulative observed-use costs and explicitly labeled projections separately. Human labor, energy, money, deployment startup and production caching are not measured.

Full experiment wall time: 302.86 seconds; peak MLX allocation: 9.64 GB. This is not total system memory. Base parameter hashes were unchanged. Reset probe logits were exactly restored before and after evaluation. All twelve answer-masked training spans independently decoded to the intended target plus EOS. The model files matched the earlier native diagnostic hashes. No paid model calls or new model downloads were used.

## Conclusion and stop decision

This recipe fit the examples but did not make parameter storage competitive with retained examples at comparable useful performance. The generated lesson was an acquisition failure of its own. There is no observed quality-matched repayment in the tested workload. This does not show that better training, richer experience, or learned writers cannot succeed.

The initial campaign stops here, as specified by the brief: no sweep to obtain an adapter win and no correction/continual-learning extension. A larger campaign is not justified solely to improve this score. A future scoped study could test an evidence-derived lesson with independent validation or a stronger acquisition objective, but must freeze new evaluation inputs and explicitly account for added supervision. No external submission has been made.

## Evidence and reproduction

- [Four-page local report](../output/pdf/procedure-acquisition.pdf), [LaTeX source](../paper/procedure-acquisition.tex), [bibliography](../paper/references.bib).
- [Raw run](../evidence/acquisition-v1/responses.jsonl), [event ledger](../evidence/acquisition-v1/events.jsonl), [final adapter](../evidence/acquisition-v1/adapter.safetensors), [checksums](../evidence/acquisition-v1/SHA256SUMS).
- [Audited analysis](../evidence/acquisition-v1-analysis/analysis.json), [analysis script](../scripts/analyze_acquisition.py).

Run `uv sync --extra adaptation`. Use the model files from `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, under `models/qwen3-4b-instruct`; the run checks their exact hashes against `evidence/pilot-03-mlx/model.json`. Then:

```
uv run --no-sync python scripts/acquisition.py --output evidence/NEW-RUN
uv run --no-sync python scripts/analyze_acquisition.py evidence/NEW-RUN
uv run --no-sync python -m unittest discover -s tests -v
```

The script rejects existing output directories. Original execution sources are snapshotted with the run. The tiny adapter is committed explicitly despite the general weight-file ignore rule. The paper build is documented in `paper/README.md`.
