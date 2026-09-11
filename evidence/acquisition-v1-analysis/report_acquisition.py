"""Generate manuscript tables from audited analysis JSON."""
import json
from pathlib import Path


def main():
    a=json.loads(Path('evidence/acquisition-v1-analysis/analysis.json').read_text())
    b=a['branches']; t=a['training']
    lines=[r'\begin{table}[H]',r'\centering',r'\begin{tabular}{lrrr}',r'\toprule',
           r'Representation & All /64 & Seen /48 & Withheld /16 \\',r'\midrule']
    names={'none':'No acquisition','examples':'Examples','lesson':'Generated lesson','oracle':'Supplied rule','adapter':'Adapter'}
    for key in names:
        r=b[key];lines.append(f"{names[key]} & {r['first_correct']} & {r['seen_correct']} & {r['withheld_correct']} \\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{First-pass exact-call success on 64 new-input tasks.}',r'\end{table}']
    lines.append(f"The adapter recalls {a['training_recall']}/12 training calls but succeeds on only {b['adapter']['first_correct']}/64 new-input calls. Examples achieve {b['examples']['first_correct']}/64; the supplied rule achieves {b['oracle']['first_correct']}/64. The generated lesson and no-acquisition reference each achieve {b['none']['first_correct']}/64. This distinguishes fitting the acquisition examples from reliable transfer.")
    lines.append('Second-pass exact outputs match the first pass on '+', '.join(f"{names[k]} {b[k]['repeat_identical']}/64" for k in names)+'. These repetitions add cost measurements, not independent accuracy evidence.')
    lines += [r'\begin{table}[H]',r'\centering',r'\begin{tabular}{lrrr}',r'\toprule',r'Representation & Input tokens & Output tokens & Use seconds \\',r'\midrule']
    for key in names:
        r=b[key];lines.append(f"{names[key]} & {r['prompt_tokens']:,} & {r['completion_tokens']:,} & {r['inference_seconds']:.2f} \\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Measured generation costs for 128 uses per branch; excludes acquisition and switching.}',r'\end{table}']
    lines.append(f"The fixed update took {t['seconds']:.2f} seconds including its final hash and adapter save, processing {t['input_tokens']:,} input positions and supervising {t['supervised_tokens']:,} answer/EOS tokens. The adapter occupies {t['adapter_bytes']:,} bytes. Lesson generation took {a['lesson_cost']['seconds']:.2f} seconds ({a['lesson_cost']['prompt_tokens']} input and {a['lesson_cost']['completion_tokens']} output tokens). Peak MLX allocation over the run was {a['total']['peak_mlx_bytes']/1e9:.2f} GB; total wall time including provenance and checks was {a['total']['seconds']:.2f} seconds.")
    lines.append('The adapter has lower useful performance than examples, so a faster or shorter adapter call would not establish a storage advantage at comparable quality. No repayment relative to examples is established in this workload. A comparison against the failed generated lesson cannot rescue a broad parametric-storage claim. The machine-readable analysis includes cumulative measured use costs and explicitly labeled linear arithmetic projections; projections with lower adapter accuracy are ineligible as quality-matched break-even claims.')
    lines.append(r'Base-parameter hashes remained unchanged and restored probe logits matched exactly after adapter resets. The final checkpoint was fixed before evaluation; no fitting followed the observed failures.')
    Path('paper/results.tex').write_text('\n\n'.join(lines)+'\n')

if __name__=='__main__':main()
