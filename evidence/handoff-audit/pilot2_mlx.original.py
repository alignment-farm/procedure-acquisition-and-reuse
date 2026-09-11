"""Pilot-02 (MLX-native): development revision for the interface/task audit.

Supersedes the aborted Docker attempt (evidence/pilot-02-attempt-1, which hit
a backend 'Compute error' 500 while a concurrent client was loading a 27B
model onto the shared local runner). Runtime here is native MLX with
Qwen/Qwen3-4B-Instruct-2507 (bf16), the same instruct line as pilot-01's
Docker Q4_K_M GGUF, so the development diagnostic and the frozen comparison
share one model/runtime. Prompt rendering uses the checkpoint's own HF chat
template with enable_thinking=False (no /no_think text marker needed).

Branches (5 x 8 cases = 40 greedy calls):
  none              strict contract, no evidence          (expected: uppercase oracle)
  lesson_reversal   strict contract + reversal rule       (expected: reversal oracle)
  lesson_upper      strict contract + uppercase rule      (expected: uppercase oracle)
  examples_reversal strict contract + 6 reversal demos    (expected: reversal oracle)
  examples_upper    strict contract + 6 uppercase demos   (expected: uppercase oracle)

Exploratory development material only; not a prospective test.
"""
import hashlib, itertools, json, os, time, importlib.metadata as md
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models' / 'qwen3-4b-instruct'
OUT = Path(__file__).resolve().parent.parent / 'evidence' / 'pilot-02'
LIMIT = 96
TEMPERATURE = 0.0

SCHEMA = ('Route an item using an unfamiliar two-tool protocol. '
          'Your reply must be exactly one line in the form NAME ARGUMENT, where NAME is the tool '
          'and ARGUMENT is the transformed item string. Write no other text. ')

RULE_REV = ('For amber mode, use tool dax and reverse the item string so it reads backwards. '
            'For blue mode, use tool wug and keep the item unchanged. '
            'For urgent priority, append -9 to the transformed item; for normal priority, append -2.')

RULE_UP = ('For amber mode, use tool dax and write the item string in UPPERCASE. '
           'For blue mode, use tool wug and keep the item unchanged. '
           'For urgent priority, append -9 to the transformed item; for normal priority, append -2.')

def oracle_rev(mode, urgent, item):
    return ('dax ' + item[::-1] if mode == 'amber' else 'wug ' + item) + ('-9' if urgent else '-2')

def oracle_up(mode, urgent, item):
    return ('dax ' + item.upper() if mode == 'amber' else 'wug ' + item) + ('-9' if urgent else '-2')

def case(mode, urgent, item):
    return f'mode={mode}, priority={"urgent" if urgent else "normal"}, item={item}'

def main():
    import mlx.core as mxc
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    OUT.mkdir(exist_ok=False)
    (OUT / 'runner.py').write_bytes(Path(__file__).read_bytes())
    cfg = json.loads((MODEL_DIR / 'config.json').read_text())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(MODEL_DIR.glob('*.safetensors'))}
    (OUT / 'model.txt').write_text(json.dumps(dict(
        repo='Qwen/Qwen3-4B-Instruct-2507', revision='main',
        config={k: cfg.get(k) for k in ('architectures', 'hidden_size', 'num_hidden_layers',
                                        'num_attention_heads', 'num_key_value_heads', 'torch_dtype',
                                        'max_position_embeddings', 'vocab_size')},
        weight_sha256=hashes,
        note=('Docker artifact used for pilot-01 and pilot-02-attempt-1 was '
              'huggingface.co/qwen/qwen3-4b-gguf:Q4_K_M (Qwen3 4B Instruct, AWQ-derived); '
              'its exact checkpoint identity is not recoverable from GGUF metadata. '
              'This MLX artifact is the bf16 HF checkpoint of the same instruct line.'),
    ), indent=2))
    (OUT / 'runtime.txt').write_text(json.dumps(dict(
        python=os.python_version(),
        mlx=mxc.__version__, mlx_lm=md.version('mlx-lm'),
        mlx_lm_commit='86b48c461feebf87c58788655b7e57b5574b9e6d',
        hardware='MacBook Pro, Apple M3 Max, 16 cores, 48 GB unified memory',
        decoding='greedy (temperature 0)',
        time_note='timestamps are host-local EDT; the Docker VM clock in this environment ran ~11 h fast',
    ), indent=2))

    training = [(m, u, s) for m, u in [('amber', False), ('blue', False), ('blue', True)] for s in ['cat', 'fern']]
    ex_rev = '\n'.join(case(*c) + ' -> ' + oracle_rev(*c) for c in training)
    ex_up = '\n'.join(case(*c) + ' -> ' + oracle_up(*c) for c in training)
    contexts = {
        'none': '',
        'lesson_reversal': RULE_REV,
        'lesson_upper': RULE_UP,
        'examples_reversal': 'Checked successful interactions:\n' + ex_rev,
        'examples_upper': 'Checked successful interactions:\n' + ex_up,
    }
    oracles = {'none': oracle_up, 'lesson_reversal': oracle_rev, 'lesson_upper': oracle_up,
               'examples_reversal': oracle_rev, 'examples_upper': oracle_up}
    cases = list(itertools.product(['amber', 'blue'], [False, True], ['planet', 'silver']))
    design = dict(exploratory=True,
        revision=('pilot-02 MLX-native: strict output contract; uppercase transformation candidate; '
                  'reversal retained under the same contract; MLX runtime shared with the planned '
                  'frozen comparison (Docker attempt aborted by concurrent 27B load, see attempt-1)'),
        purpose='execution-competence gate before any parameter-storage comparison; development material only',
        gate='oracle-lesson branch >= 7/8 on the eight new development cases (target 8/8)',
        branches=list(contexts), training=training, cases=cases,
        limit=LIMIT, temperature=TEMPERATURE)
    (OUT / 'design.json').write_text(json.dumps(design, indent=2))

    t0 = time.monotonic()
    model, tokenizer = mlx_lm.load(str(MODEL_DIR))
    load_s = time.monotonic() - t0
    sampler = make_sampler(model, temperature=TEMPERATURE)

    rows = []; start = time.monotonic()
    with (OUT / 'responses.jsonl').open('w') as f:
        for i, c in enumerate(cases):
            branches = list(contexts)[i % 5:] + list(contexts)[:i % 5]
            case_line = case(*c)
            for branch in branches:
                rendered = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': SCHEMA + '\n' + contexts[branch] + '\n' + case_line}],
                    add_generation_prompt=True, enable_thinking=False)
                expected = oracles[branch](*c)
                t = time.monotonic()
                row = dict(branch=branch, case=c, expected=expected, prompt=rendered)
                try:
                    prompt_tokens = tokenizer.encode(rendered)
                    out = model.generate(prompt_tokens, sampler=sampler, max_tokens=LIMIT, verbose=False)
                    text = tokenizer.decode(out[len(prompt_tokens):], skip_special_tokens=True)
                    action = text.split('</think>')[-1].strip()
                    n_new = len(out) - len(prompt_tokens)
                    row.update(prompt_tokens=len(prompt_tokens), completion_tokens=n_new,
                               raw=text, action=action, correct=action == expected,
                               truncated=n_new >= LIMIT)
                except Exception as e:
                    row.update(error=repr(e), correct=False)
                row['seconds'] = round(time.monotonic() - t, 3)
                rows.append(row); f.write(json.dumps(row) + '\n'); f.flush()
                print(branch, c, row['correct'], row['seconds'], flush=True)
                if 'error' in row: raise RuntimeError(row['error'])
                if time.monotonic() - start > 600: raise TimeoutError('Pilot 10-minute budget')
    summary = dict(load_seconds=round(load_s, 2), total_seconds=round(time.monotonic() - t0, 2),
                   peak_memory_gb=round(float(mxc.metal.get_peak_memory() / 2**30), 2),
                   branches={b: dict(correct=sum(r['correct'] for r in rows if r['branch'] == b),
                                     total=8,
                                     seconds=round(sum(r['seconds'] for r in rows if r['branch'] == b), 3),
                                     prompt_tokens=sum(r.get('prompt_tokens', 0) for r in rows if r['branch'] == b),
                                     completion_tokens=sum(r.get('completion_tokens', 0) for r in rows if r['branch'] == b))
                            for b in contexts})
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'SHA256SUMS').write_text('\n'.join(
        hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name
        for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'SHA256SUMS'))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
