"""Development revision pilot-02: interface/task audit after pilot-01 failures.

Standard-library Docker client. Exploratory development material, not a
prospective test. Pilot-01 showed the 4B model could not execute the
string-reversal operation even with the correct rule supplied (it parroted
the schema placeholder 'ARG' and emitted phrases like 'dax reverse planet').
This revision changes two factors, separately identifiable:

1. A strict output contract that names the placeholders explicitly
   ('NAME ARGUMENT ... ARGUMENT is the transformed item string'), because
   pilot-01's 'dax ARG, wug ARG' phrasing was repeated verbatim by the model.
2. A simpler transformation candidate (UPPERCASE for amber) alongside the
   original reversal, both under the same strict contract.

Gate for a parameter-storage comparison: the supplied-correct-procedure
(oracle lesson) branch must reach at least 7/8 on the eight new development
cases (target 8/8). These eight inputs are development material; no
prospective set is touched.
"""
import hashlib, itertools, json, subprocess, time, urllib.request
from pathlib import Path

MODEL = 'huggingface.co/qwen/qwen3-4b-gguf:Q4_K_M'
LIMIT = 96
TEMPERATURE = 0

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
    out = Path('evidence/pilot-02'); out.mkdir(exist_ok=False)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    for label, cmd in {'model':['docker','model','inspect',MODEL], 'runtime':['docker','model','status']}.items():
        p = subprocess.run(cmd, capture_output=True, text=True)
        (out/f'{label}.txt').write_text(p.stdout + p.stderr)

    # Same six checked acquisition interactions as pilot-01 (each factor covered
    # separately; amber+urgent withheld as a combination), with uppercase answers.
    training = [(m,u,s) for m,u in [('amber',False),('blue',False),('blue',True)] for s in ['cat','fern']]
    examples = '\n'.join(case(*c) + ' -> ' + oracle_up(*c) for c in training)
    contexts = {
        'none': '',
        'lesson_reversal': RULE_REV,
        'lesson_upper': RULE_UP,
        'examples_upper': 'Checked successful interactions:\n' + examples,
    }
    oracles = {'none': oracle_up, 'lesson_reversal': oracle_rev, 'lesson_upper': oracle_up, 'examples_upper': oracle_up}
    # Same eight development cases as pilot-01 (new relative to acquisition strings).
    cases = list(itertools.product(['amber','blue'],[False,True],['planet','silver']))
    design = dict(exploratory=True,
        revision='pilot-02: strict output contract; uppercase transformation candidate; reversal retained under the same contract',
        purpose='execution-competence gate before any parameter-storage comparison; development material only',
        gate='oracle-lesson branch >= 7/8 on the eight new development cases (target 8/8)',
        branches=list(contexts), training=training, cases=cases,
        limit=LIMIT, temperature=TEMPERATURE)
    (out/'design.json').write_text(json.dumps(design, indent=2))

    rows = []; start = time.monotonic()
    with (out/'responses.jsonl').open('w') as f:
        for i, c in enumerate(cases):
            branches = list(contexts)[i%4:] + list(contexts)[:i%4]
            for branch in branches:
                expected = oracles[branch](*c)
                payload = dict(model=MODEL,
                               messages=[dict(role='user', content=SCHEMA + '\n' + contexts[branch] + '\n' + case(*c) + ' /no_think')],
                               temperature=TEMPERATURE, max_tokens=LIMIT, stream=False)
                t = time.monotonic(); row = dict(branch=branch, case=c, expected=expected, request=payload)
                try:
                    request = urllib.request.Request('http://localhost:12434/engines/v1/chat/completions',
                                                     data=json.dumps(payload).encode(),
                                                     headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(request, timeout=90) as r: raw = json.load(r)
                    action = (raw['choices'][0]['message'].get('content') or '').split('</think>')[-1].strip()
                    row.update(response=raw, action=action, correct=action == expected)
                except Exception as e:
                    row.update(error=repr(e), correct=False)
                row['seconds'] = time.monotonic() - t
                rows.append(row); f.write(json.dumps(row) + '\n'); f.flush()
                print(branch, c, row['correct'], round(row['seconds'], 2), flush=True)
                if 'error' in row: raise RuntimeError(row['error'])
                if time.monotonic() - start > 600: raise TimeoutError('Pilot 10-minute budget')
    summary = {b: dict(correct=sum(r['correct'] for r in rows if r['branch'] == b),
                       total=8,
                       seconds=sum(r['seconds'] for r in rows if r['branch'] == b),
                       prompt_tokens=sum(r['response']['usage']['prompt_tokens'] for r in rows if r['branch'] == b),
                       completion_tokens=sum(r['response']['usage']['completion_tokens'] for r in rows if r['branch'] == b)) for b in contexts}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    (out/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name for p in sorted(out.iterdir()) if p.is_file()))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
