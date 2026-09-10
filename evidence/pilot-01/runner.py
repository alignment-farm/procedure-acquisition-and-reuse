"""Exploratory unfamiliar-tool argument protocol. Standard-library Docker client."""
import hashlib, itertools, json, subprocess, time, urllib.request
from pathlib import Path
MODEL = 'huggingface.co/qwen/qwen3-4b-gguf:Q4_K_M'
RULE = 'For amber mode, use tool dax and reverse the item string. For blue mode, use tool wug and keep the item unchanged. For urgent priority, append -9 to the transformed item; for normal priority append -2. Output TOOL ARG only.'
SCHEMA = 'Route an item using an unfamiliar tool protocol. Available tools: dax ARG, wug ARG. Output exactly one tool call as plain text, no explanation. '
def oracle(mode, urgent, item):
    return ('dax ' + item[::-1] if mode == 'amber' else 'wug ' + item) + ('-9' if urgent else '-2')
def case(mode, urgent, item):
    return f'mode={mode}, priority={"urgent" if urgent else "normal"}, item={item}'
def main():
    out = Path('evidence/pilot-01'); out.mkdir(exist_ok=False)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    for label, cmd in {'model':['docker','model','inspect',MODEL], 'runtime':['docker','model','status']}.items():
        p = subprocess.run(cmd,capture_output=True,text=True)
        (out/f'{label}.txt').write_text(p.stdout+p.stderr)
    # Acquisition covers each factor separately; amber+urgent withheld as a combination.
    training = [(m,u,s) for m,u in [('amber',False),('blue',False),('blue',True)] for s in ['cat','fern']]
    examples = '\n'.join(case(*c)+' -> '+oracle(*c) for c in training)
    contexts = {'none':'', 'examples':'Checked successful interactions:\n'+examples, 'oracle_lesson':RULE}
    cases = list(itertools.product(['amber','blue'],[False,True],['planet','silver']))
    (out/'design.json').write_text(json.dumps(dict(exploratory=True,training=training,cases=cases,contexts=contexts,rule=RULE,limit=96,temperature=0),indent=2))
    rows=[]; start=time.monotonic()
    with (out/'responses.jsonl').open('w') as f:
        for i,c in enumerate(cases):
            for branch in list(contexts)[i%3:]+list(contexts)[:i%3]:
                payload=dict(model=MODEL,messages=[dict(role='user',content=SCHEMA+'\n'+contexts[branch]+'\n'+case(*c)+' /no_think')],temperature=0,max_tokens=96,stream=False)
                t=time.monotonic(); row=dict(branch=branch,case=c,expected=oracle(*c),request=payload)
                try:
                    request=urllib.request.Request('http://localhost:12434/engines/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=90) as r: raw=json.load(r)
                    action=(raw['choices'][0]['message'].get('content') or '').split('</think>')[-1].strip()
                    row.update(response=raw,action=action,correct=action==oracle(*c))
                except Exception as e: row.update(error=repr(e),correct=False)
                row['seconds']=time.monotonic()-t; rows.append(row); f.write(json.dumps(row)+'\n'); f.flush()
                print(branch,c,row['correct'],round(row['seconds'],2),flush=True)
                if 'error' in row: raise RuntimeError(row['error'])
                if time.monotonic()-start>600: raise TimeoutError('Pilot 10-minute budget')
    summary={b:dict(correct=sum(r['correct'] for r in rows if r['branch']==b),total=8,seconds=sum(r['seconds'] for r in rows if r['branch']==b),prompt_tokens=sum(r['response']['usage']['prompt_tokens'] for r in rows if r['branch']==b),completion_tokens=sum(r['response']['usage']['completion_tokens'] for r in rows if r['branch']==b)) for b in contexts}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name for p in sorted(out.iterdir()) if p.is_file()))
if __name__=='__main__': main()
