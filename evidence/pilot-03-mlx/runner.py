"""Exploratory MLX interface diagnostic; not a matched Docker comparison.

Uses the locally downloaded checkpoint, with file hashes and download metadata.
Reuses pilot-01 development inputs. No parameter learning or prospective test.
"""
import argparse, hashlib, itertools, json, platform, time, importlib.metadata as md
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

def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=MODEL_DIR)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model_dir = args.model.resolve()
    index = json.loads((model_dir/'model.safetensors.index.json').read_text())
    required = set(index['weight_map'].values()) | {'config.json', 'tokenizer.json', 'tokenizer_config.json'}
    for name in required:
        if not (model_dir/name).is_file():
            raise FileNotFoundError(model_dir/name)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    metadata = {str(p.relative_to(model_dir)):p.read_text() for p in model_dir.glob('.cache/huggingface/download/**/*.metadata')}
    (out/'model.json').write_text(json.dumps(dict(path=str(model_dir),
        files={n:sha256(model_dir/n) for n in sorted(required)}, download_metadata=metadata,
        config=json.loads((model_dir/'config.json').read_text()),
        caveat='Not established to be the same checkpoint as the Docker GGUF; no cross-runtime causal comparison.'),indent=2))
    (out/'runtime.json').write_text(json.dumps(dict(python=platform.python_version(),
        packages={n:md.version(n) for n in ['mlx','mlx-lm','transformers']},
        mlx_lm_source=md.distribution('mlx-lm').read_text('direct_url.json'),
        seed=17, temperature=TEMPERATURE),indent=2))
    training = [(m,u,s) for m,u in [('amber',False),('blue',False),('blue',True)] for s in ['cat','fern']]
    contexts = {'none':'', 'lesson_reversal':RULE_REV, 'lesson_upper':RULE_UP}
    for name,oracle in [('reversal',oracle_rev),('upper',oracle_up)]:
        contexts['examples_'+name] = 'Checked successful interactions:\n'+'\n'.join(case(*c)+' -> '+oracle(*c) for c in training)
    cases = list(itertools.product(['amber','blue'],[False,True],['planet','silver']))
    (out/'design.json').write_text(json.dumps(dict(exploratory=True, cases=cases,
        training=training, contexts=contexts, schema=SCHEMA, max_tokens=LIMIT,
        gate='at least 7/8 correct in a supplied-rule branch; development only',
        budget_seconds=600, peak_memory_limit_bytes=20_000_000_000,
        caveat='Cases reused from prior pilots. Oracle lessons are researcher supplied.'),indent=2))
    rows=[]; started=time.monotonic(); load_seconds=None
    status='running'; error=None
    try:
        mx.random.seed(17)
        model,tokenizer=load(str(model_dir))
        load_seconds=time.monotonic()-started
        sampler=make_sampler(temp=TEMPERATURE)
        with (out/'responses.jsonl').open('x') as log:
            for i,c in enumerate(cases):
                branches=list(contexts); branches=branches[i%5:]+branches[:i%5]
                for branch in branches:
                    if time.monotonic()-started>600:
                        raise TimeoutError('Ten-minute run budget exceeded')
                    rendered=tokenizer.apply_chat_template([{'role':'user','content':SCHEMA+'\n'+contexts[branch]+'\n'+case(*c)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    expected=(oracle_rev if 'reversal' in branch else oracle_up)(*c)
                    tick=time.monotonic(); chunks=[]
                    row=dict(branch=branch,case=c,expected=expected,prompt=rendered)
                    try:
                        for chunk in stream_generate(model,tokenizer,rendered,max_tokens=LIMIT,sampler=sampler):
                            chunks.append(chunk)
                            if time.monotonic()-started>600: raise TimeoutError('Generation budget exceeded')
                            if mx.get_peak_memory()>20_000_000_000: raise MemoryError('20 GB MLX allocation budget exceeded')
                        raw=''.join(c.text for c in chunks)
                        action=raw.split('</think>')[-1].strip()
                        row.update(raw=raw,action=action,correct=action==expected,
                            prompt_tokens=len(tokenizer.encode(rendered)),completion_tokens=len(chunks),
                            token_ids=[int(c.token) for c in chunks],at_token_limit=len(chunks)>=LIMIT)
                    except Exception as exc:
                        row.update(error=repr(exc),partial_output=''.join(c.text for c in chunks))
                        raise
                    finally:
                        row['seconds']=time.monotonic()-tick
                        rows.append(row); log.write(json.dumps(row)+'\n'); log.flush()
                        print(branch,c,row.get('correct',row.get('error')),flush=True)
        status='complete'
    except BaseException as exc:
        status='failed'; error=repr(exc)
        raise
    finally:
        summary=dict(status=status,error=error,load_seconds=load_seconds,
            seconds=time.monotonic()-started,peak_mlx_bytes=mx.get_peak_memory(),
            branches={b:dict(planned=8,completed=sum(r['branch']==b and 'correct' in r for r in rows),
                correct=sum(r['branch']==b and r.get('correct',False) for r in rows),
                errors=sum(r['branch']==b and 'error' in r for r in rows)) for b in contexts})
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        (out/'SHA256SUMS').write_text('\n'.join(sha256(p)+'  '+p.name for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
        print(json.dumps(summary),flush=True)

if __name__ == '__main__':
    main()
