"""Exploratory MLX interface diagnostic; not a matched Docker comparison.

Uses the locally downloaded checkpoint, with file hashes and download metadata.
Reuses pilot-01 development inputs. No parameter learning or prospective test.
"""
import argparse, hashlib, json, platform, time, importlib.metadata as md
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models' / 'qwen3-4b-instruct'
LIMIT = 96
TEMPERATURE = 0.0

from procedure_task import design

def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=MODEL_DIR)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=['pilot', 'execution'], default='pilot')
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
    (out/'procedure_task.py').write_bytes(Path(__file__).with_name('procedure_task.py').read_bytes())
    metadata = {str(p.relative_to(model_dir)):p.read_text() for p in model_dir.glob('.cache/huggingface/download/**/*.metadata')}
    (out/'model.json').write_text(json.dumps(dict(path=str(model_dir),
        files={n:sha256(model_dir/n) for n in sorted(required)}, download_metadata=metadata,
        config=json.loads((model_dir/'config.json').read_text()),
        caveat='Not established to be the same checkpoint as the Docker GGUF; no cross-runtime causal comparison.'),indent=2))
    (out/'runtime.json').write_text(json.dumps(dict(python=platform.python_version(),
        packages={n:md.version(n) for n in ['mlx','mlx-lm','transformers']},
        mlx_lm_source=md.distribution('mlx-lm').read_text('direct_url.json'),
        seed=17, temperature=TEMPERATURE),indent=2))
    specification = design(args.suite)
    specification.update(max_tokens=LIMIT, budget_seconds=600, peak_memory_limit_bytes=20_000_000_000)
    (out/'design.json').write_text(json.dumps(specification,indent=2))
    requests = specification['requests']
    branches = list(dict.fromkeys(r['branch'] for r in requests))
    rows=[]; started=time.monotonic(); load_seconds=None
    status='running'; error=None
    try:
        mx.random.seed(17)
        model,tokenizer=load(str(model_dir))
        load_seconds=time.monotonic()-started
        sampler=make_sampler(temp=TEMPERATURE)
        with (out/'responses.jsonl').open('x') as log:
            for request in requests:
                branch = request['branch']; c = request['case']
                if time.monotonic()-started>600:
                    raise TimeoutError('Ten-minute run budget exceeded')
                rendered=tokenizer.apply_chat_template([{'role':'user','content':request['content']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                expected=request['expected']
                tick=time.monotonic(); chunks=[]
                row=dict(id=request['id'],branch=branch,case=c,expected=expected,prompt=rendered)
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
            branches={b:dict(planned=sum(r['branch']==b for r in requests),completed=sum(r['branch']==b and 'correct' in r for r in rows),
                correct=sum(r['branch']==b and r.get('correct',False) for r in rows),
                errors=sum(r['branch']==b and 'error' in r for r in rows)) for b in branches})
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        (out/'SHA256SUMS').write_text('\n'.join(sha256(p)+'  '+p.name for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
        print(json.dumps(summary),flush=True)

if __name__ == '__main__':
    main()
