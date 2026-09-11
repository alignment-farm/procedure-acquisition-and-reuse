"""Run the frozen followup-v1 protocol without changing the saved adapter."""
import argparse
import hashlib
import importlib.metadata as md
import itertools
import json
import platform
import random
import string
import time
from pathlib import Path

from acquisition import examples, evaluation
from procedure_task import SCHEMA, RULE_UP, case, oracle
from pilot2_mlx import sha256

WORDS = ['lamp','boat','copper','garden','notebook','mountain','helicopter','strawberry']


def followup_cases():
    previous={c[2] for c in examples()+evaluation()} | {'planet','silver','harbor','cobalt','meadow','ticket'}
    assert not previous & set(WORDS)
    exclude=previous|set(WORDS);rng=random.Random(20260912);identifiers=[]
    for length in [4,6,8,10]:
        for _ in range(2):
            while True:
                word=''.join(rng.choice(string.ascii_lowercase) for _ in range(length))
                if word not in exclude: break
            identifiers.append(word);exclude.add(word)
    # Interleave lexical strata within length to reduce timing-order confounding.
    result=[]
    for word,identifier in zip(WORDS,identifiers):
        for stratum,item in [('ordinary',word),('random',identifier)]:
            for mode,urgent in itertools.product(['amber','blue'],[False,True]):
                result.append(dict(stratum=stratum,case=(mode,urgent,item)))
    return result


def select_candidate(candidates):
    return max(range(len(candidates)),key=lambda i:(candidates[i]['correct'],-i))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('models/qwen3-4b-instruct'))
    args=parser.parse_args()
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten,tree_unflatten
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers

    started=time.monotonic();out=args.output;out.mkdir(parents=True,exist_ok=False)
    for name in ['followup.py','acquisition.py','procedure_task.py','pilot2_mlx.py']:
        (out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (out/'protocol.md').write_bytes(Path('protocol/followup-v1.md').read_bytes())
    events=(out/'events.jsonl').open('x')
    def emit(kind,**data):
        events.write(json.dumps(dict(kind=kind,elapsed=time.monotonic()-started,**data))+'\n');events.flush()
    def check():
        if time.monotonic()-started>1200: raise TimeoutError('20-minute budget')
        if mx.get_peak_memory()>24_000_000_000: raise MemoryError('24 GB MLX budget')
    def digest(items):
        h=hashlib.sha256()
        for name,value in sorted(items):
            h.update(name.encode());h.update(str((value.shape,value.dtype)).encode())
            h.update(np.asarray(value.astype(mx.float32)).tobytes())
        return h.hexdigest()
    status='failed';failure=None
    try:
        prior=Path('evidence/acquisition-v1')
        for line in (prior/'SHA256SUMS').read_text().splitlines():
            h,n=line.split('  ',1);assert sha256(prior/n)==h,n
        old_events=[json.loads(l) for l in (prior/'events.jsonl').read_text().splitlines()]
        original_config=next(r for r in old_events if r['kind']=='adapter_config')
        original_train=next(r for r in old_events if r['kind']=='training_complete')
        reference=json.loads(Path('evidence/pilot-03-mlx/model.json').read_text())
        hashes={n:sha256(args.model/n) for n in reference['files']}
        assert hashes==reference['files']
        adapter_file_hash=sha256(prior/'adapter.safetensors')
        emit('provenance',model_files=hashes,adapter_sha256=adapter_file_hash,
             packages={n:md.version(n) for n in ['mlx','mlx-lm','transformers']},python=platform.python_version(),
             source=md.distribution('mlx-lm').read_text('direct_url.json'))
        tick=time.monotonic();mx.random.seed(17)
        model,tokenizer=load(str(args.model));model.freeze()
        linear_to_lora_layers(model,8,dict(rank=8,scale=2.,dropout=0.,keys=['self_attn.q_proj','self_attn.v_proj']))
        mx.eval(model.parameters());emit('load',seconds=time.monotonic()-tick)
        initial=[(k,mx.array(v)) for k,v in tree_flatten(model.trainable_parameters())]
        trained=list(mx.load(str(prior/'adapter.safetensors')).items())
        assert digest(initial)==original_config['initial_hash']
        assert digest(trained)==original_train['hash']
        base=lambda:[(k,v) for k,v in tree_flatten(model.parameters()) if 'lora_' not in k]
        base_hash=digest(base());assert base_hash==original_config['base_hash']
        emit('hashes_verified',base_hash=base_hash,initial_hash=digest(initial),trained_hash=digest(trained))
        def prompt(content):
            return tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        probe=mx.array(tokenizer.encode(prompt(SCHEMA+'\n\n'+case(*examples()[0]))))[None,:]
        def logits():
            model.eval();value=model(probe)[:,-1,:];mx.eval(value);return value
        before=logits();sampler=make_sampler(temp=0)
        def generate(content,limit=96):
            check();model.eval();rendered=prompt(content);tick=time.monotonic();chunks=[]
            try:
                for chunk in stream_generate(model,tokenizer,rendered,max_tokens=limit,sampler=sampler):
                    chunks.append(chunk);check()
            except BaseException as exc:
                emit('generation_failure',prompt=rendered,partial=''.join(c.text for c in chunks),error=repr(exc));raise
            raw=''.join(c.text for c in chunks)
            return dict(prompt=rendered,raw=raw,action=raw.split('</think>')[-1].strip(),
                        prompt_tokens=len(tokenizer.encode(rendered)),completion_tokens=len(chunks),token_ids=[int(c.token) for c in chunks],
                        seconds=time.monotonic()-tick,at_limit=len(chunks)>=limit)
        # Saved-checkpoint replay is separate from new-input evaluation.
        model.update(tree_unflatten(trained));mx.eval(model.parameters())
        old_rows=[json.loads(l) for l in (prior/'responses.jsonl').read_text().splitlines()]
        for index in [0,16,32,48]:
            old=next(r for r in old_rows if r['repeat']==0 and r['branch']=='adapter' and r['index']==index)
            result=generate(SCHEMA+'\n\n'+case(*old['case']))
            assert result['token_ids']==old['token_ids']
            emit('replay',index=index,identical=True,**result)
        model.update(tree_unflatten(initial));mx.eval(model.parameters())
        assert float(mx.max(mx.abs(logits()-before)).item())==0
        acquisition=[dict(case=c,expected=oracle(*c)) for c in examples()]
        evidence='Checked successful interactions:\n'+'\n'.join(case(*r['case'])+' -> '+r['expected'] for r in acquisition)
        (out/'acquisition.json').write_text(json.dumps(acquisition,indent=2))
        candidate=(prior/'lesson.txt').read_text();candidates=[]
        for number in range(3):
            validation=[]
            for row in acquisition:
                result=generate(SCHEMA+'\n'+candidate+'\n'+case(*row['case']))
                tick=time.monotonic();correct=result['action']==row['expected'];verification_s=time.monotonic()-tick
                record=dict(candidate=number,case=row['case'],expected=row['expected'],correct=correct,verification_seconds=verification_s,**result)
                emit('validation',**record);validation.append(record)
            count=sum(r['correct'] for r in validation)
            candidates.append(dict(candidate=number,text=candidate,correct=count,total=12))
            (out/'candidates.json').write_text(json.dumps(candidates,indent=2))
            print('lesson candidate',number,count,'/12',flush=True)
            if count==12 or number==2: break
            feedback='\n'.join(case(*r['case'])+': produced '+repr(r['action'])+'; expected '+repr(r['expected']) for r in validation)
            revision=generate('Revise the compact lesson using only the checked interactions and validation feedback below. '
                'The lesson must let a model reproduce the checked calls for future items. '
                'Use at most 100 words. State any assumption for unseen combinations. Output only the revised lesson.\n'
                +evidence+'\nPrevious lesson:\n'+candidate+'\nValidation on the same acquisition inputs:\n'+feedback,256)
            emit('revision',candidate=number+1,**revision);candidate=revision['action']
        selected=select_candidate(candidates);chosen=candidates[selected]
        emit('selection',**chosen,validated=chosen['correct']==12)
        (out/'selected_lesson.txt').write_text(chosen['text'])
        # No evaluation cases or outputs enter construction/selection above.
        tests=followup_cases();(out/'evaluation.json').write_text(json.dumps(tests,indent=2))
        branches=['none','examples','lesson','oracle','adapter']
        contexts=dict(none='',examples=evidence,lesson=chosen['text'],oracle=RULE_UP,adapter='')
        current='initial';rows=[]
        with (out/'responses.jsonl').open('x') as log:
            for index,test in enumerate(tests):
                c=test['case'];order=branches[index%5:]+branches[:index%5]
                for branch in order:
                    desired='trained' if branch=='adapter' else 'initial';tick=time.monotonic()
                    if desired!=current:
                        model.update(tree_unflatten(trained if desired=='trained' else initial));mx.eval(model.parameters());current=desired
                    switch=time.monotonic()-tick
                    result=generate(SCHEMA+'\n'+contexts[branch]+'\n'+case(*c))
                    tick=time.monotonic();correct=result['action']==oracle(*c);verify=time.monotonic()-tick
                    row=dict(index=index,stratum=test['stratum'],case=c,branch=branch,expected=oracle(*c),correct=correct,
                             switch_seconds=switch,verification_seconds=verify,**result)
                    rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if index%16==15:print('evaluation',index+1,'/64',flush=True)
        model.update(tree_unflatten(trained));mx.eval(model.parameters())
        assert digest(tree_flatten(model.trainable_parameters()))==original_train['hash']
        assert sha256(prior/'adapter.safetensors')==adapter_file_hash
        assert digest(base())==base_hash
        model.update(tree_unflatten(initial));mx.eval(model.parameters())
        assert float(mx.max(mx.abs(logits()-before)).item())==0
        emit('final_invariants',adapter_unchanged=True,base_unchanged=True,restored_logits_exact=True)
        summary={b:dict(correct=sum(r['correct'] for r in rows if r['branch']==b),total=64,
                       ordinary_correct=sum(r['correct'] for r in rows if r['branch']==b and r['stratum']=='ordinary'),
                       random_correct=sum(r['correct'] for r in rows if r['branch']==b and r['stratum']=='random')) for b in branches}
        (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
        status='complete'
    except BaseException as exc:
        failure=repr(exc);emit('failure',error=failure);raise
    finally:
        emit('complete',status=status,error=failure,seconds=time.monotonic()-started,peak_mlx_bytes=mx.get_peak_memory());events.close()
        (out/'SHA256SUMS').write_text('\n'.join(sha256(p)+'  '+p.name for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))

if __name__=='__main__':main()
