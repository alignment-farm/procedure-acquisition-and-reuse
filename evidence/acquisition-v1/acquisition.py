"""Frozen acquisition-v1 implementation. See protocol/acquisition-v1.md."""
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

from procedure_task import SCHEMA, RULE_UP, case, oracle
from pilot2_mlx import sha256


def examples():
    return [(m,u,s) for m,u in [('amber',False),('blue',False),('blue',True)] for s in ['cat','fern','oak','reed']]


def evaluation():
    rng=random.Random(20260911)
    exclude={'cat','fern','oak','reed','planet','silver','harbor','cobalt','meadow','ticket'}
    words=[]
    for length in [4,6,8,10]:
        for _ in range(4):
            while True:
                word=''.join(rng.choice(string.ascii_lowercase) for _ in range(length))
                if word not in exclude: break
            exclude.add(word); words.append(word)
    return [(m,u,w) for w in words for m,u in itertools.product(['amber','blue'],[False,True])]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('models/qwen3-4b-instruct'))
    args=parser.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten,tree_unflatten
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers

    started=time.monotonic(); out=args.output; out.mkdir(parents=True,exist_ok=False)
    for name in ['acquisition.py','procedure_task.py','pilot2_mlx.py']:
        (out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (out/'protocol.md').write_bytes(Path('protocol/acquisition-v1.md').read_bytes())
    log=(out/'events.jsonl').open('x')
    def emit(kind,**data):
        log.write(json.dumps(dict(kind=kind,elapsed=time.monotonic()-started,**data))+'\n');log.flush()
    def check():
        if time.monotonic()-started>1200: raise TimeoutError('20-minute budget')
        if mx.get_peak_memory()>24_000_000_000: raise MemoryError('24 GB MLX budget')
    def digest(items):
        h=hashlib.sha256()
        for name,value in sorted(items):
            h.update(name.encode());h.update(str((value.shape,value.dtype)).encode())
            # bfloat16 has no direct NumPy representation; convert one array at a time.
            h.update(np.asarray(value.astype(mx.float32)).tobytes())
        return h.hexdigest()
    status='failed'; failure=None
    try:
        model_dir=args.model.resolve()
        reference=json.loads(Path('evidence/pilot-03-mlx/model.json').read_text())
        hashes={n:sha256(model_dir/n) for n in reference['files']}
        assert hashes==reference['files'], 'Checkpoint changed since diagnostics'
        emit('provenance',files=hashes,python=platform.python_version(),
             packages={n:md.version(n) for n in ['mlx','mlx-lm','transformers']},
             source=md.distribution('mlx-lm').read_text('direct_url.json'))
        tick=time.monotonic();mx.random.seed(17)
        model,tokenizer=load(str(model_dir));model.freeze()
        linear_to_lora_layers(model,8,dict(rank=8,scale=2.,dropout=0.,keys=['self_attn.q_proj','self_attn.v_proj']))
        mx.eval(model.parameters());emit('load',seconds=time.monotonic()-tick)
        initial=[(k,mx.array(v)) for k,v in tree_flatten(model.trainable_parameters())]
        assert initial and all('lora_' in k for k,v in initial)
        base=lambda:[(k,v) for k,v in tree_flatten(model.parameters()) if 'lora_' not in k]
        base_hash=digest(base()); initial_hash=digest(initial)
        emit('adapter_config',rank=8,scale=2,layers=8,lr=0.0005,steps=96,seed=17,
             initial_hash=initial_hash,base_hash=base_hash,parameters=sum(v.size for k,v in initial),modules=[k for k,v in initial])
        def prompt(content):
            return tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        sampler=make_sampler(temp=0)
        def generate(content,limit=96):
            check();model.eval();rendered=prompt(content);tick=time.monotonic();chunks=[]
            try:
                for chunk in stream_generate(model,tokenizer,rendered,max_tokens=limit,sampler=sampler):
                    chunks.append(chunk);check()
            except BaseException as exc:
                emit('generation_failure',prompt=rendered,partial=''.join(c.text for c in chunks),error=repr(exc));raise
            raw=''.join(c.text for c in chunks)
            return dict(prompt=rendered,raw=raw,action=raw.split('</think>')[-1].strip(),
                prompt_tokens=len(tokenizer.encode(rendered)),completion_tokens=len(chunks),
                token_ids=[int(c.token) for c in chunks],seconds=time.monotonic()-tick,at_limit=len(chunks)>=limit)
        tick=time.monotonic();train=examples()
        evidence='Checked successful interactions:\n'+'\n'.join(case(*c)+' -> '+oracle(*c) for c in train)
        (out/'acquisition.json').write_text(json.dumps([dict(case=c,call=oracle(*c)) for c in train],indent=2))
        emit('construction',seconds=time.monotonic()-tick,source='researcher-defined deterministic oracle',examples=12)
        lesson=generate('Infer a reusable tool protocol from these checked interactions. Write a compact lesson of at most 100 words for future new items. State any assumption needed for an unseen combination. Output the lesson only.\n'+evidence,256)
        emit('lesson',**lesson);(out/'lesson.txt').write_text(lesson['action'])
        # Assistant-only targets. Prefix IDs must match exactly; no boundary approximation.
        tick=time.monotonic();encoded=[]
        for c in train:
            prefix=tokenizer.encode(prompt(SCHEMA+'\n\n'+case(*c)))
            full=tokenizer.encode(prompt(SCHEMA+'\n\n'+case(*c))+oracle(*c))+[tokenizer.eos_token_id]
            assert full[:len(prefix)]==prefix
            encoded.append(dict(case=c,ids=full,answer_start=len(prefix),target=oracle(*c)))
        (out/'training_tokens.json').write_text(json.dumps(encoded))
        emit('target_preparation',seconds=time.monotonic()-tick)
        probe=mx.array(encoded[0]['ids'][:encoded[0]['answer_start']])[None,:]
        def logits():
            model.eval();v=model(probe)[:,-1,:];mx.eval(v);return v
        before=logits()
        def loss_fn(m,ids,start):
            outputs=m(ids[:,:-1])
            return nn.losses.cross_entropy(outputs[:,start-1:,:],ids[:,start:],reduction='mean')
        grad_fn=nn.value_and_grad(model,loss_fn)
        optimizer=optim.AdamW(learning_rate=0.0005,weight_decay=0.)
        train_tick=time.monotonic();rng=random.Random(17);step=0;train_tokens=0;loss_tokens=0
        for epoch in range(8):
            order=list(range(len(encoded)));rng.shuffle(order)
            for index in order:
                check(); row=encoded[index];ids=mx.array(row['ids'])[None,:];model.train()
                tick=time.monotonic();loss,grads=grad_fn(model,ids,row['answer_start'])
                optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state,loss)
                assert np.isfinite(loss.item());step+=1
                train_tokens+=len(row['ids'])-1;loss_tokens+=len(row['ids'])-row['answer_start']
                emit('train_step',step=step,epoch=epoch,index=index,loss=loss.item(),seconds=time.monotonic()-tick)
                if step%12==0: print('training',step,'loss',loss.item(),flush=True)
        trained=[(k,mx.array(v)) for k,v in tree_flatten(model.trainable_parameters())]
        trained_hash=digest(trained);assert trained_hash!=initial_hash
        mx.save_safetensors(str(out/'adapter.safetensors'),dict(trained))
        emit('training_complete',seconds=time.monotonic()-train_tick,input_tokens=train_tokens,
             supervised_tokens=loss_tokens,steps=step,hash=trained_hash,adapter_bytes=(out/'adapter.safetensors').stat().st_size)
        assert digest(base())==base_hash
        model.update(tree_unflatten(initial));mx.eval(model.parameters())
        assert digest(tree_flatten(model.trainable_parameters()))==initial_hash
        assert float(mx.max(mx.abs(logits()-before)).item())==0
        emit('reset_verified',exact_logits=True,base_unchanged=True)
        # Generate untouched identifiers only after training is finished.
        tests=evaluation();(out/'evaluation.json').write_text(json.dumps(tests,indent=2))
        branches=['none','examples','lesson','oracle','adapter']
        contexts=dict(none='',examples=evidence,lesson=lesson['action'],oracle=RULE_UP,adapter='')
        current='initial';rows=[]
        with (out/'responses.jsonl').open('x') as responses:
            for repeat in range(2):
                for i,c in enumerate(tests):
                    order=branches[(i+repeat)%5:]+branches[:(i+repeat)%5]
                    for branch in order:
                        desired='trained' if branch=='adapter' else 'initial';switch=time.monotonic()
                        if desired!=current:
                            model.update(tree_unflatten(trained if desired=='trained' else initial));mx.eval(model.parameters());current=desired
                        switch_s=time.monotonic()-switch
                        result=generate(SCHEMA+'\n'+contexts[branch]+'\n'+case(*c))
                        verify=time.monotonic();correct=result['action']==oracle(*c);verification_s=time.monotonic()-verify
                        record=dict(repeat=repeat,index=i,case=c,branch=branch,expected=oracle(*c),correct=correct,
                                    switch_seconds=switch_s,verification_seconds=verification_s,**result)
                        responses.write(json.dumps(record)+'\n');responses.flush();rows.append(record)
                    if i%16==15: print('evaluation pass',repeat+1,'tasks',i+1,flush=True)
        model.update(tree_unflatten(trained));mx.eval(model.parameters())
        for c in train:
            result=generate(SCHEMA+'\n\n'+case(*c));emit('training_recall',case=c,correct=result['action']==oracle(*c),**result)
        assert digest(base())==base_hash
        model.update(tree_unflatten(initial));mx.eval(model.parameters())
        assert float(mx.max(mx.abs(logits()-before)).item())==0
        emit('final_invariants',base_unchanged=True,reset_logits_exact=True)
        summary={b:dict(correct=sum(r['correct'] for r in rows if r['branch']==b and r['repeat']==0),total=64,
                       repeat_correct=sum(r['correct'] for r in rows if r['branch']==b and r['repeat']==1)) for b in branches}
        (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
        status='complete'
    except BaseException as exc:
        failure=repr(exc);emit('failure',error=failure);raise
    finally:
        emit('complete',status=status,error=failure,seconds=time.monotonic()-started,peak_mlx_bytes=mx.get_peak_memory());log.close()
        (out/'SHA256SUMS').write_text('\n'.join(sha256(p)+'  '+p.name for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))

if __name__=='__main__':main()
