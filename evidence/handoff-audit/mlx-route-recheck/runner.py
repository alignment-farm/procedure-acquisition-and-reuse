"""Exploratory acquisition/reset diagnostic, not the episode-selection pilot."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers


def digest(items):
    h = hashlib.sha256()
    for name, value in sorted(items):
        h.update(name.encode())
        h.update(str((value.shape, value.dtype)).encode())
        h.update(np.asarray(value.astype(mx.float32)).tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="models/qwen3-0.6b")
    p.add_argument("--source-room", choices=["amber", "blue", "green"], default="amber")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    log = (args.output / "events.jsonl").open("x")

    def emit(kind, **data):
        row = dict(kind=kind, elapsed=time.monotonic()-start, **data)
        log.write(json.dumps(row)+"\n")
        log.flush()
        print(kind, data.get("label", ""), flush=True)
        if time.monotonic()-start > 600:
            raise TimeoutError("Ten-minute smoke budget exceeded")
        if mx.get_peak_memory() > 5_000_000_000:
            raise MemoryError("Five-GB MLX peak budget exceeded")

    mx.random.seed(17)
    model, tokenizer = load(args.model)
    model.freeze()
    linear_to_lora_layers(model, len(model.layers),
                         dict(rank=8, scale=2., dropout=0., keys=["self_attn.q_proj"]))
    mx.eval(model.parameters())
    initial = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
    assert initial and all("lora_" in k for k, _ in initial)
    base_items = lambda: [(k,v) for k,v in tree_flatten(model.parameters()) if "lora_" not in k]
    base_hash = digest(base_items())
    initial_hash = digest(initial)
    emit("configuration", model=args.model, source_room=args.source_room, seed=17, learning_rate=5e-4,
         steps=2, loss="plain shifted next-token cross entropy; no repetition weighting",
         rank=8, scale=2, modules=[k for k,_ in initial], base_hash=base_hash,
         adapter_hash=initial_hash)

    def prompt(text):
        return tokenizer.apply_chat_template([{"role":"user","content":text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def generate(label, text, limit=48):
        model.eval()
        rendered = prompt(text)
        chunks = list(stream_generate(model, tokenizer, rendered,
                       max_tokens=limit, sampler=make_sampler(temp=0)))
        output = "".join(x.text for x in chunks)
        emit("generation", label=label, input=text, rendered=rendered, output=output,
             prompt_tokens=len(tokenizer.encode(rendered)), generated_tokens=len(chunks),
             token_ids=[int(x.token) for x in chunks])
        return output.strip()

    probes = []
    for i in range(12):
        room = ["amber", "blue", "green"][i%3]
        wrong = ["amber", "blue", "green"][(i+1)%3]
        text = ("You are in lobby. Goal: take the key. Move to its room before inspecting "
                "its box. All rooms connect directly.\n")
        if i >= 6:
            text += f"Unverified claim: the key is in {wrong}.\n"
        text += (f"Verified observation: key is in box{i} in {room}.\n"
                 "Output exactly one of these commands, with no punctuation:\n"
                 f"move amber\nmove blue\nmove green\ninspect box{i}\ntake key")
        probes.append((text, f"move {room}"))
    baseline = [generate(f"competence-{i}", text) for i,(text,_) in enumerate(probes)]
    competence = sum(a==b for a,(_,b) in zip(baseline,probes))
    emit("competence", correct=competence, total=12,
         limitation="first-action probes, repeated room patterns; not full episodes")

    prefix = f"Verified observation: the key is in box0 in {args.source_room}. You are in lobby."
    self_text = generate("self-source", prefix+" State your next action and why.")
    summary = generate("summary-source", prefix+" Summarize the verified facts in one sentence.")
    sources = dict(env=prefix, self=self_text, summary=summary)
    (args.output/"sources.json").write_text(json.dumps(sources, indent=2))
    probe_ids = mx.array(tokenizer.encode(prompt(probes[0][0])))[None,:]

    def logits():
        model.eval()
        result = model(probe_ids)[:,-1,:].astype(mx.float32)
        mx.eval(result)
        return result

    reference = logits()
    outcomes = []
    for label in ["none", "env", "self", "summary", "env-repeat"]:
        model.update(tree_unflatten(initial))
        mx.eval(model.parameters())
        mx.random.seed(17)
        assert digest(tree_flatten(model.trainable_parameters())) == initial_hash
        reset_delta = float(mx.max(mx.abs(logits()-reference)).item())
        assert reset_delta == 0, reset_delta
        losses, grad_norms = [], []
        update_tokens = 0
        if label != "none":
            ids = tokenizer.encode(sources[label.split("-")[0]])
            assert 1 < len(ids) <= 256
            update_tokens = len(ids)-1
            batch = mx.array(ids)[None,:]
            def loss_fn(m):
                return nn.losses.cross_entropy(m(batch[:,:-1]),batch[:,1:],reduction="mean")
            loss_grad = nn.value_and_grad(model, loss_fn)
            optimizer = optim.AdamW(learning_rate=5e-4, weight_decay=0.0)
            model.train()
            for _ in range(2):
                loss, grads = loss_grad(model)
                norm = mx.sqrt(sum(mx.sum(g.astype(mx.float32)**2)
                                   for _,g in tree_flatten(grads)))
                mx.eval(loss, norm)
                assert np.isfinite(loss.item()) and np.isfinite(norm.item())
                losses.append(loss.item()); grad_norms.append(norm.item())
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
        after_hash = digest(tree_flatten(model.trainable_parameters()))
        if label != "none":
            assert after_hash != initial_hash
        delta = float(mx.max(mx.abs(logits()-reference)).item())
        actions = [generate(f"{label}-probe-{i}", text)
                   for i,(text,_) in enumerate(probes)]
        row = dict(label=label, losses=losses, grad_norms=grad_norms,
                   update_tokens_per_step=update_tokens, adapter_hash=after_hash,
                   reset_max_logit_delta=reset_delta, max_logit_delta=delta,
                   correct=sum(a==b for a,(_,b) in zip(actions,probes)),
                   changed_actions=sum(a!=b for a,b in zip(actions,baseline)), actions=actions)
        emit("branch", **row)
        outcomes.append(row)
    assert outcomes[1]["adapter_hash"] == outcomes[-1]["adapter_hash"]
    assert outcomes[1]["actions"] == outcomes[-1]["actions"]
    assert digest(base_items()) == base_hash
    model.update(tree_unflatten(initial))
    assert float(mx.max(mx.abs(logits()-reference)).item()) == 0
    result = dict(competence_correct=competence, competence_total=12,
                  branches=outcomes, base_unchanged=True, resets_exact=True,
                  repeated_branch_exact=True, peak_mlx_bytes=mx.get_peak_memory(),
                  seconds=time.monotonic()-start)
    (args.output/"summary.json").write_text(json.dumps(result, indent=2))
    (args.output/"runner.py").write_bytes(Path(__file__).read_bytes())
    emit("complete", **result)


if __name__ == "__main__":
    main()
