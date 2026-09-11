"""Audit the frozen follow-up and separate lexical/combination transfer."""
import argparse
import json
from pathlib import Path
from pilot2_mlx import sha256
from procedure_task import oracle
from followup import followup_cases, select_candidate
from analyze_components import fields


def analyze(path):
    for line in (path/'SHA256SUMS').read_text().splitlines():
        h,n=line.split('  ',1);assert sha256(path/n)==h,n
    events=[json.loads(l) for l in (path/'events.jsonl').read_text().splitlines()]
    rows=[json.loads(l) for l in (path/'responses.jsonl').read_text().splitlines()]
    cases=json.loads((path/'evaluation.json').read_text())
    assert cases==json.loads(json.dumps(followup_cases()))
    assert len(rows)==320 and len({(r['index'],r['branch']) for r in rows})==320
    for r in rows:
        assert r['case']==cases[r['index']]['case'] and r['stratum']==cases[r['index']]['stratum']
        assert r['expected']==oracle(*r['case'])
        assert r['correct']==(r['action']==r['expected'])
    candidates=json.loads((path/'candidates.json').read_text())
    selection=next(e for e in events if e['kind']=='selection')
    assert selection['candidate']==select_candidate(candidates)
    assert selection['validated']==(selection['correct']==12)
    acquisition=json.loads((path/'acquisition.json').read_text())
    for candidate in candidates:
        rr=[e for e in events if e['kind']=='validation' and e['candidate']==candidate['candidate']]
        assert len(rr)==12
        assert [e['case'] for e in rr]==[e['case'] for e in acquisition]
        assert all(e['correct']==(e['action']==oracle(*e['case'])) for e in rr)
        assert sum(e['correct'] for e in rr)==candidate['correct']
        assert all(e['elapsed']<selection['elapsed'] for e in rr)
    assert len([e for e in events if e['kind']=='revision'])<=2
    assert all(e['identical'] for e in events if e['kind']=='replay')
    assert len([e for e in events if e['kind']=='replay'])==4
    prior_events=[json.loads(l) for l in Path('evidence/acquisition-v1/events.jsonl').read_text().splitlines()]
    original_lesson=next(e for e in prior_events if e['kind']=='lesson')
    original_training=next(e for e in prior_events if e['kind']=='training_complete')
    original_preparation=next(e for e in prior_events if e['kind']=='target_preparation')
    build=[e for e in events if e['kind'] in ['validation','revision']]
    costs=dict(original_lesson_seconds=original_lesson['seconds'],
        followup_construction_seconds=sum(e['seconds']+e.get('verification_seconds',0) for e in build),
        total_lesson_construction_seconds=original_lesson['seconds']+sum(e['seconds']+e.get('verification_seconds',0) for e in build),
        followup_construction_prompt_tokens=sum(e['prompt_tokens'] for e in build),
        followup_construction_completion_tokens=sum(e['completion_tokens'] for e in build),
        original_lesson_prompt_tokens=original_lesson['prompt_tokens'],original_lesson_completion_tokens=original_lesson['completion_tokens'],
        validation_calls=sum(e['kind']=='validation' for e in build),revision_calls=sum(e['kind']=='revision' for e in build),
        adapter_original_training_seconds=original_training['seconds'],adapter_original_preparation_seconds=original_preparation['seconds'],
        adapter_additional_training_seconds=0.)
    branches={}
    for b in ['none','examples','lesson','oracle','adapter']:
        rr=[r for r in rows if r['branch']==b]
        strata={}
        for st in ['ordinary','random']:
            ss=[r for r in rr if r['stratum']==st]
            seen=[r for r in ss if not(r['case'][0]=='amber' and r['case'][1])]
            held=[r for r in ss if r['case'][0]=='amber' and r['case'][1]]
            diagnostic=[fields(r['action'],r['expected']) for r in ss]
            strata[st]=dict(correct=sum(r['correct'] for r in ss),total=32,
                seen_correct=sum(r['correct'] for r in seen),seen_total=24,
                withheld_correct=sum(r['correct'] for r in held),withheld_total=8,
                by_length={str(n):sum(r['correct'] for r in ss if len(r['case'][2])==n) for n in [4,6,8,10]},
                field_correct={k:sum(d[k] for d in diagnostic) for k in diagnostic[0]})
        branches[b]=dict(correct=sum(r['correct'] for r in rr),total=64,strata=strata,
            prompt_tokens=sum(r['prompt_tokens'] for r in rr),completion_tokens=sum(r['completion_tokens'] for r in rr),
            inference_seconds=sum(r['seconds'] for r in rr),verification_seconds=sum(r['verification_seconds'] for r in rr),
            switch_seconds=sum(r['switch_seconds'] for r in rr))
    return dict(branches=branches,candidates=candidates,selection=selection,costs=costs,
        total=next(e for e in events if e['kind']=='complete'),invariants=next(e for e in events if e['kind']=='final_invariants'),
        total_model_calls=len(rows)+len(build)+4,
        caveats=['The selected lesson is validated only if acquisition score is 12/12.',
                 'Sixteen identifier clusters; no significance or independence claim.',
                 'Lexical strata matched by character length only, not tokenization or frequency.',
                 'No additional adapter training; original acquisition costs remain applicable.',
                 'Unparseable full-call diagnostics conservatively fail all fields.'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    print(json.dumps(analyze(args.run),indent=2))
