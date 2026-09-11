"""Recompute behavior and empirical cost ledgers from acquisition-v1 evidence."""
import argparse
import json
from pathlib import Path
from pilot2_mlx import sha256
from procedure_task import oracle
from analyze_components import fields


def analyze(path):
    for line in (path/'SHA256SUMS').read_text().splitlines():
        h,n=line.split('  ',1);assert sha256(path/n)==h,n
    rows=[json.loads(l) for l in (path/'responses.jsonl').read_text().splitlines()]
    events=[json.loads(l) for l in (path/'events.jsonl').read_text().splitlines()]
    assert len(rows)==640
    assert len({(r['repeat'],r['index'],r['branch']) for r in rows})==640
    assert all(r['expected']==oracle(*r['case']) for r in rows)
    assert all(r['correct']==(r['action']==r['expected']) for r in rows)
    cases=json.loads((path/'evaluation.json').read_text())
    assert all(r['case']==cases[r['index']] for r in rows)
    event=lambda name: next(e for e in events if e['kind']==name)
    train=event('training_complete'); lesson=event('lesson')
    costs={'none':0.,'oracle':0.,'examples':0.,'lesson':lesson['seconds'],
           'adapter':train['seconds']+event('target_preparation')['seconds']}
    result={}
    for branch in ['none','examples','lesson','oracle','adapter']:
        rr=[r for r in rows if r['branch']==branch];first=[r for r in rr if r['repeat']==0];second=[r for r in rr if r['repeat']==1]
        assert len(first)==len(second)==64
        first.sort(key=lambda r:r['index']);second.sort(key=lambda r:r['index'])
        diagnostics=[fields(r['action'],r['expected']) for r in first]
        total=lambda key:sum(r[key] for r in rr)
        result[branch]=dict(first_correct=sum(r['correct'] for r in first),total=64,
            withheld_correct=sum(r['correct'] for r in first if r['case'][0]=='amber' and r['case'][1]),withheld_total=16,
            seen_correct=sum(r['correct'] for r in first if not(r['case'][0]=='amber' and r['case'][1])),seen_total=48,
            length_correct={str(n):sum(r['correct'] for r in first if len(r['case'][2])==n) for n in [4,6,8,10]},
            second_correct=sum(r['correct'] for r in second),repeat_identical=sum(a['token_ids']==b['token_ids'] for a,b in zip(first,second)),
            field_correct={k:sum(d[k] for d in diagnostics) for k in diagnostics[0]},
            uses=128,prompt_tokens=total('prompt_tokens'),completion_tokens=total('completion_tokens'),
            inference_seconds=total('seconds'),switch_seconds=total('switch_seconds'),verification_seconds=total('verification_seconds'),
            incremental_acquisition_seconds=costs[branch],
            cumulative_seconds=[dict(uses=n,seconds=costs[branch]+sum(r['seconds']+r['verification_seconds'] for r in rr[:n]),successes=sum(r['correct'] for r in rr[:n])) for n in [1,16,32,64,128]])
    # A projection is eligible only at at least the comparator's observed accuracy;
    # still not a significance/equivalence or future-workload guarantee.
    projection={}
    adapter=result['adapter']
    for branch in ['examples','lesson']:
        other=result[branch]
        saving=(other['inference_seconds']-adapter['inference_seconds'])/128
        delta=costs['adapter']-costs[branch]
        projection[branch]=dict(eligible=adapter['first_correct']>=other['first_correct'],
            seconds_saved_per_call=saving,
            linear_crossing_calls=max(0,delta/saving) if saving>0 else None,
            caveat='Arithmetic projection only; not observed repayment. Ineligible when adapter has lower useful performance; excludes researcher labor, switching, energy and deployment startup.')
    return dict(branches=result,training=train,lesson_cost={k:lesson[k] for k in ['seconds','prompt_tokens','completion_tokens']},
        training_recall=sum(e['correct'] for e in events if e['kind']=='training_recall'),training_recall_total=12,
        final_invariants=event('final_invariants'),total=event('complete'),projection=projection,
        caveat='Single seed and 16 identifier clusters; two repetitions are not independent samples. No automatic retries, persistent KV cache or dollar/energy conversion.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    print(json.dumps(analyze(args.run),indent=2))
