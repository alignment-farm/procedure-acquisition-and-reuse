"""Audit component diagnostic; semantic diagnostics never replace exact scoring."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


def fields(action, expected):
    target_tool, target_argument = expected.split(' ')
    target_item, target_suffix = target_argument.rsplit('-', 1)
    # Anchored grammar; permit whitespace around the suffix solely to diagnose
    # format failures. Missing/extra text is unparsed and fails every field.
    match = re.fullmatch(r'(dax|wug)\s+([A-Za-z]+)\s*(-[29])', action)
    if not match:
        return dict(parseable=False, tool=False, item=False, suffix=False, strict_format=False)
    tool, item, suffix = match.groups()
    return dict(parseable=True, tool=tool==target_tool, item=item==target_item,
                suffix=suffix=='-'+target_suffix,
                strict_format=action==f'{tool} {item}{suffix}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    for line in (args.run/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256((args.run/name).read_bytes()).hexdigest()==digest, name
    design=json.loads((args.run/'design.json').read_text())
    rows=[json.loads(l) for l in (args.run/'responses.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows]==[r['id'] for r in design['requests']]
    assert len({r['id'] for r in rows})==len(rows)
    result={}
    for branch in dict.fromkeys(r['branch'] for r in rows):
        group=[r for r in rows if r['branch']==branch]
        for r in group:
            assert r['correct']==(r['action']==r['expected'])
        diagnostic=[fields(r['action'],r['expected']) for r in group] if branch.startswith('full_') else []
        result[branch]=dict(total=len(group),correct=sum(r['correct'] for r in group),
            prompt_tokens=sum(r['prompt_tokens'] for r in group),completion_tokens=sum(r['completion_tokens'] for r in group),
            field_correct={k:sum(d[k] for d in diagnostic) for k in diagnostic[0]} if diagnostic else None,
            failures=[dict(case=r['case'],actual=r['action'],expected=r['expected']) for r in group if not r['correct']])
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
