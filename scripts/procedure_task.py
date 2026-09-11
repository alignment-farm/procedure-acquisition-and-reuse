"""Pure task definitions for exploratory interface diagnostics."""
import itertools

SCHEMA = ('Route an item using an unfamiliar two-tool protocol. '
          'Your reply must be exactly one line in the form NAME ARGUMENT, where NAME is the tool '
          'and ARGUMENT is the transformed item string. Write no other text. ')
RULE_REV = ('For amber mode, use tool dax and reverse the item string so it reads backwards. '
            'For blue mode, use tool wug and keep the item unchanged. '
            'For urgent priority, append -9 to the transformed item; for normal priority, append -2.')
RULE_UP = ('For amber mode, use tool dax and write the item string in UPPERCASE. '
           'For blue mode, use tool wug and keep the item unchanged. '
           'For urgent priority, append -9 to the transformed item; for normal priority, append -2.')
RULE_STEPS = ('Compute the call in this order:\n'
              '1. Read the value of item. If mode=amber, uppercase that value and choose dax. '
              'If mode=blue, leave that value unchanged and choose wug.\n'
              '2. Append -9 if priority=urgent, otherwise append -2.\n'
              '3. Output the chosen tool, one space, and the resulting argument.')


def oracle(mode, urgent, item, transformation='upper'):
    if mode not in {'amber', 'blue'} or type(urgent) is not bool:
        raise ValueError('Invalid condition')
    if transformation not in {'upper', 'reversal'}:
        raise ValueError('Unknown transformation')
    argument = item
    if mode == 'amber':
        argument = item.upper() if transformation == 'upper' else item[::-1]
    return f'{"dax" if mode == "amber" else "wug"} {argument}-{"9" if urgent else "2"}'


def case(mode, urgent, item):
    return f'mode={mode}, priority={"urgent" if urgent else "normal"}, item={item}'


def design(suite):
    training = [(m,u,s) for m,u in [('amber',False),('blue',False),('blue',True)] for s in ['cat','fern']]
    contexts = {'none':'', 'lesson_reversal':RULE_REV, 'lesson_upper':RULE_UP}
    transformations = {b:'reversal' if 'reversal' in b else 'upper' for b in contexts}
    for name in ['reversal','upper']:
        contexts['examples_'+name] = 'Checked successful interactions:\n'+'\n'.join(case(*c)+' -> '+oracle(*c,name) for c in training)
        transformations['examples_'+name] = name
    words = ['planet','silver']
    if suite == 'execution':
        # Fixed development diagnostic: original wording vs ordered steps vs
        # operation-only control. Includes prior failures and four fresh words.
        words = ['planet','silver','harbor','cobalt','meadow','ticket']
        contexts = {'lesson_upper':RULE_UP,'lesson_steps':RULE_STEPS}
        transformations = {b:'upper' for b in contexts}
    requests=[]
    for i,c in enumerate(itertools.product(['amber','blue'],[False,True],words)):
        branches=list(contexts); branches=branches[i%len(branches):]+branches[:i%len(branches)]
        for branch in branches:
            requests.append(dict(id=f'{branch}:{case(*c)}', branch=branch, case=c,
                content=SCHEMA+'\n'+contexts[branch]+'\n'+case(*c),
                expected=oracle(*c,transformations[branch])))
    if suite == 'execution':
        for word in words:
            requests.append(dict(id='uppercase_only:'+word,branch='uppercase_only',case=[word],
                content=f'Convert the string {word} to uppercase. Output only the converted string.',
                expected=word.upper()))
    return dict(suite=suite,exploratory=True,training=training,contexts=contexts,
        requests=requests,gate='Exploratory screening only; no prospective holdout or acquisition claim',
        caveat='Oracle rules are researcher supplied; prior words reused; execution suite adds four development words.')
