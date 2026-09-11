import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from followup import followup_cases, select_candidate
from acquisition import examples, evaluation

class FollowupTests(unittest.TestCase):
    def test_fresh_stratified_inputs(self):
        cases=followup_cases()
        self.assertEqual(len(cases),64)
        self.assertEqual(len({tuple(r['case']) for r in cases}),64)
        old={c[2] for c in examples()+evaluation()}|{'planet','silver','harbor','cobalt','meadow','ticket'}
        self.assertFalse(old & {r['case'][2] for r in cases})
        for stratum in ['ordinary','random']:
            rows=[r for r in cases if r['stratum']==stratum]
            self.assertEqual(len(rows),32)
            self.assertEqual(sum(r['case'][0]=='amber' and r['case'][1] for r in rows),8)
            for length in [4,6,8,10]:
                self.assertEqual(sum(len(r['case'][2])==length for r in rows),8)

    def test_selection_uses_score_and_earliest_tie(self):
        self.assertEqual(select_candidate([{'correct':0},{'correct':9},{'correct':9}]),1)
        self.assertEqual(select_candidate([{'correct':0},{'correct':9},{'correct':12}]),2)
