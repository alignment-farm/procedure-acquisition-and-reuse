import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from acquisition import examples, evaluation
from procedure_task import oracle

class AcquisitionTests(unittest.TestCase):
    def test_disjointness_and_factor_holdout(self):
        train=examples(); test=evaluation()
        self.assertEqual(len(train),12)
        self.assertEqual(len(test),64)
        self.assertEqual(len(set(test)),64)
        self.assertFalse({c[2] for c in train}&{c[2] for c in test})
        self.assertTrue(all(not (m=='amber' and u) for m,u,w in train))
        self.assertEqual(sum(m=='amber' and u for m,u,w in test),16)
        self.assertFalse({oracle(*c) for c in train}&{oracle(*c) for c in test})
        for n in [4,6,8,10]:
            self.assertEqual(sum(len(w)==n for m,u,w in test),16)
