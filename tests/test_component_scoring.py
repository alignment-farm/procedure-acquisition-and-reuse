import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_components import fields
from procedure_task import design

class ComponentTests(unittest.TestCase):
    def test_wrong_field_and_spacing_remain_distinct(self):
        result=fields('wug PLANET -2','dax PLANET-2')
        self.assertFalse(result['tool'])
        self.assertTrue(result['item'])
        self.assertTrue(result['suffix'])
        self.assertFalse(result['strict_format'])

    def test_extra_text_is_not_repaired(self):
        self.assertFalse(fields('Answer: dax PLANET-2','dax PLANET-2')['parseable'])

    def test_design_coverage(self):
        requests=design('components')['requests']
        self.assertEqual(len(requests),144)
        self.assertEqual(len({r['id'] for r in requests}),144)
        for branch in {r['branch'] for r in requests}:
            rows=[r for r in requests if r['branch']==branch]
            self.assertEqual(len(rows),24)
            self.assertEqual(len({tuple(r['case']) for r in rows}),24)
