"""Independent fixtures protect scoring and historical design during refactoring."""
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from procedure_task import design, oracle


class TaskTests(unittest.TestCase):
    def test_hand_checked_calls(self):
        fixtures = [
            (('amber', True, 'planet', 'upper'), 'dax PLANET-9'),
            (('amber', False, 'silver', 'reversal'), 'dax revlis-2'),
            (('blue', True, 'harbor', 'reversal'), 'wug harbor-9'),
            (('blue', False, 'meadow', 'upper'), 'wug meadow-2'),
        ]
        for args, expected in fixtures:
            self.assertEqual(oracle(*args), expected)

    def test_preserves_pilot_inputs_targets_and_order(self):
        root = Path(__file__).resolve().parents[1]
        old = [json.loads(l) for l in (root/'evidence/pilot-03-mlx/responses.jsonl').read_text().splitlines()]
        new = design('pilot')['requests']
        self.assertEqual(len(old), len(new))
        for before, after in zip(old, new):
            self.assertEqual(before['branch'], after['branch'])
            self.assertEqual(before['case'], list(after['case']))
            self.assertEqual(before['expected'], after['expected'])
            self.assertIn(after['content'], before['prompt'])

    def test_execution_coverage(self):
        requests = design('execution')['requests']
        self.assertEqual(len(requests), 54)
        self.assertEqual(len({r['id'] for r in requests}), 54)
        for branch in ['lesson_upper','lesson_steps']:
            rows = [r for r in requests if r['branch']==branch]
            self.assertEqual(len(rows), 24)
            self.assertEqual(len({tuple(r['case'][:2]) for r in rows}), 4)

if __name__ == '__main__':
    unittest.main()
