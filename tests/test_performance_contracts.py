"""Output and concurrency contracts for the accelerated pipeline; no API calls."""
import itertools
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dig'))
import extract_lines as el
import figure_index as fi
import vlm_tasks as vt
from detection_cache import cached_detection
from vlm_client import DeepSeekVLM


class PerformanceContracts(unittest.TestCase):
    def test_longest_run_matches_reference_exhaustively(self):
        def reference(mask):
            best, start, current = 0, -1, 0
            for i, value in enumerate(mask):
                current = current + 1 if value else 0
                if current > best:
                    best, start = current, i - current + 1
            return best, start
        for size in range(11):
            for mask in itertools.product((False, True), repeat=size):
                self.assertEqual(el.longest_run(mask), reference(mask))
        rng = np.random.default_rng(42)
        for _ in range(100):
            mask = rng.random(4096) < .7
            self.assertEqual(el.longest_run(mask[::2]), reference(mask[::2]))

    def test_detection_cache_content_parameters_and_mutation(self):
        calls = []
        @cached_detection
        def detect(img, threshold=5):
            calls.append(1)
            return [int((img < threshold).sum())]
        img = np.zeros((20, 30), dtype=np.uint8)
        result = detect(img)
        result.append('caller mutation')
        self.assertEqual(detect(img.copy(), threshold=5), [600])
        self.assertEqual(len(calls), 1)
        img[0, 0] = 10
        self.assertEqual(detect(img), [599])
        self.assertEqual(detect(img, 11), [600])
        for i in range(20):
            detect(img, i)
        before = len(calls)
        detect(img, 100)
        self.assertEqual(len(calls), before + 1)

    def test_parallel_selection_matches_serial_and_merges_usage(self):
        cards = [{'id': f'f{i}', 'page': 1, 'label': f'Figure {i}',
                  'caption': 'test', 'thumb': Path(f'{i}.png')} for i in range(1, 5)]
        barrier = threading.Barrier(2)
        children = []
        def inspect(client, want, index, image):
            children.append(client)
            if os.environ['DIG_SELECT_WORKERS'] == '2':
                barrier.wait(timeout=5)
            client.usage['calls'] += 1
            client.usage['prompt_tokens'] += 10
            return {'is_data_chart': True, 'panels': [{'matches_request': True,
                    'axes_match': True, 'conditions_match': True}]}
        results = []
        for workers in ('1', '2'):
            client = DeepSeekVLM(api_key='test', log=False, cache=False)
            with patch.dict(os.environ, DIG_SELECT_WORKERS=workers), \
                 patch.object(vt, 'inspect_figure', side_effect=inspect), \
                 patch.object(vt, 'select_figures_by_evidence', return_value={'selected': [1, 3]}):
                results.append(fi.select('time versus distance', cards, client))
            self.assertEqual(client.usage['calls'], 4)
            self.assertEqual(client.usage['prompt_tokens'], 40)
        self.assertEqual(results[0], results[1])
        self.assertEqual(len({id(child) for child in children[-4:]}), 4)

    def test_auth_failure_stops_following_batches_and_retains_uncertainty(self):
        cards = [{'id': f'f{i}', 'page': 1, 'label': f'Figure {i}',
                  'caption': 'test', 'thumb': Path(f'{i}.png')} for i in range(7)]
        client = DeepSeekVLM(api_key='test', log=False, cache=False)
        with patch.dict(os.environ, DIG_SELECT_WORKERS='2'), \
             patch.object(vt, 'inspect_figure', side_effect=RuntimeError('HTTP 401')) as inspect:
            result = fi.select('time versus distance', cards, client)
        self.assertEqual(inspect.call_count, 2)
        self.assertIsNone(result['missing'])
        self.assertEqual(len(result['unresolved_ids']), 7)

    def test_fork_preserves_configuration_but_not_usage(self):
        client = DeepSeekVLM(api_key='test', base_url='http://localhost:1234',
                            timeout=8, cache=False, log=False, thinking=True)
        client.usage['calls'] = 9
        child = client.fork()
        self.assertEqual(child.timeout, 8)
        self.assertEqual(child.base_url, client.base_url)
        self.assertTrue(child.thinking)
        self.assertFalse(child.use_cache)
        self.assertFalse(child.use_log)
        self.assertEqual(child.usage['calls'], 0)


if __name__ == '__main__':
    unittest.main()
