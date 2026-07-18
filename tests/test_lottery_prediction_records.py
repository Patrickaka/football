import copy
import unittest
from unittest.mock import patch

from src import ssq
from src import lottery


class PredictionRecordTests(unittest.TestCase):
    def _store_patches(self, module):
        state = []

        def load(_key, default):
            return copy.deepcopy(state) if state else copy.deepcopy(default)

        def save(_key, value):
            state[:] = copy.deepcopy(value)

        return state, patch.object(module.kv_store, 'load', side_effect=load), patch.object(
            module.kv_store, 'save', side_effect=save
        )

    def test_ssq_snapshot_is_deduplicated_and_settled_per_set(self):
        state, load_patch, save_patch = self._store_patches(ssq)
        sets = [
            {'red': [1, 2, 3, 10, 20, 30], 'blue': 8},
            {'red': [4, 5, 6, 11, 21, 31], 'blue': 9},
        ]
        with load_patch, save_patch:
            ssq.save_prediction_record('26002', sets)
            ssq.save_prediction_record('26002', sets)
            self.assertEqual(len(state), 1)

            settled = ssq.settle_prediction_records([
                {'period': '26002', 'red': [1, 2, 4, 7, 20, 33], 'blue': 8}
            ])
            records = ssq.load_prediction_records()

        self.assertEqual(settled, 1)
        self.assertTrue(records[0]['settled'])
        self.assertEqual(records[0]['results'][0]['red_hits'], 3)
        self.assertTrue(records[0]['results'][0]['blue_hit'])
        self.assertEqual(records[0]['results'][1]['red_hits'], 1)

    def test_dlt_pending_record_is_automatically_compared(self):
        state, load_patch, save_patch = self._store_patches(lottery)
        state.append({
            'period': '2026079',
            'based_on_issue': '2026078',
            'recommendations': {
                'rank': {'front': [1, 2, 3, 4, 5], 'back': [6, 7]},
            },
            'actual': None,
            'settled': False,
        })
        with load_patch, save_patch:
            settled = lottery.settle_predictions([
                {'issue': '2026079', 'front': [1, 2, 8, 9, 10], 'back': [6, 12]}
            ])
            records = lottery.load_online_predictions()

        self.assertEqual(settled, 1)
        self.assertEqual(records[0]['rank_front_hit'], 2)
        self.assertEqual(records[0]['rank_back_hit'], 1)
        self.assertTrue(records[0]['settled'])
        self.assertEqual(records[0]['integrity_status'], 'verified_forward')

    def test_dlt_legacy_record_is_not_reported_as_a_hit(self):
        state, load_patch, save_patch = self._store_patches(lottery)
        state.append({
            'period': '2026077',
            'recommendations': {
                'fusion': {'front': [4, 14, 19, 24, 27], 'back': [6, 7]},
            },
            'actual': None,
            'settled': False,
        })
        with load_patch, save_patch:
            lottery.settle_predictions([
                {'issue': '2026077', 'front': [4, 14, 19, 24, 27], 'back': [6, 7]}
            ])
            records = lottery.load_online_predictions()
            stats = lottery.calculate_online_stats()

        self.assertEqual(records[0]['integrity_status'], 'legacy_unverified')
        self.assertNotIn('fusion_front_hit', records[0])
        self.assertEqual(stats['settled_count'], 0)


if __name__ == '__main__':
    unittest.main()
