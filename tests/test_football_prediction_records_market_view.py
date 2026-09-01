import unittest
from unittest.mock import patch

from src.football import result_sync


class FootballPredictionRecordsMarketViewTests(unittest.TestCase):
    def test_save_keeps_both_markets_when_both_are_offered(self):
        history = result_sync.PredictionHistory.__new__(result_sync.PredictionHistory)
        history.records = []
        spf = {'H': 0.55, 'D': 0.25, 'A': 0.20}
        rqspf = {'让胜': 0.25, '让平': 0.45, '让负': 0.30}

        with patch.object(history, '_save_record', return_value='test'):
            history.add_prediction(
                match_id='spf-and-rqspf',
                league='英超',
                home='主队',
                away='客队',
                match_time='2099-08-21 22:00',
                match_num='周一001',
                predicted_scores={'2-1': 0.30},
                predicted_1x2=spf,
                lottery_handicap=-1,
                predicted_rqspf=rqspf,
                odds_data={'lottery': {
                    'offer_matched': True,
                    'spf_available': True,
                    'spf_odds': {'胜': 1.9, '平': 3.4, '负': 4.2},
                    'rqspf_available': True,
                    'rqspf_odds': {'让胜': 2.8, '让平': 3.5, '让负': 2.1},
                }},
            )

        saved = history.records[0]
        self.assertEqual(saved['predicted_1x2'], spf)
        self.assertEqual(saved['predicted_rqspf'], rqspf)
        self.assertEqual(saved['match_num'], '周一001')

    def test_save_completes_missing_rqspf_when_both_markets_are_offered(self):
        history = result_sync.PredictionHistory.__new__(result_sync.PredictionHistory)
        history.records = []

        with patch.object(history, '_save_record', return_value='test'):
            history.add_prediction(
                match_id='both-markets-missing-rqspf',
                league='德甲',
                home='主队',
                away='客队',
                match_time='2099-08-21 22:00',
                predicted_scores={'2-0': 0.30, '2-1': 0.25, '1-1': 0.25,
                                  '0-1': 0.20},
                predicted_1x2={'H': 0.55, 'D': 0.25, 'A': 0.20},
                lottery_handicap=-1,
                predicted_rqspf=None,
                odds_data={'lottery': {
                    'offer_matched': True,
                    'spf_available': True,
                    'spf_odds': {'胜': 1.9, '平': 3.4, '负': 4.2},
                    'rqspf_available': True,
                    'rqspf_odds': {'让胜': 2.8, '让平': 3.5, '让负': 2.1},
                }},
            )

        saved = history.records[0]
        self.assertTrue(saved['predicted_1x2'])
        self.assertEqual(set(saved['predicted_rqspf']), {'让胜', '让平', '让负'})
        self.assertAlmostEqual(sum(saved['predicted_rqspf'].values()), 1.0)

    def test_save_keeps_only_rqspf_when_standard_market_is_not_offered(self):
        history = result_sync.PredictionHistory.__new__(result_sync.PredictionHistory)
        history.records = []
        rqspf = {'让胜': 0.40, '让平': 0.35, '让负': 0.25}

        with patch.object(history, '_save_record', return_value='test'):
            history.add_prediction(
                match_id='rqspf-only',
                league='德国杯',
                home='奥斯纳',
                away='拜仁',
                match_time='2099-09-03 02:45',
                predicted_scores={'2-0': 0.30},
                predicted_1x2={'H': 0.80, 'D': 0.12, 'A': 0.08},
                base_1x2={'H': 0.78, 'D': 0.13, 'A': 0.09},
                ml_1x2={'H': 0.82, 'D': 0.11, 'A': 0.07},
                lottery_handicap=3,
                predicted_rqspf=rqspf,
                odds_data={'lottery': {
                    'offer_matched': True,
                    'primary_market': 'rqspf',
                    'spf_available': False,
                    'spf_odds': None,
                    'rqspf_available': True,
                    'rqspf_odds': {'让胜': 2.1, '让平': 4.2, '让负': 2.42},
                }},
            )

        saved = history.records[0]
        self.assertEqual(saved['predicted_1x2'], {})
        self.assertEqual(saved['base_1x2'], {})
        self.assertEqual(saved['ml_1x2'], {})
        self.assertEqual(saved['predicted_rqspf'], rqspf)
        self.assertEqual(saved['lottery_handicap'], 3)

    def test_list_exposes_market_predictions_and_only_actual_score(self):
        record = {
            'match_id': 'market-view-1',
            'home': '主队',
            'away': '客队',
            'match_time': '2020-01-01 12:00',
            'match_num': '周三006',
            'settled': True,
            'sync_status': 'synced',
            'predicted_scores': {'2-1': 0.3},
            'predicted_1x2': {'H': 0.55, 'D': 0.25, 'A': 0.20},
            'predicted_rqspf': {'让胜': 0.25, '让平': 0.45, '让负': 0.30},
            'lottery_handicap': -1,
            'actual_score': '2-1',
            'actual_result': 'H',
            'actual_rqspf': '让平',
            'hit_1x2': True,
            'hit_rqspf': True,
        }

        with patch.object(result_sync._global_history, 'records', [record]):
            row = result_sync.get_prediction_records(include_hidden=True)[0]

        self.assertEqual(row['predicted_1x2']['H'], 0.55)
        self.assertEqual(row['predicted_rqspf']['让平'], 0.45)
        self.assertEqual(row['lottery_handicap'], -1)
        self.assertEqual(row['match_num'], '周三006')
        self.assertEqual(row['actual_score'], '2-1')
        self.assertTrue(row['hit_1x2'])
        self.assertTrue(row['hit_rqspf'])
        self.assertNotIn('predicted_scores', row)

    def test_list_hides_legacy_spf_prediction_when_official_market_was_closed(self):
        record = {
            'match_id': 'market-view-spf-closed',
            'home': '阿森纳',
            'away': '考文垂',
            'match_time': '2026-08-21 22:00',
            'predicted_1x2': {'H': 0.80, 'D': 0.12, 'A': 0.08},
            'predicted_rqspf': {'让胜': 0.40, '让平': 0.35, '让负': 0.25},
            'lottery_handicap': -1,
            'hit_1x2': True,
            'odds_snapshot': {
                'lottery': {
                    'offer_matched': True,
                    'spf_available': True,
                    'spf_odds': None,
                    'rqspf_available': True,
                    'rqspf_odds': {'让胜': 2.1, '让平': 4.2, '让负': 2.42},
                },
            },
        }

        with patch.object(result_sync._global_history, 'records', [record]):
            row = result_sync.get_prediction_records(include_hidden=True)[0]

        self.assertEqual(row['predicted_1x2'], {})
        self.assertIsNone(row['hit_1x2'])
        self.assertTrue(row['predicted_rqspf'])

    def test_list_hides_model_spf_when_lottery_offer_was_not_verified(self):
        record = {
            'match_id': 'market-unverified',
            'home': '奥斯纳',
            'away': '拜仁',
            'match_time': '2026-09-03 02:45',
            'predicted_1x2': {'H': 0.31, 'D': 0.43, 'A': 0.26},
            'predicted_rqspf': None,
            'odds_snapshot': {'lottery': {
                'offer_matched': False,
                'unavailable_reason': 'network_error',
            }},
        }

        with patch.object(result_sync._global_history, 'records', [record]):
            row = result_sync.get_prediction_records(include_hidden=True)[0]

        self.assertEqual(row['predicted_1x2'], {})
        self.assertFalse(row['lottery_offer_matched'])
        self.assertEqual(row['lottery_unavailable_reason'], 'network_error')

    def test_list_completes_both_open_markets_for_legacy_record(self):
        record = {
            'match_id': 'legacy-both-open',
            'home': '主队',
            'away': '客队',
            'match_time': '2026-09-03 02:45',
            'predicted_scores': {
                '2-0': 0.30, '2-1': 0.25, '1-1': 0.25, '0-1': 0.20,
            },
            'predicted_1x2': {'H': 0.55, 'D': 0.25, 'A': 0.20},
            'predicted_rqspf': None,
            'lottery_handicap': -1,
            'odds_snapshot': {'lottery': {
                'offer_matched': True,
                'spf_available': True,
                'spf_odds': {'胜': 1.9, '平': 3.4, '负': 4.2},
                'rqspf_available': True,
                'rqspf_odds': {'让胜': 2.8, '让平': 3.5, '让负': 2.1},
            }},
        }

        with patch.object(result_sync._global_history, 'records', [record]):
            row = result_sync.get_prediction_records(include_hidden=True)[0]

        self.assertTrue(row['lottery_spf_available'])
        self.assertTrue(row['lottery_rqspf_available'])
        self.assertTrue(row['predicted_1x2'])
        self.assertEqual(set(row['predicted_rqspf']), {'让胜', '让平', '让负'})


if __name__ == '__main__':
    unittest.main()
