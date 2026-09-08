"""Use server-owned current market snapshots for cold predictions, without network or persistence."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.api.services import bff
from src.api.services import football as service


def schedule_match():
    return {
        'match_id': 'sporttery-123', 'analysis_id': 'analysis-456', 'zgzcw_id': '789',
        'home': 'Home United', 'away': 'Away City', 'time': datetime.now().strftime('%Y-%m-%d 21:00'),
        'hkjc_id': 'hkjc-321', 'hkjc_updated_at': 'snapshot-current',
        'hkjc_had_odds': {'胜': 1.8, '平': 3.4, '负': 4.2},
        'asian_source': 'hkjc', 'asian_offer_matched': True,
        'asian_current': {'handicap': -0.5, 'home_odds': 1.84, 'away_odds': 2.02},
        'total_source': 'hkjc', 'total_offer_matched': True,
        'total_current': {'line': 2.5, 'over_odds': 1.95, 'under_odds': 1.88},
    }


class ColdPredictionContext(unittest.TestCase):
    def setUp(self):
        self.known = schedule_match()
        self.snapshot = mock.patch.object(service, '_prediction_schedule_snapshot', return_value=[self.known])
        self.read_snapshot = self.snapshot.start()
        self.addCleanup(self.snapshot.stop)
        self.analyze_patch = mock.patch.object(service, 'analyze_match', side_effect=lambda match, **_: deepcopy(match))
        self.analyze = self.analyze_patch.start()
        self.addCleanup(self.analyze_patch.stop)

    def test_single_cold_request_recovers_real_asian_and_total(self):
        params = {key: [self.known[key]] for key in ('match_id', 'home', 'away', 'time')}
        # Legacy MM-DD HH:MM is the same kickoff, not another fixture.
        params['time'] = [self.known['time'][5:]]
        params['asian_current'] = [json.dumps({'handicap': 99})]
        result = service.predict_payload(params)['result']
        self.assertEqual(result['asian_current'], self.known['asian_current'])
        self.assertEqual(result['total_current'], self.known['total_current'])
        self.assertEqual(result['hkjc_had_odds'], self.known['hkjc_had_odds'])
        self.assertTrue(result['asian_offer_matched'])
        self.assertEqual(result['hkjc_updated_at'], 'snapshot-current')
        self.assertEqual(result['time'], params['time'][0])

    def test_batch_uses_one_snapshot_and_does_not_trust_posted_markets(self):
        body = {'matches': [
            {'match_id': self.known['match_id'], 'home': self.known['home'], 'hkjc_id': 'injected',
             'asian_current': {'handicap': 99}, 'total_current': {'line': 99}},
            {'match_id': 'unknown', 'home': self.known['home'], 'away': self.known['away'],
             'hkjc_id': 'injected', 'asian_offer_matched': True},
        ]}
        outcomes = service.predict_batch_payload(body)['results']
        self.assertEqual(outcomes[0]['result']['hkjc_id'], self.known['hkjc_id'])
        self.assertEqual(outcomes[0]['result']['asian_current'], self.known['asian_current'])
        self.assertNotIn('hkjc_id', outcomes[1]['result'])
        self.assertNotIn('asian_offer_matched', outcomes[1]['result'])
        self.read_snapshot.assert_called_once_with()

    def test_unknown_match_is_not_matched_by_team_names(self):
        request = {'match_id': 'unknown', 'home': self.known['home'], 'away': self.known['away'],
                   'time': self.known['time']}
        result = service._with_current_market_context(request, [self.known])
        self.assertEqual(result, request)

    def test_alias_identifier_can_bind_when_primary_identifier_is_absent_in_snapshot(self):
        known = {key: value for key, value in self.known.items() if key != 'match_id'}
        request = {'match_id': 'request-id', 'analysis_id': self.known['analysis_id']}
        result = service._with_current_market_context(request, [known])
        self.assertEqual(result['hkjc_id'], known['hkjc_id'])
        self.assertEqual(result['match_id'], 'request-id')

    def test_conflicting_ids_teams_or_kickoffs_reject_enrichment(self):
        base = {key: self.known[key] for key in ('match_id', 'analysis_id', 'home', 'away', 'time')}
        for change in (
            {'analysis_id': 'wrong'}, {'home': 'Other United'}, {'away': 'Other City'},
            {'time': datetime.now().strftime('%Y-%m-%d 22:00')},
            {'time': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d 21:00')},
        ):
            with self.subTest(change=change):
                request = {**base, **change}
                self.assertEqual(service._with_current_market_context(request, [self.known]), request)

    def test_duplicate_stable_identifiers_are_not_arbitrarily_resolved(self):
        request = {'match_id': self.known['match_id']}
        self.assertEqual(service._with_current_market_context(request, [self.known, deepcopy(self.known)]), request)

    def test_context_copy_does_not_mutate_the_server_snapshot_or_request(self):
        request = {'match_id': self.known['match_id'], 'home': ' Home  United '}
        original = deepcopy(self.known)
        result = service._with_current_market_context(request, [self.known])
        result['asian_current']['handicap'] = 99
        self.assertEqual(self.known, original)
        self.assertEqual(request, {'match_id': self.known['match_id'], 'home': ' Home  United '})


class SnapshotFreshness(unittest.TestCase):
    def setUp(self):
        self.state = mock.patch.object(service, '_CURRENT_MATCH_SNAPSHOT', (None, ()))
        self.state.start()
        self.addCleanup(self.state.stop)

    def test_yesterday_memory_is_discarded_without_fetching(self):
        service._CURRENT_MATCH_SNAPSHOT = (datetime.now() - timedelta(days=1), (schedule_match(),))
        with mock.patch.object(service, '_read_persisted_match_snapshot', return_value=(None, ())), \
             mock.patch.object(service, 'fetch_match_list') as fetch:
            self.assertEqual(service._prediction_schedule_snapshot(), ())
        self.assertEqual(service._CURRENT_MATCH_SNAPSHOT, (None, ()))
        fetch.assert_not_called()

    def test_newer_disk_snapshot_wins_over_old_memory(self):
        now = datetime.now()
        before = schedule_match()
        after = {**before, 'hkjc_updated_at': 'newer-disk'}
        service._CURRENT_MATCH_SNAPSHOT = (now - timedelta(seconds=30), (before,))
        with mock.patch.object(service, '_read_persisted_match_snapshot', return_value=(now, (after,))):
            self.assertEqual(service._prediction_schedule_snapshot()[0]['hkjc_updated_at'], 'newer-disk')

    def test_today_snapshot_read_and_yesterday_rejected(self):
        from src.football import config
        now = datetime.now()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'matches.json'
            with mock.patch.object(config, 'MATCH_LIST_CACHE_PATH', str(path)):
                for delta, expected_count in ((0, 1), (-1, 0), (1, 0)):
                    with self.subTest(day_offset=delta):
                        path.write_text(json.dumps({'saved_at': (now + timedelta(days=delta)).isoformat(),
                                                   'matches': [schedule_match()]}), encoding='utf-8')
                        self.assertEqual(len(service._read_persisted_match_snapshot(now)[1]), expected_count)

    def test_disk_fallback_is_not_relabelled_as_a_new_live_snapshot(self):
        with mock.patch.object(service, 'fetch_match_list', return_value=[schedule_match()]), \
             mock.patch.object(service, 'get_match_list_status', return_value={'stale': True}), \
             mock.patch.object(service, '_remember_match_snapshot') as remember, \
             mock.patch.object(service, '_match_started', return_value=False), \
             mock.patch.object(service, 'football_reportable_ids', return_value=set()), \
             mock.patch.object(service, '_trigger_football_report_sync'), \
             mock.patch.object(service, '_trigger_football_analysis'), \
             mock.patch.object(service, '_attach_bayes_report_url', side_effect=lambda matches: matches):
            service.matches_payload()
        remember.assert_not_called()


class HomeMarketFreshness(unittest.TestCase):
    def test_bff_rejects_changed_hkjc_snapshot_but_keeps_identical_one(self):
        from src.football.config import FOOTBALL_PREDICTION_LOGIC_VERSION
        known = schedule_match()
        cached = {
            'model': {'prediction_logic_version': FOOTBALL_PREDICTION_LOGIC_VERSION},
            'asian': {'source_matched': True, 'updated_at': known['hkjc_updated_at'], 'handicap': -0.5},
            'total': {'source_matched': True, 'updated_at': known['hkjc_updated_at'], 'close_line': 2.5,
                      'source_event_id': known['hkjc_id'],
                      'close_water': {'over': 1.95, 'under': 1.88}},
        }
        for change, expected_ready in (('same', 1), ('timestamp', 0), ('event', 0), ('missing_event', 0)):
            with self.subTest(change=change):
                current, previous = deepcopy(known), deepcopy(cached)
                if change == 'timestamp':
                    current['hkjc_updated_at'] = 'new-snapshot'
                elif change == 'event':
                    current['hkjc_id'] = 'another-event'
                elif change == 'missing_event':
                    previous['total'].pop('source_event_id')
                with mock.patch.object(service, 'matches_payload', return_value={'matches': [current]}), \
                     mock.patch.object(bff, '_professional_status', return_value={}), \
                     mock.patch('src.football.config.CACHE_AVAILABLE', True), \
                     mock.patch('src.football.config.get_cache', return_value=previous), \
                     mock.patch('src.football.pipeline.analyze_match') as analyze:
                    payload = bff.football_home_payload()
                self.assertEqual(payload['coverage']['ready'], expected_ready)
                self.assertEqual(payload['coverage']['pending'], 1 - expected_ready)
                analyze.assert_not_called()


if __name__ == '__main__':
    unittest.main()
