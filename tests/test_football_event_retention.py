"""Keep first-before selection immutable while bounding active event documents."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from src.common.football_storage import FootballStorageError, serialized_size_bytes
from src.domain.sports.football.prediction_evaluation import evaluate_frozen_events, matrix_outcomes, validate_event
from src.football.prediction_events import (
    ACTIVE_LIMIT_ENV, append_prediction_event, compact_prediction_events,
    hydrate_prediction_events, prediction_event_semantic_hash,
)
from tests.test_football_prediction_events import payload


START = datetime(2026, 1, 1, 8, tzinfo=timezone.utc)
KICKOFF = START + timedelta(hours=10)


def initial_payload():
    data = payload(START)
    data['kickoff_at'] = KICKOFF.isoformat()
    data['context']['intelligence']['items'][0].update(
        published_at=(START-timedelta(hours=1)).isoformat(),
        collected_at=(START-timedelta(minutes=5)).isoformat(),
        text='An observed, sourced team report. ' * 40,
    )
    data['context']['market'] = {'odds': {'H': 1.8, 'D': 3.4, 'A': 4.2},
                                 'source': {'captured_at': (START-timedelta(minutes=3)).isoformat()}}
    return data


def refresh(original, minute):
    data = deepcopy(original)
    observed = START+timedelta(minutes=minute)
    data['as_of'] = data['captured_at'] = observed.isoformat()
    for variant in data['variants'].values():
        variant['captured_at'] = observed.isoformat()
    return data


def add(row, data, minute, **kwargs):
    return append_prediction_event(row, data, now=START+timedelta(minutes=minute, seconds=1), **kwargs)


def unique_events(count, *, limit=8):
    row, originals = {'match_id': 'fixture-1'}, []
    base = initial_payload()
    for index in range(count):
        data = refresh(base, index)
        data['context']['intelligence']['items'][0]['id'] = f'news-{index}'
        result = add(row, data, index, max_active_events=limit)
        assert result['appended'], result
        originals.append(deepcopy(row['prediction_events'][-1]))
    return row, originals


class SemanticDeduplicationTests(unittest.TestCase):
    def test_many_run_clock_only_refreshes_keep_one_original_event(self):
        row, base = {'match_id': 'fixture-1'}, initial_payload()
        first = add(row, refresh(base, 0), 0)
        frozen = deepcopy(row['prediction_events'][0])
        for minute in range(1, 50):
            result = add(row, refresh(base, minute), minute)
            self.assertEqual(result['status'], 'duplicate')
            self.assertEqual(result['duplicate_kind'], 'semantic')
            self.assertFalse(result['appended'])
            self.assertFalse(result['storage_changed'])
            self.assertEqual(result['event_id'], first['event_id'])
        self.assertEqual(row['prediction_events'], [frozen])
        self.assertEqual(row['selected_prediction_event_id'], frozen['event_id'])
        self.assertNotIn('prediction_event_archive', row)

    def test_all_real_model_probability_evidence_and_cutoff_changes_are_distinct(self):
        def change_matrix(data):
            variant = data['variants']['statistical']
            variant['score_probabilities'].update({'0-0': .1, '1-0': .6})
            variant['probabilities'] = matrix_outcomes(variant['score_probabilities'])

        changes = {
            'model': lambda d: d.update(model_version='production-v2'),
            'logic': lambda d: d.update(prediction_logic_version='logic-v2'),
            'training': lambda d: d.update(training_cutoff_at=(START-timedelta(days=2)).isoformat()),
            'ml_training': lambda d: d['execution_trace']['ml_candidate'].update(training_cutoff_at=(START-timedelta(days=2)).isoformat()),
            'feature': lambda d: d['execution_trace']['ml_candidate']['feature_audit'].update(observed_form=2.1),
            'odds': lambda d: d['context']['market']['odds'].update(H=1.81),
            'source_capture': lambda d: d['context']['market']['source'].update(captured_at=(START-timedelta(minutes=2)).isoformat()),
            'published': lambda d: d['context']['intelligence']['items'][0].update(published_at=START.isoformat()),
            'collected': lambda d: d['context']['intelligence']['items'][0].update(collected_at=START.isoformat()),
            'evidence': lambda d: d['context']['intelligence']['items'][0].update(id='corrected-report'),
            'nested_as_of': lambda d: d['context']['intelligence'].update(as_of=START.isoformat()),
            'kickoff': lambda d: d.update(kickoff_at=(KICKOFF+timedelta(minutes=1)).isoformat()),
            'execution': lambda d: d['variants']['agent_adjusted'].update(applied=True, status='applied', fallback_reason=None),
            'matrix': change_matrix,
        }
        for label, change in changes.items():
            with self.subTest(change=label):
                row, base = {'match_id': 'fixture-1'}, initial_payload()
                add(row, refresh(base, 0), 0)
                first = deepcopy(row['prediction_events'][0])
                changed = refresh(base, 1)
                change(changed)
                result = add(row, changed, 1)
                self.assertTrue(result['appended'], result)
                self.assertEqual(len(row['prediction_events']), 2)
                self.assertEqual(row['prediction_events'][0], first)

    def test_existing_events_without_semantic_metadata_are_not_rewritten(self):
        from src.domain.sports.football.prediction_evaluation import event_hash
        row = {'match_id': 'fixture-1'}
        base = initial_payload()
        add(row, refresh(base, 0), 0)
        frozen = row['prediction_events'][0]
        frozen.pop('semantic_hash')
        frozen.pop('semantic_policy')
        frozen['hash'] = event_hash(frozen)
        frozen['event_id'] = frozen['hash']
        row['selected_prediction_event_id'] = frozen['event_id']
        original = deepcopy(frozen)
        result = add(row, refresh(base, 1), 1)
        self.assertEqual(result['duplicate_kind'], 'semantic')
        self.assertEqual(row['prediction_events'][0], original)
        self.assertIsNone(validate_event(original))

    def test_incoming_validation_runs_before_any_duplicate_shortcut(self):
        row, base = {'match_id': 'fixture-1'}, initial_payload()
        add(row, refresh(base, 0), 0)
        before = deepcopy(row)
        invalid_inputs = []
        naive = refresh(base, 1)
        naive['as_of'] = '2026-01-01T08:01:00'
        invalid_inputs.append(naive)
        future_capture = refresh(base, 1)
        future_capture['captured_at'] = (START+timedelta(hours=1)).isoformat()
        invalid_inputs.append(future_capture)
        missing_matrix = refresh(base, 1)
        missing_matrix['variants']['production']['score_probabilities'] = None
        invalid_inputs.append(missing_matrix)
        invalid_probability = refresh(base, 1)
        invalid_probability['variants']['production']['probabilities']['H'] = float('nan')
        invalid_inputs.append(invalid_probability)
        for data in invalid_inputs:
            self.assertEqual(add(row, data, 1)['status'], 'rejected')
            self.assertEqual(row, before)
        # Even the exact original payload cannot be called a duplicate after kickoff.
        late = append_prediction_event(row, refresh(base, 0), now=KICKOFF)
        self.assertEqual(late['status'], 'rejected')
        self.assertEqual(row, before)
        row['settled'] = True
        settled_before = deepcopy(row)
        self.assertEqual(add(row, refresh(base, 1), 1)['reason'], 'match_already_settled')
        self.assertEqual(row, settled_before)


class LosslessRetentionTests(unittest.TestCase):
    def test_keep_first_plus_latest_seven_and_restore_every_event_losslessly(self):
        row, originals = unique_events(25)
        self.assertEqual(len(row['prediction_events']), 8)
        self.assertEqual(row['prediction_events'], [originals[0], *originals[-7:]])
        self.assertEqual(row['selected_prediction_event_id'], originals[0]['event_id'])
        self.assertEqual(row['prediction_event_archive']['count'], 17)
        self.assertNotIn('event_ids', row['prediction_event_archive'])
        self.assertEqual(hydrate_prediction_events(row), originals)
        hydrated = hydrate_prediction_events(row)
        hydrated[0]['context']['intelligence'] = {}
        self.assertEqual(row['prediction_events'][0], originals[0])
        for event in originals:
            self.assertIsNone(validate_event(event))
        expanded = {**row, 'prediction_events': originals}
        expanded.pop('prediction_event_archive')
        expanded.pop('prediction_event_retention')
        self.assertLess(serialized_size_bytes(row), serialized_size_bytes(expanded) * .75)

    def test_archived_semantic_duplicate_does_not_create_a_new_event(self):
        row, originals = unique_events(14)
        first = deepcopy(row['prediction_events'][0])
        data = refresh(initial_payload(), 20)
        data['context']['intelligence']['items'][0]['id'] = 'news-2'
        result = add(row, data, 20)
        self.assertEqual(result['status'], 'duplicate')
        self.assertEqual(result['event_id'], originals[2]['event_id'])
        self.assertFalse(result['storage_changed'])
        self.assertEqual(hydrate_prediction_events(row), originals)
        self.assertEqual(row['prediction_events'][0], first)

    def test_limit_configuration_and_legacy_compaction_keep_the_first_id(self):
        row, originals = unique_events(15, limit=100)
        first = deepcopy(originals[0])
        with patch.dict('os.environ', {ACTIVE_LIMIT_ENV: '1'}):
            status = compact_prediction_events(row)
        self.assertEqual(status['active_count'], 2)
        self.assertEqual(row['prediction_events'], [first, originals[-1]])
        self.assertEqual(hydrate_prediction_events(row), originals)
        self.assertEqual(row['selected_prediction_event_id'], first['event_id'])
        # Increasing a limit can hydrate recent events back into the active window.
        compact_prediction_events(row, max_active_events=4)
        self.assertEqual(row['prediction_events'], [first, *originals[-3:]])
        self.assertEqual(hydrate_prediction_events(row), originals)
        self.assertFalse(compact_prediction_events(row, max_active_events=4)['storage_changed'])

    def test_duplicate_refresh_can_compact_old_unbounded_active_list(self):
        row, originals = unique_events(20, limit=100)
        data = refresh(initial_payload(), 25)
        data['context']['intelligence']['items'][0]['id'] = 'news-19'
        result = add(row, data, 25)
        self.assertEqual(result['status'], 'duplicate')
        self.assertFalse(result['appended'])
        self.assertTrue(result['storage_changed'])
        self.assertEqual(len(row['prediction_events']), 8)
        self.assertEqual(hydrate_prediction_events(row), originals)

    def test_normal_first_before_evaluation_is_identical_after_compaction(self):
        row, originals = unique_events(20, limit=100)
        row.update(settled=True, actual_score='1-0', actual_result='H',
                   settled_at=(KICKOFF+timedelta(hours=3)).isoformat())
        before = evaluate_frozen_events([row], include_confidence_intervals=False)
        compact_prediction_events(row)
        after = evaluate_frozen_events([row], include_confidence_intervals=False)
        self.assertEqual(after, before)
        self.assertEqual(row['prediction_events'][0], originals[0])

    def test_missing_corrupt_archive_or_count_metadata_fails_closed_without_mutation(self):
        original, _ = unique_events(12)
        def corrupt_data(row):
            row['prediction_event_archive']['data'] = '!invalid!'
        mutations = (
            lambda row: row.pop('prediction_event_archive'),
            lambda row: row.pop('prediction_event_retention'),
            corrupt_data,
            lambda row: row['prediction_event_archive'].update(count=900),
            lambda row: row['prediction_event_archive'].update(sha256='0'*64),
        )
        for mutation in mutations:
            row = deepcopy(original)
            mutation(row)
            before = deepcopy(row)
            with self.assertRaises(FootballStorageError):
                hydrate_prediction_events(row)
            self.assertEqual(compact_prediction_events(row)['status'], 'rejected')
            self.assertEqual(add(row, refresh(initial_payload(), 20), 20)['status'], 'rejected')
            self.assertEqual(row, before)

    def test_archive_encoding_failure_never_drops_or_appends_events(self):
        row, originals = unique_events(8)
        before = deepcopy(row)
        data = refresh(initial_payload(), 9)
        data['context']['intelligence']['items'][0]['id'] = 'new-unique-news'
        with patch('src.football.prediction_events.encode_json_gzip', side_effect=FootballStorageError('test failure')):
            result = add(row, data, 9)
        self.assertEqual(result['reason'], 'event_archive_write_failed')
        self.assertFalse(result['appended'])
        self.assertEqual(row, before)
        self.assertEqual(hydrate_prediction_events(row), originals)

    def test_reordered_active_tail_is_rejected(self):
        row, _ = unique_events(12)
        row['prediction_events'][-2:] = reversed(row['prediction_events'][-2:])
        with self.assertRaises(FootballStorageError):
            hydrate_prediction_events(row)

    def test_bad_limit_and_falsey_non_list_events_cannot_replace_existing_state(self):
        row, _ = unique_events(3)
        before = deepcopy(row)
        with patch.dict('os.environ', {ACTIVE_LIMIT_ENV: 'bad'}):
            result = add(row, refresh(initial_payload(), 4), 4)
        self.assertEqual(result['reason'], 'invalid_retention_configuration')
        self.assertEqual(row, before)
        for value in ({}, '', False):
            malformed = {'match_id': 'fixture-1', 'prediction_events': value}
            with self.assertRaises(FootballStorageError):
                hydrate_prediction_events(malformed)
            before = deepcopy(malformed)
            self.assertEqual(add(malformed, refresh(initial_payload(), 0), 0)['status'], 'rejected')
            self.assertEqual(malformed, before)


if __name__ == '__main__':
    unittest.main()
