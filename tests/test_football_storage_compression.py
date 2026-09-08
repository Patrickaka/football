"""Repository archives preserve complete football evidence and fail closed."""
import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.common import repositories
from src.common import football_storage as storage
from src.football.prediction_events import append_prediction_event
from src.domain.sports.football.prediction_evaluation import event_hash, validate_event


NOW = datetime(2026, 9, 8, 12, tzinfo=timezone.utc)


def large_record(match_id='archive-1'):
    as_of = NOW - timedelta(days=60)
    matrix = {'0-0': .2, '0-1': .2, '1-0': .5, '1-1': .1}
    p = {'H': .5, 'D': .3, 'A': .2}
    variant = {'probabilities': p, 'score_probabilities': matrix, 'status': 'applied', 'applied': True}
    snapshot = {'probabilities': p, 'details': '完整证据，保留赛前资料。' * 300}
    record = {'match_id': match_id, 'home': '主队', 'away': '客队', 'league': 'League',
              'match_time': (as_of + timedelta(hours=2)).isoformat(),
              'predicted_1x2': p, 'predicted_scores': matrix, 'model_version': 'production-v1',
              'created_at': as_of.isoformat(), 'updated_at': (as_of + timedelta(hours=6)).isoformat(),
              'market_timeline': [{'captured_at': as_of.isoformat(), 'data': deepcopy(snapshot)} for _ in range(20)],
              'professional_snapshot': deepcopy(snapshot), 'time_layers': {'opening': snapshot},
              'unknown_open_field': {'keep': [None, False, 0, '']}}
    payload = {'as_of': as_of.isoformat(), 'kickoff_at': (as_of + timedelta(hours=2)).isoformat(),
               'model_version': 'production-v1', 'prediction_logic_version': 'logic-v1',
               'variants': {name: deepcopy(variant) for name in
                            ('statistical', 'market_adjusted', 'agent_adjusted', 'production')},
               'context': {'details': '冻结证据' * 1000}}
    payload['variants']['market_only'] = {'probabilities': p, 'score_probabilities': None,
                                         'status': 'applied', 'applied': True, 'captured_at': as_of.isoformat()}
    result = append_prediction_event(record, payload, now=as_of + timedelta(seconds=1))
    assert result['appended'], result
    record.update(settled=True, settled_at=(as_of + timedelta(hours=5)).isoformat(),
                  actual_result='H', actual_score='1-0', hit_1x2=True, result_quality={'grade': 'high'})
    return record


class GzipJsonTests(unittest.TestCase):
    def test_roundtrip_deterministic_and_outer_metadata_is_supported(self):
        value = {'中文': [None, False, 0, 1.25, '', {}, [], '重复' * 500]}
        envelope = storage.encode_json_gzip(value)
        self.assertEqual(envelope, storage.encode_json_gzip(value))
        self.assertEqual(storage.decode_json_gzip({**envelope, 'schema_version': 'events-v1', 'count': 1}), value)

    def test_corrupt_or_unsupported_envelopes_raise(self):
        envelope = storage.encode_json_gzip({'history': ['full evidence'] * 100})
        for field, value in [('encoding', 'pickle'), ('data', '*invalid*'), ('data', None),
                             ('sha256', '0' * 64), ('sha256', None), ('uncompressed_bytes', 1),
                             ('uncompressed_bytes', True), ('uncompressed_bytes', -1),
                             ('uncompressed_bytes', envelope['uncompressed_bytes'] + 1)]:
            with self.subTest(field=field, value=value):
                with self.assertRaises(storage.FootballStorageError):
                    storage.decode_json_gzip({**envelope, field: value})

    def test_truncated_crc_corrupt_trailing_and_concatenated_gzip_are_rejected(self):
        envelope = storage.encode_json_gzip({'history': ['evidence'] * 100})
        packed = base64.b64decode(envelope['data'])
        broken_crc = bytearray(packed)
        broken_crc[-8] ^= 1
        for bad in (packed[:-4], bytes(broken_crc), packed + b'trailing', packed + gzip.compress(b'{}')):
            with self.subTest(size=len(bad)):
                with self.assertRaises(storage.FootballStorageError):
                    storage.decode_json_gzip({**envelope, 'data': base64.b64encode(bad).decode('ascii')})

    def test_declared_oversize_is_rejected_before_decompression(self):
        envelope = storage.encode_json_gzip({'value': 'x' * 100})
        with patch.object(storage.zlib, 'decompressobj') as decompress:
            with self.assertRaises(storage.FootballStorageError):
                storage.decode_json_gzip({**envelope, 'uncompressed_bytes': storage.MAX_DECOMPRESSED_BYTES + 1})
        decompress.assert_not_called()
        with self.assertRaises(storage.FootballStorageError):
            storage.decode_json_gzip(envelope, max_uncompressed_bytes=10)
        with self.assertRaises(storage.FootballStorageError):
            storage.encode_json_gzip({'value': 'x' * 100}, max_uncompressed_bytes=10)

    def test_duplicate_json_keys_and_nonfinite_values_are_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'):
            envelope = {'encoding': 'gzip+base64', 'sha256': hashlib.sha256(raw).hexdigest(),
                        'uncompressed_bytes': len(raw), 'data': base64.b64encode(gzip.compress(raw)).decode('ascii')}
            with self.assertRaises(storage.FootballStorageError):
                storage.decode_json_gzip(envelope)
        with self.assertRaises(storage.FootballStorageError):
            storage.encode_json_gzip({'a': float('nan')})


class RecordArchiveTests(unittest.TestCase):
    def test_roundtrip_keeps_frozen_hash_and_basic_query_fields_without_mutating_input(self):
        record = large_record()
        original = deepcopy(record)
        packed = storage.encode_record(record, now=NOW)
        self.assertTrue(storage.is_archived(packed))
        self.assertLess(storage.serialized_size_bytes(packed), storage.serialized_size_bytes(record) * .2)
        for name in ('match_id', 'home', 'away', 'league', 'settled_at', 'actual_result', 'hit_1x2',
                     'result_quality', 'predicted_1x2', 'unknown_open_field', 'selected_prediction_event_id'):
            self.assertEqual(packed[name], record[name])
        self.assertNotIn('prediction_events', packed)
        self.assertNotIn('market_timeline', packed)
        restored = storage.decode_record(packed)
        self.assertEqual(restored, record)
        self.assertEqual(record, original)
        self.assertEqual(event_hash(restored['prediction_events'][0]), original['prediction_events'][0]['hash'])
        self.assertIsNone(validate_event(restored['prediction_events'][0]))

    def test_age_boundary_future_missing_and_unsettled_are_not_archived(self):
        record = large_record()
        for update in ({'settled': False}, {'settled': 'true'}, {'settled_at': None},
                       {'settled_at': (NOW + timedelta(days=1)).isoformat()},
                       {'settled_at': (NOW - timedelta(days=30) + timedelta(seconds=1)).isoformat()},
                       {'settled_at': '2026-01-01T00:00:00'}):
            with self.subTest(update=update):
                candidate = {**record, **update}
                self.assertEqual(storage.encode_record(candidate, now=NOW), candidate)
        self.assertTrue(storage.is_archived(storage.encode_record(
            {**record, 'settled_at': (NOW - timedelta(days=30)).isoformat()}, now=NOW)))

    def test_age_configuration_and_invalid_configuration(self):
        record = large_record()
        with patch.dict('os.environ', {'FOOTBALL_ARCHIVE_AFTER_DAYS': '100'}):
            self.assertFalse(storage.is_archived(storage.encode_record(record, now=NOW)))
            self.assertTrue(storage.is_archived(storage.encode_record(record, now=NOW, archive_after_days=10)))
        with patch.dict('os.environ', {'FOOTBALL_ARCHIVE_AFTER_DAYS': 'invalid'}):
            with self.assertRaises(storage.FootballStorageError):
                storage.encode_record(record, now=NOW)
        for days in (-1, True, 1.5):
            with self.assertRaises(storage.FootballStorageError):
                storage.encode_record(record, now=NOW, archive_after_days=days)

    def test_small_payload_stays_plain_and_existing_archive_is_idempotent(self):
        record = {'match_id': 'small', 'settled': True, 'settled_at': '2026-01-01T00:00:00Z',
                  'prediction_events': [{'value': 'small'}]}
        self.assertEqual(storage.encode_record(record, now=NOW), record)
        packed = storage.encode_record(large_record(), now=NOW)
        self.assertEqual(storage.encode_record(packed, now=NOW, archive_after_days=100), packed)
        self.assertIsNone(storage.decode_record(None))

    def test_lost_archive_or_manifest_cannot_be_mistaken_for_a_plain_record(self):
        packed = storage.encode_record(large_record(), now=NOW)
        for field in (storage.ARCHIVE_KEY, storage.ARCHIVE_MANIFEST_KEY):
            with self.subTest(field=field):
                broken = {key: value for key, value in packed.items() if key != field}
                self.assertTrue(storage.is_archived(broken))
                with self.assertRaises(storage.FootballStorageError):
                    storage.decode_record(broken)
                with self.assertRaises(storage.FootballStorageError):
                    storage.encode_record(broken)

    def test_manifest_collisions_missing_keys_and_unknown_schema_raise(self):
        packed = storage.encode_record(large_record(), now=NOW)
        for change in ({'schema_version': 'future-schema'}, {'fields': ['match_id']},
                       {'fields': ['prediction_events', 'prediction_events']}, {'fields': []},
                       {'fields': ['prediction_events']}, {'data': '!'}):
            with self.subTest(change=change):
                broken = {**packed, storage.ARCHIVE_KEY: {**packed[storage.ARCHIVE_KEY], **change}}
                with self.assertRaises(storage.FootballStorageError):
                    storage.decode_record(broken)
                with self.assertRaises(storage.FootballStorageError):
                    storage.encode_record(broken)
        with self.assertRaises(storage.FootballStorageError):
            storage.decode_record({**packed, 'prediction_events': []})

    def test_offloaded_memory_record_must_be_hydrated_before_persistence(self):
        with self.assertRaisesRegex(storage.FootballStorageError, 'Hydrate'):
            storage.encode_record({**large_record(), '_timeline_offloaded': True})


class RepositoryArchiveTests(unittest.TestCase):
    def test_upsert_get_load_boundaries_are_lossless_and_secondary_table_is_unchanged(self):
        record = large_record()
        with patch.object(repositories.doc_store, 'upsert_one', return_value='mysql') as write:
            self.assertEqual(repositories.football_prediction_upsert(record), 'mysql')
        persisted = json.loads(write.call_args.args[2][-1])
        self.assertTrue(storage.is_archived(persisted))
        with patch.object(repositories.doc_store, 'load_one', return_value=persisted):
            self.assertEqual(repositories.football_prediction_get(record['match_id']), record)
        with patch.object(repositories.doc_store, 'load_all', return_value=[persisted]):
            self.assertEqual(repositories.football_prediction_load(), [record])
        with patch.object(repositories.doc_store, 'load_all', return_value=[record]):
            self.assertEqual(repositories.prediction_record_load(), [record])
        with patch.object(repositories.doc_store, 'replace_all') as write:
            repositories.football_prediction_save([record])
            self.assertTrue(storage.is_archived(json.loads(write.call_args.args[2][0][-1])))
            repositories.prediction_record_save([record])
            self.assertEqual(json.loads(write.call_args.args[2][0][-1]), record)

    def test_corrupt_archive_read_and_batch_write_do_not_silently_drop_rows(self):
        packed = storage.encode_record(large_record(), now=NOW)
        packed[storage.ARCHIVE_KEY]['sha256'] = '0' * 64
        with patch.object(repositories.doc_store, 'load_all', return_value=[large_record(), packed]):
            with self.assertRaises(storage.FootballStorageError):
                repositories.football_prediction_load()
        with patch.object(repositories.doc_store, 'replace_all') as write:
            with self.assertRaises(storage.FootballStorageError):
                repositories.football_prediction_save([large_record(), packed])
        write.assert_not_called()
        with patch.object(repositories.doc_store, 'upsert_one') as write:
            with self.assertRaises(storage.FootballStorageError):
                repositories.football_prediction_upsert(packed)
        write.assert_not_called()

    def test_export_restores_archived_timeline_and_keeps_frozen_event_hash(self):
        from src.football import result_sync
        record = large_record()
        packed = storage.encode_record(record, now=NOW)
        history = result_sync.PredictionHistory.__new__(result_sync.PredictionHistory)
        history.records = [storage.decode_record(packed)]
        history.offload_stale_timelines(now=NOW.replace(tzinfo=None))
        with patch.object(repositories.doc_store, 'load_one', return_value=packed), \
             patch.object(result_sync, '_global_history', history):
            exported = result_sync.get_prediction_export()['records'][0]
        self.assertEqual(exported['market_timeline'], record['market_timeline'])
        self.assertEqual(exported['prediction_events'], record['prediction_events'])
        self.assertNotIn(storage.ARCHIVE_KEY, exported)
        self.assertEqual(event_hash(exported['prediction_events'][0]), record['prediction_events'][0]['hash'])


class ArchiveMaintenanceTests(unittest.TestCase):
    def test_mysql_dry_run_and_cas_preserve_exact_document_and_business_columns(self):
        record = large_record()
        raw = json.dumps(record, ensure_ascii=False)
        with patch.object(repositories.db, 'query_one', return_value={'raw_doc': raw}), \
             patch.object(repositories.db, 'execute', return_value=1) as write:
            report = repositories.football_prediction_archive_one(record['match_id'], now=NOW, dry_run=True)
            self.assertEqual(report['status'], 'would_archive')
            self.assertGreater(report['bytes_saved'], 1024)
            write.assert_not_called()
            report = repositories.football_prediction_archive_one(record['match_id'], now=NOW)
        self.assertEqual(report['status'], 'archived')
        sql, params = write.call_args.args
        self.assertIn('SET doc=%s WHERE match_id=%s', sql)
        self.assertIn('AS BINARY', sql)
        self.assertEqual(params[1:], (record['match_id'], raw))
        self.assertEqual(storage.decode_record(json.loads(params[0])), record)

    def test_cas_conflict_is_not_retried_over_newer_settlement(self):
        raw = json.dumps(large_record())
        with patch.object(repositories.db, 'query_one', return_value={'raw_doc': raw}), \
             patch.object(repositories.db, 'execute', return_value=0) as write, \
             patch.object(repositories, '_football_archive_fallback') as fallback:
            report = repositories.football_prediction_archive_one('archive-1', now=NOW)
        self.assertEqual(report['status'], 'changed')
        self.assertEqual(report['bytes_saved'], 0)
        self.assertEqual(write.call_count, 1)
        fallback.assert_not_called()

    def test_failed_or_uncertain_mysql_update_never_writes_stale_fallback(self):
        with patch.object(repositories.db, 'query_one', return_value={'raw_doc': json.dumps(large_record())}), \
             patch.object(repositories.db, 'execute', side_effect=OSError('connection lost')), \
             patch.object(repositories, '_football_archive_fallback') as fallback:
            report = repositories.football_prediction_archive_one('archive-1', now=NOW)
        self.assertEqual(report['status'], 'failed')
        fallback.assert_not_called()

    def test_already_archived_missing_and_not_eligible_never_rewrite(self):
        for row, status in ((None, 'missing'),
                            ({'raw_doc': json.dumps(storage.encode_record(large_record(), now=NOW))}, 'already_archived'),
                            ({'raw_doc': json.dumps({**large_record(), 'settled': False})}, 'not_eligible')):
            with self.subTest(status=status), patch.object(repositories.db, 'query_one', return_value=row), \
                 patch.object(repositories.db, 'execute') as write:
                report = repositories.football_prediction_archive_one('archive-1', now=NOW)
                self.assertEqual(report['status'], status)
                write.assert_not_called()

    def test_fallback_atomic_roundtrip_dry_run_idempotence_and_failure_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'football.json'
            record = large_record()
            content = json.dumps({'records': [record], 'backup_metadata': 'keep'}).encode()
            path.write_bytes(content)
            with patch.object(repositories.db, 'query_one', side_effect=OSError('offline')), \
                 patch.object(repositories.doc_store, '_record_degradation'), \
                 patch.object(repositories.doc_store, '_fallback_path', return_value=path):
                dry = repositories.football_prediction_archive_one('archive-1', now=NOW, dry_run=True)
                self.assertEqual(dry['status'], 'would_archive')
                self.assertEqual(path.read_bytes(), content)
                with patch.object(repositories.os, 'replace', side_effect=OSError('write failed')):
                    self.assertEqual(repositories.football_prediction_archive_one('archive-1', now=NOW)['status'], 'failed')
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(list(Path(folder).glob('*.tmp')), [])
                report = repositories.football_prediction_archive_one('archive-1', now=NOW)
                self.assertEqual(report['status'], 'archived')
                persisted = json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(persisted['backup_metadata'], 'keep')
                self.assertEqual(storage.decode_record(persisted['records'][0]), record)
                self.assertEqual(repositories.football_prediction_get('archive-1'), record)
                archived_bytes = path.read_bytes()
                self.assertEqual(repositories.football_prediction_archive_one('archive-1', now=NOW)['status'], 'already_archived')
                self.assertEqual(path.read_bytes(), archived_bytes)

    def test_corrupt_fallback_reports_failure_without_replacing_any_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'football.json'
            original = b'{malformed, retained for recovery'
            path.write_bytes(original)
            with patch.object(repositories.db, 'query_one', side_effect=OSError('offline')), \
                 patch.object(repositories.doc_store, '_record_degradation'), \
                 patch.object(repositories.doc_store, '_fallback_path', return_value=path):
                result = repositories.football_prediction_archive_one('archive-1', now=NOW)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
