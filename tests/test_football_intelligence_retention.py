"""Real temporary files only; no application history, databases or network."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.domain.sports.football.match_context import SCHEMA_VERSION, build_match_context
from src.football.intelligence import IntelligenceAgent, cleanup_intelligence_cache
from src.football.intelligence import retention

NOW = datetime(2026, 9, 8, 10, tzinfo=timezone.utc)


class IntelligenceRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / 'cache'
        self.root.mkdir()
        self.counter = 0

    def directory(self, match_id):
        return self.root / hashlib.sha256(str(match_id).encode()).hexdigest()

    def context(self, match_id='match', captured=NOW, padding=0):
        match = {'match_id': match_id, 'home': 'Home', 'away': 'Away', 'kickoff': (NOW + timedelta(days=1)).isoformat()}
        context = build_match_context(match, [], as_of=captured, captured_at=captured)
        if padding:
            context['padding'] = 'x' * padding
        return context

    def snapshot(self, match_id='match', *, captured=NOW, padding=0):
        directory = self.directory(match_id)
        directory.mkdir(exist_ok=True)
        self.counter += 1
        path = directory / (captured.strftime('%Y%m%dT%H%M%S%f') + f'-{self.counter:032x}.json')
        path.write_text(json.dumps(self.context(match_id, captured, padding)), encoding='utf-8')
        return path

    def temporary_file(self, *, captured, mtime, name=None):
        directory = self.directory('match')
        directory.mkdir(exist_ok=True)
        self.counter += 1
        name = name or captured.strftime('%Y%m%dT%H%M%S%f') + f'-{self.counter:032x}.tmp'
        path = directory / name
        path.write_bytes(b'incomplete-write')
        os.utime(path, (mtime.timestamp(), mtime.timestamp()))
        return path

    def clean(self, **kwargs):
        return cleanup_intelligence_cache(self.root, now=NOW, **kwargs)

    def test_seven_day_age_uses_capture_time_not_refreshed_mtime(self):
        old = self.snapshot(captured=NOW - timedelta(days=8))
        old_size = old.stat().st_size
        boundary = self.snapshot(captured=NOW - timedelta(days=7))
        recent = self.snapshot(captured=NOW - timedelta(days=6))
        os.utime(old, (NOW.timestamp(), NOW.timestamp()))
        result = self.clean()
        self.assertEqual(result['removed_count'], 1)
        self.assertEqual(result['removed_bytes'], old_size)
        self.assertFalse(old.exists())
        self.assertTrue(boundary.exists())
        self.assertTrue(recent.exists())
        self.assertEqual(result['files'][0]['reason'], 'age')
        self.assertFalse(result['errors'])

    def test_default_eight_snapshots_per_match_preserve_newest(self):
        paths = [self.snapshot(captured=NOW - timedelta(minutes=i)) for i in range(11)]
        other = self.snapshot('other')
        result = self.clean()
        self.assertEqual(result['max_snapshots_per_match'], 8)
        self.assertEqual(result['removed_count'], 3)
        self.assertTrue(all(path.exists() for path in paths[:8]))
        self.assertTrue(all(not path.exists() for path in paths[8:]))
        self.assertTrue(other.exists())

    def test_global_capacity_evicts_oldest_across_matches(self):
        old = self.snapshot('old', captured=NOW - timedelta(hours=3), padding=5000)
        middle = self.snapshot('middle', captured=NOW - timedelta(hours=2), padding=3000)
        newest = self.snapshot('new', captured=NOW - timedelta(hours=1), padding=2000)
        capacity = middle.stat().st_size + newest.stat().st_size
        result = self.clean(max_bytes=capacity)
        self.assertFalse(old.exists())
        self.assertTrue(middle.exists() and newest.exists())
        self.assertLessEqual(result['remaining_bytes'], capacity)
        self.assertEqual(result['over_budget_bytes'], 0)
        self.assertEqual(result['files'][0]['reason'], 'capacity')

    def test_disabled_capacity_and_environment_overrides(self):
        paths = [self.snapshot(captured=NOW - timedelta(hours=i), padding=1000) for i in range(3)]
        self.assertEqual(self.clean(max_bytes=0)['removed_count'], 0)
        with patch.dict(os.environ, {'FOOTBALL_INTELLIGENCE_RETENTION_DAYS': '3',
                                    'FOOTBALL_INTELLIGENCE_MAX_SNAPSHOTS': '1',
                                    'FOOTBALL_INTELLIGENCE_MAX_BYTES': '0'}):
            result = self.clean()
        self.assertEqual(result['retention_days'], 3)
        self.assertEqual(result['removed_count'], 2)
        self.assertTrue(paths[0].exists())

    def test_dry_run_reports_exact_plan_and_never_creates_lock_or_deletes(self):
        paths = [self.snapshot(captured=NOW - timedelta(days=i)) for i in (1, 9)]
        before = {path.relative_to(self.root): path.read_bytes() for path in paths}
        result = self.clean(dry_run=True)
        self.assertEqual(result['planned_count'], 1)
        self.assertEqual(result['planned_bytes'], paths[1].stat().st_size)
        self.assertEqual(result['removed_count'], 0)
        self.assertEqual(result['removed_bytes'], 0)
        self.assertFalse((self.root / '.retention.lock').exists())
        self.assertTrue(all((self.root / path).read_bytes() == data for path, data in before.items()))

    def test_only_owned_json_and_old_orphan_tmp_are_deleted(self):
        foreign = self.snapshot(captured=NOW - timedelta(days=9))
        foreign.write_text(json.dumps({'prediction_events': ['business-history']}), encoding='utf-8')
        wrong_match = self.snapshot(captured=NOW - timedelta(days=9))
        wrong_match.write_text(json.dumps(self.context('wrong-match', NOW - timedelta(days=9))), encoding='utf-8')
        manual = self.directory('match') / 'manual.json'
        manual.write_text(json.dumps({'schema_version': SCHEMA_VERSION}), encoding='utf-8')
        stale = self.temporary_file(captured=NOW - timedelta(hours=2), mtime=NOW - timedelta(hours=2))
        writing = self.temporary_file(captured=NOW - timedelta(hours=2), mtime=NOW)
        new = self.temporary_file(captured=NOW, mtime=NOW - timedelta(hours=2))
        foreign_tmp = self.temporary_file(captured=NOW - timedelta(days=3), mtime=NOW - timedelta(days=3), name='other.tmp')
        outside = self.base / 'prediction_history.json'
        outside.write_text('business history', encoding='utf-8')
        result = self.clean(max_bytes=1)
        self.assertEqual(result['removed_count'], 1)
        self.assertFalse(stale.exists())
        self.assertTrue(all(p.exists() for p in (foreign, wrong_match, manual, writing, new, foreign_tmp, outside)))
        self.assertEqual(result['files'][0]['reason'], 'orphan_tmp')

    def test_changed_file_is_not_deleted_after_selection(self):
        path = self.snapshot(captured=NOW - timedelta(days=9))
        original = retention._scan
        def changed(*args):
            entries = original(*args)
            path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
            return entries
        with patch.object(retention, '_scan', side_effect=changed):
            result = self.clean()
        self.assertEqual(result['removed_count'], 0)
        self.assertTrue(path.exists())
        self.assertIn('changed during cleanup', result['errors'][0]['error'])

    def test_missing_root_invalid_settings_and_unsafe_leaf_fail_closed(self):
        missing = self.base / 'missing'
        self.assertEqual(cleanup_intelligence_cache(missing, now=NOW)['removed_count'], 0)
        self.assertFalse(missing.exists())
        path = self.snapshot(captured=NOW - timedelta(days=9))
        with patch.dict(os.environ, {'FOOTBALL_INTELLIGENCE_RETENTION_DAYS': 'nan'}):
            result = self.clean()
        self.assertTrue(result['errors'])
        self.assertTrue(path.exists())
        original = retention._is_reparse
        with patch.object(retention, '_is_reparse', side_effect=lambda candidate: candidate == path or original(candidate)):
            result = self.clean()
        self.assertTrue(path.exists())
        self.assertTrue(any('symlink or junction' in entry['error'] for entry in result['errors']))

    def _link_directory(self, link, target):
        if os.name == 'nt':
            completed = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)],
                                       capture_output=True, text=True)
            if completed.returncode:
                self.skipTest('Windows junction creation unavailable')
        else:
            link.symlink_to(target, target_is_directory=True)
        self.addCleanup(lambda: os.rmdir(link) if os.name == 'nt' and link.exists() else
                        link.unlink() if link.is_symlink() else None)

    def test_real_symlink_or_junction_cannot_escape_root_or_receive_writes(self):
        outside = self.base / 'outside'
        outside.mkdir()
        payload = self.context('linked', NOW - timedelta(days=9))
        directory = self.directory('linked')
        self._link_directory(directory, outside)
        path = outside / (timestamp_name(payload['captured_at']) + '-' + 'a' * 32 + '.json')
        path.write_text(json.dumps(payload), encoding='utf-8')
        result = self.clean()
        self.assertEqual(result['removed_count'], 0)
        self.assertTrue(path.exists())
        self.assertTrue(result['errors'])
        agent = IntelligenceAgent(cache_dir=self.root, clock=lambda: NOW)
        with self.assertRaises(OSError):
            agent._save(self.context('linked'))
        linked_root = self.base / 'linked-root'
        self._link_directory(linked_root, outside)
        result = cleanup_intelligence_cache(linked_root, now=NOW)
        self.assertTrue(result['errors'])
        self.assertTrue(path.exists())
        self.assertFalse((outside / '.retention.lock').exists())

    def test_match_scoped_cleanup_leaves_other_directories_untouched(self):
        target = self.snapshot('target', captured=NOW - timedelta(days=9))
        other = self.snapshot('other', captured=NOW - timedelta(days=9))
        result = self.clean(match_id='target', max_bytes=0)
        self.assertEqual(result['removed_count'], 1)
        self.assertFalse(target.exists())
        self.assertTrue(other.exists())

    def test_writer_prunes_locally_and_preserves_ttl_and_prediction_as_of(self):
        agent = IntelligenceAgent(cache_dir=self.root, clock=lambda: NOW)
        for i in range(12):
            agent._save(self.context(captured=NOW - timedelta(minutes=i)))
        self.assertEqual(len(list(self.directory('match').glob('*.json'))), 8)
        current = agent.get_cached_context('match', as_of=NOW)
        self.assertIsNotNone(current)
        self.assertEqual(current['captured_at'], NOW.isoformat())
        self.assertIsNone(agent.get_cached_context('match', as_of=NOW - timedelta(hours=1)))
        self.assertIsNone(agent.get_cached_context('match', as_of=NOW + timedelta(minutes=31)))

    def test_readers_writers_and_cleanup_can_race_without_partial_snapshots(self):
        agent = IntelligenceAgent(cache_dir=self.root, clock=lambda: NOW)
        agent._save(self.context())
        def write(i):
            agent._save(self.context(captured=NOW - timedelta(seconds=i)))
        def sweep(_):
            result = self.clean()
            self.assertFalse(result['errors'], result['errors'])
        def read(_):
            result = agent.get_cached_context('match', as_of=NOW)
            self.assertIsNotNone(result)
            self.assertEqual(result['captured_at'], NOW.isoformat())
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fn, i) for i in range(1, 13) for fn in (write, sweep, read)]
            for future in futures:
                future.result(timeout=10)
        self.assertLessEqual(len(list(self.directory('match').glob('*.json'))), 8)
        self.assertEqual(agent.get_cached_context('match', as_of=NOW)['captured_at'], NOW.isoformat())

    def test_cross_process_lock_prevents_cleanup_of_a_writer_owned_cache(self):
        path = self.snapshot(captured=NOW - timedelta(days=9))
        code = (
            'import json,sys; '
            'from src.football.intelligence.retention import cleanup_intelligence_cache; '
            'print(json.dumps(cleanup_intelligence_cache(sys.argv[1],now=sys.argv[2])))'
        )
        with retention.intelligence_cache_lock(self.root):
            child = subprocess.run([sys.executable, '-c', code, str(self.root), NOW.isoformat()],
                                   capture_output=True, text=True, timeout=8, check=True)
        result = json.loads(child.stdout)
        self.assertEqual(result['removed_count'], 0)
        self.assertTrue(any('lock is busy' in item['error'] for item in result['errors']))
        self.assertTrue(path.exists())
        self.assertEqual(self.clean()['removed_count'], 1)


def timestamp_name(value):
    return datetime.fromisoformat(value).strftime('%Y%m%dT%H%M%S%f')


if __name__ == '__main__':
    unittest.main()
