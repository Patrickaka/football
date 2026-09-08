"""Background-only fact research with immutable, time-addressable JSON caches."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid

from ...domain.sports.football.match_context import (
    CATEGORIES, SCHEMA_VERSION, build_match_context, require_timestamp, timestamp,
    audit_fact, facts_to_live_context,
)

_CALL_SLOTS = threading.BoundedSemaphore(2)


def utcnow():
    return datetime.now(timezone.utc)


class _Budget:
    """Hard caller deadline, including slow adapters/DNS/local model loading.

    A timed-out read-only call may finish in a daemon thread, but cannot write
    context. The global two-slot limit bounds outstanding calls across jobs.
    """
    def __init__(self, seconds, max_requests):
        self.deadline = time.monotonic() + max(0.01, min(float(seconds), 60))
        self.remaining_requests = max(0, min(int(max_requests), 30))

    def call(self, function):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or self.remaining_requests <= 0:
            raise TimeoutError('research budget exhausted')
        if not _CALL_SLOTS.acquire(blocking=False):
            raise TimeoutError('read-only tool slots exhausted')
        self.remaining_requests -= 1
        output = queue.Queue(maxsize=1)

        def run():
            try:
                output.put((True, function(max(0.01, remaining))))
            except Exception as exc:
                output.put((False, exc))
            finally:
                _CALL_SLOTS.release()

        threading.Thread(target=run, name='football-context-read', daemon=True).start()
        try:
            ok, value = output.get(timeout=max(0.01, remaining))
        except queue.Empty:
            raise TimeoutError('read-only tool timed out') from None
        if not ok:
            raise value
        return value


def _match(match):
    if not isinstance(match, dict) or not str(match.get('match_id') or '').strip():
        raise ValueError('match_id is required')
    normalized = deepcopy(match)
    normalized['match_id'] = str(match['match_id'])
    normalized['kickoff'] = require_timestamp(
        match.get('kickoff') or match.get('match_time') or match.get('time'), 'kickoff').isoformat()
    return normalized


def _existing_findings(match):
    """Reuse existing live_context without treating legacy ts as published_at."""
    context = match.get('live_context') or {}
    if not isinstance(context, dict):
        return []
    rows = list(context.get('evidence') or []) if isinstance(context.get('evidence'), list) else []
    rows += [dict(row, category='injuries', data={k: row[k] for k in ('player', 'status') if k in row})
             for row in context.get('injuries', []) if isinstance(row, dict)]
    for category, key in (('lineup', 'lineup'), ('schedule', 'schedule_density')):
        entry = context.get(key) or {}
        if not isinstance(entry, dict):
            continue
        for team in ('home', 'away'):
            data = entry.get(team)
            if isinstance(data, dict):
                rows.append(dict(entry, category=category, team=team, data=data))
    for category in ('h2h', 'weather'):
        entry = context.get(category)
        if isinstance(entry, dict) and entry:
            fields = ('games', 'home_wins', 'draws', 'away_wins', 'avg_goals', 'most_recent_match_at') if category == 'h2h' else (
                'temperature_celsius', 'precipitation_mm', 'wind_kmh', 'forecast_for')
            rows.append(dict(entry, category=category, team='match', data={k: entry[k] for k in fields if k in entry}))
    return rows[:100]


def _facts(document, collected_at):
    """Metadata comes from the captured source, not from extraction output."""
    if not isinstance(document, dict):
        return []
    claims = document.get('facts')
    if claims is None:
        claims = [document] if isinstance(document.get('data'), dict) else []
    if not isinstance(claims, list):
        return []
    rows = []
    for claim in claims[:30]:
        if not isinstance(claim, dict):
            continue
        rows.append({
            'category': claim.get('category'), 'team': claim.get('team'), 'data': claim.get('data'),
            'source': document.get('source') or claim.get('source'),
            'source_url': claim.get('source_url') or claim.get('url') or document.get('source_url') or document.get('url'),
            'published_at': claim.get('published_at', document.get('published_at')),
            'collected_at': claim.get('collected_at', document.get('collected_at', collected_at)),
            'confirmation_status': claim.get('confirmation_status', document.get('confirmation_status', 'unknown')),
            'quote': claim.get('quote', ''),
            'extraction_method': claim.get('extraction_method', 'structured_source'),
        })
    return rows


class IntelligenceAgent:
    def __init__(self, *, cache_dir=None, sources=(), extractor=None,
                 budget_seconds=12, max_requests=8, cache_ttl_seconds=1800,
                 queue_size=8, clock=utcnow):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[3] / 'data' / 'intelligence_context'
        self.sources, self.extractor = tuple(sources), extractor
        self.budget_seconds, self.max_requests = budget_seconds, max_requests
        self.cache_ttl_seconds, self.clock = cache_ttl_seconds, clock
        self._queue = queue.Queue(maxsize=max(1, min(int(queue_size), 64)))
        self._lock, self._pending = threading.Lock(), set()
        self._worker = None
        self.configuration_errors = []

    def get_status(self):
        return {'sources': [str(source.name) for source in self.sources],
                'ollama_configured': self.extractor is not None,
                'ollama_availability': 'not_probed' if self.extractor is not None else 'not_configured',
                'configuration_errors': list(self.configuration_errors),
                'pending_jobs': len(self._pending), 'queue_capacity': self._queue.maxsize,
                'budget_seconds': self.budget_seconds, 'max_requests': self.max_requests}

    def _directory(self, match_id):
        return self.cache_dir / hashlib.sha256(str(match_id).encode()).hexdigest()

    def get_cached_context(self, match_id, *, as_of, kickoff=None, max_age_seconds=None):
        """No network, model or queue wait. Only completed pre-cutoff snapshots."""
        cutoff = require_timestamp(as_of)
        expected_kickoff = require_timestamp(kickoff, 'kickoff') if kickoff is not None else None
        ttl = self.cache_ttl_seconds if max_age_seconds is None else max_age_seconds
        directory = self._directory(match_id)
        if not directory.is_dir():
            return None
        best, best_cutoff = None, None
        for path in sorted(directory.glob('*.json'), reverse=True)[:128]:
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                context = json.loads(path.read_text(encoding='utf-8'))
                if context.get('schema_version') != SCHEMA_VERSION or str(context.get('match_id')) != str(match_id):
                    continue
                captured, stored_cutoff = timestamp(context.get('captured_at')), timestamp(context.get('as_of'))
                match_kickoff = timestamp(context.get('kickoff'))
                if (not captured or not stored_cutoff or not match_kickoff or captured > cutoff or stored_cutoff > cutoff
                        or captured >= match_kickoff or cutoff >= match_kickoff
                        or (expected_kickoff is not None and expected_kickoff != match_kickoff)
                        or (cutoff - captured).total_seconds() > max(0, ttl)):
                    continue
                if best_cutoff is not None and stored_cutoff <= best_cutoff:
                    continue
                # Recheck cached claims. A serialized verified flag is not proof.
                result = build_match_context(context, context.get('evidence') or [], as_of=stored_cutoff,
                                             captured_at=captured, tool_log=context.get('tool_log', ()),
                                             errors=context.get('errors', ()))
                result['conflicts'] = sorted(set(result['conflicts']) | {
                    item for item in context.get('conflicts', []) if isinstance(item, str)})
                # Eligibility may expire after capture. Preserve snapshot time
                # while ensuring cached old injuries cannot become current features.
                for fact in result['evidence']:
                    if fact.get('verified'):
                        fresh = audit_fact(fact, as_of=cutoff, kickoff=match_kickoff)
                        if not fresh['verified']:
                            fact.update(verified=False, reasons=fresh['reasons'])
                result['live_context'] = facts_to_live_context(result['evidence'], as_of=cutoff, conflicts=result['conflicts'])
                result['eligibility_checked_at'] = cutoff.isoformat()
                available = {f['category'] for f in result['evidence'] if f.get('verified')}
                for category in ('injuries', 'lineup', 'schedule'):
                    if {f['team'] for f in result['evidence'] if f.get('verified') and f['category'] == category} != {'home', 'away'}:
                        available.discard(category)
                result['missing_categories'] = [category for category in CATEGORIES if category not in available]
                result['status'] = 'complete' if not result['missing_categories'] else 'partial' if available else 'unavailable'
                result['source_status'] = context.get('source_status', {})
                best, best_cutoff = result, stored_cutoff
            except (OSError, ValueError, TypeError, KeyError):
                continue
        return best

    def _save(self, context):
        directory = self._directory(context['match_id'])
        directory.mkdir(parents=True, exist_ok=True)
        name = require_timestamp(context['captured_at']).strftime('%Y%m%dT%H%M%S%f') + '-' + uuid.uuid4().hex
        path, temporary = directory / (name + '.json'), directory / (name + '.tmp')
        try:
            temporary.write_text(json.dumps(context, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def research_match_context(self, match, *, as_of=None, findings=(), force=False):
        """Blocking research: invoke only from a background job/CLI.

        Omitted as_of means current research; completion time freezes knowledge.
        Explicit as_of is a historical cutoff: no fresh network fetches, and a
        later collected document can never qualify for that earlier prediction.
        """
        match = _match(match)
        started = require_timestamp(self.clock())
        cutoff = require_timestamp(as_of) if as_of is not None else started
        if cutoff > started:
            raise ValueError('research as_of cannot be in the future')
        if not force and not findings:
            cached = self.get_cached_context(match['match_id'], as_of=cutoff, kickoff=match['kickoff'])
            if cached is not None:
                return cached
        facts, logs, errors = [], [], list(self.configuration_errors)
        budget = _Budget(self.budget_seconds, self.max_requests)

        def ingest(document):
            collected = require_timestamp(self.clock()).isoformat()
            existing = _facts(document, collected)
            facts.extend(existing)
            if (self.extractor is not None and isinstance(document, dict)
                    and document.get('snippet')):
                try:
                    extracted = budget.call(lambda timeout: self.extractor.extract(document, match, timeout=timeout))
                    # Only whitelisted extractor fields survive. Metadata is
                    # anchored to the original source and cannot be fabricated.
                    anchored = [{k: row[k] for k in ('category', 'team', 'data', 'quote', 'extraction_method') if k in row}
                                for row in extracted[:12] if isinstance(row, dict)]
                    for row in anchored:
                        row['confirmation_status'] = 'reported'
                    facts.extend(_facts(dict(document, facts=anchored), collected))
                    logs.append({'action': 'local_fact_extraction', 'status': 'completed', 'count': len(anchored)})
                except Exception as exc:
                    errors.append({'tool': 'local_fact_extraction', 'error': type(exc).__name__})
            elif not existing and isinstance(document, dict) and document.get('snippet'):
                facts.extend(_facts(dict(document, category='news', team='match',
                                         data={'headline': str(document['snippet'])[:500]},
                                         confirmation_status='reported'), collected))

        for document in [*_existing_findings(match), *list(findings)[:100]]:
            ingest(document)
        if as_of is None and started < timestamp(match['kickoff']):
            for category in CATEGORIES:
                for source in self.sources:
                    if category not in source.categories:
                        continue
                    interim = build_match_context(match, facts, as_of=self.clock(), captured_at=self.clock())
                    if category not in interim['missing_categories']:
                        break
                    source_name = str(source.name)[:100]
                    try:
                        documents = budget.call(lambda timeout: source.fetch(
                            match, category, as_of=started, timeout=timeout))
                        if not isinstance(documents, list):
                            raise ValueError('source must return a list')
                        logs.append({'action': 'read_source', 'source': source_name, 'category': category,
                                     'status': 'completed', 'documents': len(documents)})
                        for document in documents[:30]:
                            ingest(document)
                    except Exception as exc:
                        errors.append({'tool': source_name, 'category': category, 'error': type(exc).__name__})
                        logs.append({'action': 'read_source', 'source': source_name, 'category': category,
                                     'status': 'failed', 'error': type(exc).__name__})
                    if time.monotonic() >= budget.deadline or budget.remaining_requests <= 0:
                        break
                if time.monotonic() >= budget.deadline or budget.remaining_requests <= 0:
                    break
        elif as_of is not None:
            logs.append({'action': 'historical_cutoff', 'status': 'fresh_network_disabled'})
        completed = require_timestamp(self.clock())
        context = build_match_context(match, facts[:300], as_of=cutoff if as_of is not None else completed,
                                      captured_at=completed, tool_log=logs, errors=errors)
        context['tool_log'].append({'action': 'research', 'status': 'completed',
                                    'mode': 'historical_cutoff' if as_of is not None else 'current'})
        context['source_status'] = self.get_status()
        try:
            self._save(context)
        except (OSError, ValueError, TypeError) as exc:
            context['errors'].append({'tool': 'cache', 'error': type(exc).__name__})
        return context

    def submit_research(self, match, *, findings=(), force=False):
        """Nonblocking single-worker submission with finite queue and dedup."""
        try:
            match = _match(match)
        except (ValueError, TypeError):
            return {'submitted': False, 'status': 'invalid_match'}
        now = require_timestamp(self.clock())
        if timestamp(match['kickoff']) <= now:
            return {'submitted': False, 'status': 'match_started', 'match_id': match['match_id']}
        if not force and self.get_cached_context(match['match_id'], as_of=now, kickoff=match['kickoff']) is not None:
            return {'submitted': False, 'status': 'cached', 'match_id': match['match_id']}
        key = (match['match_id'], match['kickoff'])
        with self._lock:
            if key in self._pending:
                return {'submitted': False, 'status': 'pending', 'match_id': match['match_id']}
            try:
                self._queue.put_nowait((key, match, deepcopy(list(findings)[:100]), force))
            except queue.Full:
                return {'submitted': False, 'status': 'queue_full', 'match_id': match['match_id']}
            self._pending.add(key)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name='football-intelligence', daemon=True)
                self._worker.start()
        return {'submitted': True, 'status': 'queued', 'match_id': match['match_id']}

    def _work(self):
        while True:
            key, match, findings, force = self._queue.get()
            try:
                self.research_match_context(match, findings=findings, force=force)
            except Exception:
                # A failed background job cannot fail an online prediction.
                pass
            finally:
                with self._lock:
                    self._pending.discard(key)
                self._queue.task_done()
