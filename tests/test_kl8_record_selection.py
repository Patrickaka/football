from copy import deepcopy
from datetime import datetime
import hashlib
import json

import pytest

from src.kl8 import config, records
from src.kl8.record_selection import audit_prediction, audit_settlement, select_canonical_snapshots, strategy_cohort
from src.api.services import kl8 as service


def draw(issue='2026240'):
    return {'issue': issue, 'date': '2026-09-06', 'numbers': list(range(1, 21))}


def snapshot(identity='first', issue='2026240', **updates):
    return {'snapshot_id': identity, 'target_issue': issue, 'based_on_issue': str(int(issue) - 1),
            'version': config.KL8_PREDICTOR_VERSION, 'is_experiment': False,
            'predicted_at': '2026-09-06T10:00:00+08:00',
            'select_6': [1, 2, 3, 50, 51, 52], 'fu_shi_7': [1, 2, 3, 50, 51, 52, 53],
            'play_strategies': {'select_6': 'ranked'},
            'resolved_strategies': {'select_6': {'window_size': 100}}, **updates}


def settlement(prediction, digest):
    return {'snapshot_id': prediction['snapshot_id'], 'snapshot_sha256': digest,
            'actual_issue': prediction['target_issue'], 'actual_numbers': list(range(1, 21)),
            'prize_settlement': {'select_6': {'placed': True, 'hits': 3, 'bet': 2, 'prize': 3}},
            'fushi_settlement': {'fu_shi_7': {'placed': True, 'pool_hits': 3,
                                             'total_bet': 42, 'total_prize': 18}}}


def test_first_predraw_formal_wins_over_current_configuration_and_experiments():
    first = snapshot(version='old-version', strategy_config_fingerprint='old-config')
    later = snapshot('current', predicted_at='2026-09-06T11:00:00+08:00',
                     strategy_config_fingerprint='current-config')
    experiment = snapshot('experiment', is_experiment=True, predicted_at='2026-09-06T09:00:00+08:00')
    after = snapshot('after', predicted_at='2026-09-06T22:00:00+08:00')
    draws = {'2026240': draw()}
    for items in ([experiment, later, after, first], [first, after, later, experiment]):
        picked = service._dedupe_kl8_snapshots(items, draw_records=draws)
        assert len(picked) == 1
        assert picked[0]['snapshot_id'] == 'first'
        assert picked[0]['prediction_audit']['eligible'] is True


@pytest.mark.parametrize('updates,reason', [
    ({'predicted_at': '2026-09-06T10:00:00'}, 'unverified_legacy_timestamp'),
    ({'predicted_at_ns': 123}, 'invalid_prediction_epoch'),
    ({'predicted_at': '2026-09-06T21:30:00+08:00'}, 'prediction_at_or_after_draw'),
    ({'is_experiment': True}, 'experimental_prediction'),
    ({'based_on_issue': '2026240'}, 'invalid_prediction_issue_boundary'),
    ({'is_experiment': None}, 'formal_status_unverified'),
])
def test_unverifiable_predictions_remain_visible_without_accuracy_credit(updates, reason):
    original = snapshot(**updates)
    result = select_canonical_snapshots([original], {'2026240': draw()})[0]
    assert result['snapshot_id'] == original['snapshot_id']
    assert result['prediction_audit']['eligible'] is False
    assert result['prediction_audit']['reason'] == reason
    assert 'prediction_audit' not in original


def test_epoch_proves_legacy_local_timestamp_but_conflicting_evidence_fails():
    ns = int(datetime.fromisoformat('2026-09-06T10:00:00+08:00').timestamp() * 1_000_000_000)
    pred = snapshot(predicted_at='2026-09-06T10:00:00', predicted_at_ns=ns)
    assert audit_prediction(pred, draw())['eligible'] is True
    pred['predicted_at'] = '2026-09-06T10:01:00+08:00'
    assert audit_prediction(pred, draw())['reason'] == 'conflicting_prediction_timestamps'
    assert audit_prediction(snapshot(), {**draw(), 'draw_at': '2026-09-06T09:00:00+08:00'})[
        'reason'] == 'prediction_at_or_after_draw'
    assert audit_prediction(snapshot(), {'issue': '2026240'})['eligible'] is False


@pytest.mark.parametrize('tamper,reason', [
    ('hash', 'settlement_snapshot_hash_mismatch'), ('identity', 'settlement_identity_mismatch'),
    ('numbers', 'settlement_draw_numbers_mismatch'), ('hits', 'settlement_hit_count_mismatch'),
    ('pool_hits', 'settlement_hit_count_mismatch'),
])
def test_settlement_requires_matching_snapshot_draw_and_recomputed_hits(tamper, reason):
    pred = snapshot()
    settled = settlement(pred, 'correct-hash')
    assert audit_settlement(pred, settled, draw(), snapshot_sha256='correct-hash')['eligible']
    if tamper == 'hash':
        settled['snapshot_sha256'] = 'wrong'
    elif tamper == 'identity':
        settled['snapshot_id'] = 'other'
    elif tamper == 'numbers':
        settled['actual_numbers'][0] = 80
    elif tamper == 'hits':
        settled['prize_settlement']['select_6']['hits'] = 6
    else:
        settled['fushi_settlement']['fu_shi_7']['pool_hits'] = 7
    assert audit_settlement(pred, settled, draw(), snapshot_sha256='correct-hash')['reason'] == reason


def test_real_file_stats_count_each_issue_once_and_separate_version_and_strategy(tmp_path, monkeypatch):
    snapshots_dir, settlements_dir = tmp_path / 'snapshots', tmp_path / 'settlements'
    snapshots_dir.mkdir()
    settlements_dir.mkdir()
    monkeypatch.setattr(config, 'KL8_SNAPSHOT_DIR', snapshots_dir)
    monkeypatch.setattr(config, 'KL8_SETTLEMENT_DIR', settlements_dir)
    predictions = [snapshot(), snapshot('duplicate-later', predicted_at='2026-09-06T11:00:00+08:00'),
                   snapshot('old-version', '2026239', version='old-version'),
                   snapshot('old-strategy', '2026238', play_strategies={'select_6': 'previous'}),
                   snapshot('experiment', '2026237', is_experiment=True),
                   snapshot('after-draw', '2026236', predicted_at='2026-09-06T23:00:00+08:00')]
    actuals = [draw(pred['target_issue']) for pred in predictions]
    for pred in predictions:
        raw = json.dumps(pred)
        identity = pred['snapshot_id']
        (snapshots_dir / f'snapshot_{identity}.json').write_text(raw, encoding='utf-8')
        settled = settlement(pred, hashlib.sha256(raw.encode()).hexdigest())
        (settlements_dir / f'settlement_{identity}.json').write_text(json.dumps(settled), encoding='utf-8')
        if identity == 'first':
            (settlements_dir / 'settlement_copied.json').write_text(json.dumps(settled), encoding='utf-8')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*.json')}
    result = records._build_recent_settlement_performance(windows=(30,), history_data=actuals)
    assert result['available_count'] == 2  # current version, two independent issues
    assert result['all_versions_available_count'] == 3
    current = result['windows'][0]['play_stats']['select_6']
    assert current['settled_count'] == 1  # most recent current-version strategy only
    assert current['avg_hits'] == 3
    assert current['cohort']['strategy_id'] == 'ranked'
    assert len([c for c in result['strategy_cohorts'] if c['play_type'] == 'select_6']) == 3
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*.json')}


def test_cohorts_use_saved_parameters_and_do_not_attribute_previous_strategy_to_new_one(monkeypatch):
    pred = snapshot()
    changed = deepcopy(pred)
    changed['resolved_strategies']['select_6']['window_size'] = 500
    assert strategy_cohort(pred, 'select_6')['key'] != strategy_cohort(changed, 'select_6')['key']
    monkeypatch.setattr(config, 'ACTIVE_STRATEGIES', {'select_6': {'strategy_id': 'new', 'is_validated': True}})
    health = records._build_strategy_health({'windows': [{'window_size': 30, 'settled_count': 30,
        'play_stats': {'select_6': {'settled_count': 30, 'hit_delta_vs_random': 1.,
                                  'cohort': {'strategy_id': 'old'}}}}]})
    assert health['health_by_play']['select_6']['settled_count'] == 0
    assert health['health_by_play']['select_6']['status'] == 'pending'
