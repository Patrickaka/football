from copy import deepcopy
from unittest.mock import patch

from src.kl8 import config
from src.kl8.records import _strategy_fingerprint
from src.kl8.strategies import resolve_play_strategy


def strategy():
    return {'strategy_id': 'existing_select6', 'status': 'validated',
            'feature_weights': {'frequency': 1}, 'model_weights': {'rank': 1}, 'window_size': 100}


def test_legacy_select6_validation_does_not_claim_the_compound_play_is_validated(monkeypatch):
    active = {'select_6': strategy()}
    before = deepcopy(active)
    monkeypatch.setattr(config, 'ACTIVE_STRATEGIES', active)
    six, seven = resolve_play_strategy('select_6'), resolve_play_strategy('fu_shi_7')
    assert six['is_validated'] is True
    assert seven['is_validated'] is False
    assert seven['validation_scope'] == 'select_6_only'
    assert seven['feature_weights'] == six['feature_weights']
    assert seven['strategy_id'] == six['strategy_id']
    assert active == before


def test_compound_status_requires_joint_report_for_the_actual_configuration(monkeypatch):
    current = strategy()
    current['validation_report'] = {'main_play_validation': {'candidate_fingerprint': _strategy_fingerprint(current)}}
    monkeypatch.setattr(config, 'ACTIVE_STRATEGIES', {'select_6': current})
    with patch('src.kl8.main_play_validation.has_main_play_evidence', return_value=True):
        assert resolve_play_strategy('fu_shi_7')['is_validated'] is True
        current['window_size'] = 150
        assert resolve_play_strategy('fu_shi_7')['is_validated'] is False
