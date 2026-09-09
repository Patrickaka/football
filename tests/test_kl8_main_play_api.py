from unittest.mock import patch

import pytest

from src.api.services import kl8 as service


def parameters(**updates):
    data = {'play_type': 'select_6', 'feature_weights': '{"frequency":1}',
            'model_weights': '{"rank":1}', 'window_size': '100',
            'frequency_mode': 'hot', 'final_selection_mode': 'concentrated'}
    data.update(updates)
    return {key: [value] for key, value in data.items()}


def test_activation_validates_requested_ranking_modes_instead_of_silent_defaults():
    with patch.object(service, 'validate_and_activate_strategy', return_value={'activated': False}) as validate:
        response = service.kl8_activate_payload(parameters())
    assert 'error' not in response
    assert validate.call_args.kwargs['frequency_mode'] == 'hot'
    assert validate.call_args.kwargs['final_selection_mode'] == 'concentrated'


@pytest.mark.parametrize('updates', [
    {'frequency_mode': 'unknown'}, {'final_selection_mode': 'unknown'},
    {'final_max_last_numbers': '2'}, {'final_min_last_numbers': '0'},
])
def test_unsupported_selection_settings_cannot_be_silently_dropped(updates):
    with patch.object(service, 'validate_and_activate_strategy') as validate:
        response = service.kl8_activate_payload(parameters(**updates))
    assert 'error' in response
    validate.assert_not_called()
