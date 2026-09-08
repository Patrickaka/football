from datetime import datetime, timezone
from unittest.mock import patch

from src.domain.sports.football.readiness import build_match_evidence_profile
from src.football import parsing


def _results_page():
    def block(team, goals_for, goals_against):
        return (
            f'<span>{team}</span>近10场战绩<span class="ying">6胜</span>'
            '<span class="ping">3平</span><span class="shu">1负</span>'
            f'<span>进<span class="ying">{goals_for}球</span>'
            f'失<span class="shu">{goals_against}球</span></span>'
        )
    return block('丹麦', 28, 9) + block('民主刚果', 12, 4)


def test_fetched_team_history_keeps_origin_and_does_not_become_event_xg():
    before = datetime.now(timezone.utc)
    with patch.object(parsing._fetching_mod, 'fetch', return_value=_results_page()), \
            patch.object(parsing, 'ELO_AVAILABLE', False):
        team = parsing.fetch_team_strength('audit-match', '丹麦', '民主刚果')
    after = datetime.now(timezone.utc)

    metadata = team['data_provenance']['team_form']
    assert metadata['source'].endswith('/fenxi/shuju-audit-match.shtml')
    assert metadata['kind'] == 'historical_results'
    assert metadata['timestamp_basis'] == 'retrieved_at'
    assert metadata['source_published_at'] is None
    assert before <= datetime.fromisoformat(metadata['collected_at']) <= after
    audit = build_match_evidence_profile({'team': team})['data_provenance']
    assert audit['team_form']['verified'] is True
    assert audit['expected_goals']['verified'] is False


def test_failed_parse_does_not_invent_a_verified_data_source():
    with patch.object(parsing._fetching_mod, 'fetch', return_value='<html>missing</html>'):
        assert parsing.fetch_team_strength('audit-match', '丹麦', '民主刚果') is None


def test_internal_elo_origin_is_recorded_separately_from_observed_history():
    with patch.object(parsing._fetching_mod, 'fetch', return_value=_results_page()), \
            patch.object(parsing, 'ELO_AVAILABLE', True), \
            patch.object(parsing, 'get_elo_system') as get_elo, \
            patch.object(parsing, 'elo_to_goals_expected', return_value=1.5), \
            patch.object(parsing, 'elo_to_strength_factor', return_value=1.0):
        get_elo.return_value.get_rating.return_value = 1500
        get_elo.return_value.predict_match.return_value = {'home': .5}
        team = parsing.fetch_team_strength('audit-match', '丹麦', '民主刚果')
    audit = build_match_evidence_profile({'team': team})['data_provenance']
    assert audit['elo_expected_goals']['kind'] == 'elo_estimate'
    assert audit['elo_expected_goals']['source'] == 'internal_elo_rating_model'
    assert audit['expected_goals']['verified'] is False
