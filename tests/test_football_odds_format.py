"""市场公平概率必须使用来源明确的赔率格式。"""
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.domain.sports.football import markets
from src.domain.sports.football import parsing as domain_parsing
from src.football import parsing


def _market(odds_format, first, second):
    return {
        'odds_format': odds_format,
        'open': {'handicap': 0.5, 'line': 2.5,
                 'home_odds': first, 'away_odds': second,
                 'over_odds': first, 'under_odds': second},
        'close': {'handicap': 0.5, 'line': 2.5,
                  'home_odds': first, 'away_odds': second,
                  'over_odds': first, 'under_odds': second},
    }


@pytest.mark.parametrize('water,expected', [
    ((0.8, 1.0), 10 / 19),
    ((1.2, 1.4), 12 / 23),  # 两边均高于 1 仍是不含本金的香港水位。
])
def test_hong_kong_water_includes_principal_before_devig(water, expected):
    raw = _market('hong_kong', *water)
    asian, total = markets.analyze_asian(raw), markets.analyze_total(raw)
    assert asian['close_prob']['home_give'] == pytest.approx(expected)
    assert total['close_prob']['over'] == pytest.approx(expected)
    assert asian['odds_format'] == total['odds_format'] == 'hong_kong'
    assert asian['close_water']['home'] == water[0]  # 展示仍保留源站水位。
    assert total['close_water']['over'] == water[0]
    assert sum(asian['close_prob'].values()) == pytest.approx(1)
    assert sum(total['close_prob'].values()) == pytest.approx(1)


def test_hkjc_decimal_price_does_not_add_principal_twice():
    raw = _market('decimal', 1.79, 2.02)
    asian, total = markets.analyze_asian(raw), markets.analyze_total(raw)
    assert asian['close_prob']['home_give'] == pytest.approx(2.02 / 3.81)
    assert total['close_prob']['over'] == pytest.approx(2.02 / 3.81)
    assert asian['odds_format'] == total['odds_format'] == 'decimal'


def test_equivalent_prices_have_identical_probability_and_goal_target():
    hong_kong = _market('hong_kong', 0.79, 1.02)
    decimal = _market('decimal', 1.79, 2.02)
    assert markets.analyze_asian(hong_kong)['close_prob'] == pytest.approx(
        markets.analyze_asian(decimal)['close_prob'])
    assert markets.analyze_total(hong_kong)['implied_total'] == pytest.approx(
        markets.analyze_total(decimal)['implied_total'])


def test_unmarked_legacy_call_keeps_existing_contract():
    raw = _market('decimal', 0.8, 1.0)
    del raw['odds_format']
    assert markets.analyze_asian(raw)['close_prob']['home_give'] == pytest.approx(5 / 9)
    assert markets.analyze_total(raw)['close_prob']['over'] == pytest.approx(5 / 9)


@pytest.mark.parametrize('analyze', [markets.analyze_asian, markets.analyze_total])
def test_unknown_explicit_format_is_rejected(analyze):
    with pytest.raises(ValueError, match='unsupported market odds format'):
        analyze(_market('unknown', 0.8, 1.0))


def test_500_fetchers_mark_real_recorded_page_water():
    path = Path(__file__).parent / 'fixtures/football_odds_pages.json.gz'
    pages = json.loads(gzip.decompress(path.read_bytes()))
    for page_key, fetcher, probability_key in (
        ('yazhi:1430311', parsing.fetch_yazhi, 'home'),
        ('daxiao:1430311', parsing.fetch_daxiao, 'over'),
    ):
        html = pages[page_key]
        numbers = domain_parsing.extract_avg_numbers(html)
        with patch.object(parsing, '_fetch_avg_page', return_value=(html, numbers)):
            raw = fetcher('1430311')
        assert raw['odds_format'] == 'hong_kong'
        close = raw['close']
        if probability_key == 'home':
            first, second = close['home_odds'], close['away_odds']
            fair = next(iter(markets.analyze_asian(raw)['close_prob'].values()))
        else:
            first, second = close['over_odds'], close['under_odds']
            fair = markets.analyze_total(raw)['close_prob']['over']
        assert 0 < first < 1.5 and 0 < second < 1.5
        expected = (1 + second) / (2 + first + second)
        assert fair == pytest.approx(expected)
        assert fair != pytest.approx(second / (first + second))


def test_500_individual_company_prices_also_mark_water_format():
    row = [0.8, 0.5, 1.0, 0.9, 0.5, 1.1]
    with patch.object(parsing._fetching_mod, 'fetch', return_value='<html>'), \
            patch.object(parsing._p, 'extract_company_odds', return_value=row):
        result = parsing.fetch_single_company_odds('format-fixture')
    for company in result.values():
        assert company['asian']['odds_format'] == 'hong_kong'
        assert company['total']['odds_format'] == 'hong_kong'
