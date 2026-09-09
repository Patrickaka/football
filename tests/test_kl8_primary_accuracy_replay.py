from copy import deepcopy
import json

import pytest

from scripts.diagnose.replay_kl8_primary_accuracy import (
    evaluate_rows, load_history, paired_seventh_number, summarize,
)


def draw(index):
    return {'issue': str(2026000 + index),
            'numbers': sorted(((index + n) % 80) + 1 for n in range(20))}


def test_sources_deduplicate_identical_draws_and_reject_conflicting_results(tmp_path):
    one, two = tmp_path / 'one.json', tmp_path / 'two.json'
    one.write_text(json.dumps({'results': [draw(1), draw(2)]}))
    two.write_text(json.dumps({'results': [draw(2), draw(3)]}))
    rows, audit = load_history([one, two])
    assert [row['issue'] for row in rows] == ['2026001', '2026002', '2026003']
    assert audit['exclusions'] == {'duplicate_issue': 1}
    changed = draw(3)
    changed['issue'] = '2026002'
    two.write_text(json.dumps({'results': [changed]}))
    with pytest.raises(ValueError, match='Conflicting draw'):
        load_history([one, two])


def test_replay_predictor_sees_only_the_past_and_future_results_do_not_change_earlier_tickets():
    history = [draw(index) for index in range(1, 105)]
    seen = []

    def predict(past):
        seen.append([row['issue'] for row in past])
        return {'select_6': past[-1]['numbers'][:6]}

    rows, _ = evaluate_rows(history, predictor=predict)
    assert len(rows) == 4
    for index, row in enumerate(rows, 100):
        assert max(seen[index - 100]) < row['issue']
        assert row['based_on_issue'] == history[index - 1]['issue']
    changed = deepcopy(history)
    changed[-1]['numbers'] = list(range(61, 81))
    second, _ = evaluate_rows(changed, predictor=predict)
    assert rows[:-1] == second[:-1]
    assert rows[-1]['tickets'] == second[-1]['tickets']


def test_report_compares_equal_size_primary_tickets_and_separates_mean_from_high_hits():
    rows = [{'issue': str(i), 'tickets': {'select_6': [1, 2, 3, 4, 5, 6]},
             'hits': {'select_6': h}} for i, h in enumerate([0, 1, 2, 3, 4])]
    result = summarize(rows)['plays']['select_6']
    assert result['n'] == 5 and result['mean_hits'] == 2
    assert result['fair_mean_hits'] == 1.5
    assert result['at_least']['4']['observed'] == .2
    assert result['at_least']['5']['observed'] == 0
    assert result['zero_rate'] == .2
    assert result['mean_consecutive_ticket_overlap'] == 6
    assert result['most_selected_numbers'][0] == (1, 5)


def test_seventh_number_gain_does_not_count_a_later_exclusion_round_as_primary_hit():
    rows = [{'hits': {'fu_shi_7_primary_ranked': new, 'fu_shi_7_legacy_reserved': old,
                      'round_9_best_hit': 7}}
            for new, old in [(3, 2), (2, 3), (2, 2)]]
    result = paired_seventh_number(rows)
    assert result['mean_hit_difference'] == 0
    assert result['new_better_issues'] == result['old_better_issues'] == 1
    assert result['descriptive_normal_ci95'][0] < 0 < result['descriptive_normal_ci95'][1]
