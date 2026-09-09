"""Paired, descriptive audits of the two primary KL8 plays.

Each draw supplies the actual 20 numbers and exactly one six-number ticket plus
one seven-number pool for all C(7, 5) five-number combinations.  Pool hits and
per-combination hits use different denominators.  This module neither chooses
numbers nor activates strategies; inspected historical results are not a fresh
holdout or evidence of future non-inferiority.
"""
from __future__ import annotations

from collections import Counter
from math import comb

from .stats import hypergeom_expected, hypergeom_p_ge, hypergeom_pmf

PRIMARY_PLAY_KEYS = ('select_6', 'fu_shi_7')
PLAYS = {'select_6': 6, 'fu_shi_7': 7}


def _numbers(value, size, label):
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(type(number) is not int or not 1 <= number <= 80 for number in value)
            or len(set(value)) != size):
        raise ValueError(f'{label} must contain {size} distinct integers in 1..80')
    return frozenset(value)


def _validated_rows(rows, label):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError(f'{label} must contain at least one draw')
    validated = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f'{label} contains a non-object draw')
        issue = row.get('issue')
        if not isinstance(issue, str) or len(issue) != 7 or not issue.isascii() or not issue.isdigit():
            raise ValueError(f'{label} issue must be a seven-digit string')
        if issue in validated:
            raise ValueError(f'{label} contains duplicate issue {issue}')
        actual = _numbers(row.get('actual_numbers'), 20, f'{label}/{issue}/actual_numbers')
        tickets = row.get('tickets')
        if not isinstance(tickets, dict) or set(tickets) != set(PLAYS):
            raise ValueError(f'{label}/{issue} must contain only both primary play tickets')
        hits = {}
        for play, size in PLAYS.items():
            ticket = _numbers(tickets[play], size, f'{label}/{issue}/{play}')
            hits[play] = len(ticket & actual)
        validated[issue] = {'actual_numbers': actual, 'hits': hits}
    return validated


def _choose(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def _summarize(hits, play):
    size, n = PLAYS[play], len(hits)
    thresholds = (3, 4, 5, 6) if play == 'select_6' else (3, 4, 5)
    distribution = Counter(hits)
    summary = {
        'n': n, 'pick_size': size, 'total_hits': sum(hits),
        'mean_hits': sum(hits) / n, 'zero_count': distribution[0],
        'zero_rate': distribution[0] / n,
        'hit_distribution': {str(k): distribution[k] for k in range(size + 1)},
        'at_least': {
            str(k): {'count': sum(h >= k for h in hits),
                     'rate': sum(h >= k for h in hits) / n}
            for k in thresholds
        },
        'fair_baseline': {
            'mean_hits': hypergeom_expected(size), 'zero_rate': hypergeom_pmf(size, 0),
            'at_least': {str(k): hypergeom_p_ge(size, k) for k in thresholds},
        },
    }
    if play == 'fu_shi_7':
        # For h winners in a seven-number pool, exactly j winners in one of its
        # five-number tickets occurs C(h,j) * C(7-h,5-j) times.
        combo_counts = {str(j): sum(_choose(h, j) * _choose(7 - h, 5 - j) for h in hits)
                        for j in range(6)}
        total_tickets = n * comb(7, 5)
        summary['combinations'] = {
            'base_pick': 5, 'tickets_per_draw': 21, 'total_tickets': total_tickets,
            'hit_distribution': combo_counts,
            'exact_hit_rates': {key: count / total_tickets for key, count in combo_counts.items()},
            'at_least': {
                str(k): {'count': sum(combo_counts[str(j)] for j in range(k, 6)),
                         'rate': sum(combo_counts[str(j)] for j in range(k, 6)) / total_tickets,
                         'fair_baseline': hypergeom_p_ge(5, k)}
                for k in (3, 4, 5)
            },
            'full_hit_draws': sum(h >= 5 for h in hits),
            'full_hit_tickets': combo_counts['5'],
            'denominator_note': 'pool rates count draws; combination rates count all 21 tickets per draw',
        }
    return summary


def evaluate_main_play_pair(incumbent_rows, candidate_rows):
    """Compare identical draw coverage without trusting externally supplied hits.

    Input rows: ``{'issue': '2026241', 'actual_numbers': [20 unique ints],
    'tickets': {'select_6': [6 unique ints], 'fu_shi_7': [7 unique ints]}}``.
    Every monitored metric for both plays must be non-worse for the descriptive
    gate to pass, including per-combination prize thresholds.  Passing this
    historical gate never authorizes activation.  Sort ``ranking_key`` descending
    only within an explicitly designated development experiment.
    """
    incumbent = _validated_rows(incumbent_rows, 'incumbent')
    candidate = _validated_rows(candidate_rows, 'candidate')
    if set(incumbent) != set(candidate):
        raise ValueError('incumbent and candidate must cover exactly the same issues')
    issues = sorted(incumbent)
    for issue in issues:
        if incumbent[issue]['actual_numbers'] != candidate[issue]['actual_numbers']:
            raise ValueError(f'actual draw mismatch for issue {issue}')
    summaries = {
        name: {play: _summarize([rows[issue]['hits'][play] for issue in issues], play)
               for play in PLAYS}
        for name, rows in (('incumbent', incumbent), ('candidate', candidate))
    }
    comparison, failures, improvements = {}, [], []
    high_deltas, three_deltas, mean_deltas = [], [], []
    for play in PLAYS:
        before, after = summaries['incumbent'][play], summaries['candidate'][play]
        metrics = [
            ('mean_hits', before['total_hits'], after['total_hits'], len(issues), True),
            ('zero_rate', before['zero_count'], after['zero_count'], len(issues), False),
        ]
        for threshold in before['at_least']:
            metrics.append((f'at_least_{threshold}', before['at_least'][threshold]['count'],
                            after['at_least'][threshold]['count'], len(issues), True))
        if play == 'fu_shi_7':
            for threshold in ('3', '4', '5'):
                metrics.append((f'combination_at_least_{threshold}',
                                before['combinations']['at_least'][threshold]['count'],
                                after['combinations']['at_least'][threshold]['count'],
                                before['combinations']['total_tickets'], True))
        comparison[play] = {}
        for metric, old_count, new_count, denominator, higher_is_better in metrics:
            delta = (new_count - old_count) / denominator
            worse = new_count < old_count if higher_is_better else new_count > old_count
            better = new_count > old_count if higher_is_better else new_count < old_count
            item = {'metric': metric, 'incumbent': old_count / denominator,
                    'candidate': new_count / denominator, 'delta': delta,
                    'higher_is_better': higher_is_better, 'non_worse': not worse}
            comparison[play][metric] = item
            if worse:
                failures.append({'play': play, **item})
            if better:
                improvements.append({'play': play, 'metric': metric, 'delta': delta})
        high_deltas.extend(comparison[play][f'at_least_{k}']['delta'] for k in (4, 5))
        three_deltas.append(comparison[play]['at_least_3']['delta'])
        mean_deltas.append(comparison[play]['mean_hits']['delta'])
    return {
        'n': len(issues), 'issues': issues, **summaries, 'comparison': comparison,
        'gate': {'passed': not failures, 'has_improvement': bool(improvements),
                 'failures': failures, 'improvements': improvements,
                 'scope': 'descriptive_same_draw_comparison'},
        # The weakest of both plays' 4+/5+ changes takes priority over averages;
        # 3+ improvements cannot compensate for a failed per-metric gate.
        'ranking_key': [int(not failures), min(high_deltas), sum(high_deltas) / len(high_deltas),
                        min(three_deltas), sum(mean_deltas) / len(mean_deltas)],
        'promotion_allowed': False,
        'limitations': [
            'Historical non-worse metrics are not proof of future non-inferiority.',
            'Inspected periods are development data, not a fresh final holdout.',
            'The 21 tickets from one pool are dependent and are not 21 independent observations.',
        ],
    }
