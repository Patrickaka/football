"""High-total scenarios from the final score matrix, without changing its rank."""
from .prediction_evaluation import score_matrix
from .scoring import _total_market_tempo_signal


def assess_score_tail(candidates, total=None):
    """4+ is a union of scores, never the probability of one suggested score."""
    try:
        probabilities = {f'{h}-{a}': p for (h, a), p in candidates}
        if len(probabilities) != len(candidates):
            raise ValueError('duplicate scores')
        matrix = score_matrix(probabilities)
    except (TypeError, ValueError):
        matrix = None
    if matrix is None:
        return {'available': False, 'reason': 'incomplete_or_invalid_score_matrix'}
    ordered = sorted(matrix, key=lambda score: (-matrix[score], tuple(map(int, score.split('-')))))
    tails = {f'{threshold}+': sum(p for score, p in matrix.items()
                                if sum(map(int, score.split('-'))) >= threshold)
             for threshold in (4, 5, 6)}
    high_scores = [
        {'score': score, 'probability': matrix[score], 'rank': rank,
         'total_goals': sum(map(int, score.split('-')))}
        for rank, score in enumerate(ordered, 1)
        if sum(map(int, score.split('-'))) >= 4 and matrix[score] > 0
    ][:3]
    movement = _total_market_tempo_signal(total or {})
    return {
        'available': True, 'version': 'final-score-tail-v1',
        'distribution_source': 'final_score_matrix', 'tail_probabilities': tails,
        'high_score_candidates': high_scores,
        'top3_contains_high_score': any(sum(map(int, score.split('-'))) >= 4 for score in ordered[:3]),
        'market_movement': movement,
        'probability_semantics': 'unconditional_full_match',
    }
