"""Dixon-Coles 1X2 must use home goals as rows on both execution paths."""

from unittest.mock import patch

import pytest

from src.football import ml


@pytest.mark.skipif(not ml.NUMPY_AVAILABLE, reason="NumPy is unavailable")
@pytest.mark.parametrize("lam_home,lam_away", [(2.2, 0.7), (0.7, 2.2), (1.3, 1.3)])
@pytest.mark.parametrize("rho", [-0.08, 0.0, 0.08])
def test_numpy_and_list_1x2_probabilities_agree(lam_home, lam_away, rho):
    with patch.object(ml, "NUMPY_AVAILABLE", True):
        numpy_probs = ml.dixon_coles_1x2_prob(lam_home, lam_away, rho=rho)
    with patch.object(ml, "NUMPY_AVAILABLE", False):
        list_probs = ml.dixon_coles_1x2_prob(lam_home, lam_away, rho=rho)

    assert numpy_probs == pytest.approx(list_probs)
    assert sum(numpy_probs.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("use_numpy", [False, True])
def test_swapping_strong_and_weak_teams_swaps_win_probabilities(use_numpy):
    if use_numpy and not ml.NUMPY_AVAILABLE:
        pytest.skip("NumPy is unavailable")

    with patch.object(ml, "NUMPY_AVAILABLE", use_numpy):
        strong_home = ml.dixon_coles_1x2_prob(2.2, 0.7, rho=0.08)
        strong_away = ml.dixon_coles_1x2_prob(0.7, 2.2, rho=0.08)

    assert strong_home["home"] > strong_home["away"]
    assert strong_away["away"] > strong_away["home"]
    assert strong_home["home"] == pytest.approx(strong_away["away"])
    assert strong_home["away"] == pytest.approx(strong_away["home"])
    assert strong_home["draw"] == pytest.approx(strong_away["draw"])
