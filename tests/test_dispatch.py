from __future__ import annotations

import numpy as np

from electricity.dispatch import optimize_day, rank_day_pairs


def test_optimize_day_respects_block_constraints() -> None:
    prices = np.zeros(96)
    prices[10:18] = -1.0
    prices[30:38] = 2.0

    result = optimize_day(prices)

    assert result.charge_start == 10
    assert result.discharge_start == 30
    assert result.top_candidate_count == 3321
    assert result.top2_spread <= result.predicted_spread
    assert result.top1_top2_gap >= 0.0
    assert (result.power[10:18] == -1000).all()
    assert (result.power[30:38] == 1000).all()
    assert np.count_nonzero(result.power) == 16


def test_optimize_day_can_choose_no_trade() -> None:
    prices = np.ones(96)
    result = optimize_day(prices, tau=1.0)
    assert result.charge_start is None
    assert result.discharge_start is None
    assert np.count_nonzero(result.power) == 0


def test_rank_day_pairs_returns_sorted_legal_pairs() -> None:
    prices = np.zeros(96)
    prices[10:18] = -1.0
    prices[30:38] = 2.0

    pairs = rank_day_pairs(prices, top_k=3)

    assert len(pairs) == 3
    assert pairs[0] == (24.0, 10, 30)
    assert pairs[0][0] >= pairs[1][0] >= pairs[2][0]
    for _, charge_start, discharge_start in pairs:
        assert discharge_start >= charge_start + 8
