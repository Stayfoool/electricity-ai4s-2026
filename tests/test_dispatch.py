from __future__ import annotations

import numpy as np
import pandas as pd

from electricity.dispatch import (
    build_dispatch_prior,
    compute_oracle_slot_counts,
    optimize_day,
    rank_day_pairs,
    smoothed_log_prior,
)


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


def test_optimize_day_zero_lambda_matches_no_prior() -> None:
    rng = np.random.default_rng(0)
    prices = rng.normal(size=96)
    log_pc = np.zeros(81)
    log_pd = np.zeros(81)
    a = optimize_day(prices)
    b = optimize_day(
        prices,
        log_prior_charge=log_pc,
        log_prior_discharge=log_pd,
        lambda_charge=0.0,
        lambda_discharge=0.0,
    )
    assert a.charge_start == b.charge_start
    assert a.discharge_start == b.discharge_start
    assert a.predicted_spread == b.predicted_spread


def test_optimize_day_uniform_log_prior_is_neutral() -> None:
    rng = np.random.default_rng(1)
    prices = rng.normal(size=96)
    log_pc = np.full(81, np.log(1.0 / 81))
    log_pd = np.full(81, np.log(1.0 / 81))
    a = optimize_day(prices)
    b = optimize_day(
        prices,
        log_prior_charge=log_pc,
        log_prior_discharge=log_pd,
        lambda_charge=5.0,
        lambda_discharge=5.0,
    )
    # Adding the same constant to every score does not change argmax order.
    assert a.charge_start == b.charge_start
    assert a.discharge_start == b.discharge_start


def test_optimize_day_strong_prior_overrides_spread() -> None:
    prices = np.zeros(96)
    prices[10:18] = -1.0  # natural choice: charge at slot 10
    prices[30:38] = 2.0   # natural choice: discharge at slot 30

    log_pc = np.full(81, -10.0)
    log_pd = np.full(81, -10.0)
    log_pc[20] = 0.0  # strongly prefer charging at slot 20
    log_pd[40 - 8] = 0.0  # strongly prefer discharging at slot 40 (td-block index)

    biased = optimize_day(
        prices,
        tau=-1.0,  # do not filter on zero-spread; the prior is what matters here
        log_prior_charge=log_pc,
        log_prior_discharge=log_pd,
        lambda_charge=100.0,
        lambda_discharge=100.0,
    )
    assert biased.charge_start == 20
    assert biased.discharge_start == 40
    # Reported spread is the raw price spread of the biased pick (here zero).
    expected_spread = float(prices[40:48].sum() - prices[20:28].sum())
    assert biased.predicted_spread == expected_spread


def test_smoothed_log_prior_normalizes_to_distribution() -> None:
    counts = np.array([2.0, 0.0, 0.0, 0.0])
    log_p = smoothed_log_prior(counts, alpha=1.0)
    p = np.exp(log_p)
    assert np.isclose(p.sum(), 1.0)
    # Slot 0 should have higher prior than the others (3 vs 1 after smoothing).
    assert log_p[0] > log_p[1]


def test_build_dispatch_prior_picks_up_oracle_slots() -> None:
    # Construct a tiny labels frame: 3 days where oracle is always
    # charge_start=4, discharge_start=20.
    rows = []
    base = pd.Timestamp("2025-01-01")
    for d in range(3):
        for slot in range(96):
            t = base + pd.Timedelta(days=d) + pd.Timedelta(minutes=15 * slot)
            price = 10.0
            if 4 <= slot < 12:
                price = -5.0
            if 20 <= slot < 28:
                price = 50.0
            rows.append({"times": t, "A": price})
    labels = pd.DataFrame(rows)

    counts_c, counts_d = compute_oracle_slot_counts(
        labels, time_col="times", target_col="A"
    )
    assert counts_c[4] == 3
    assert counts_c.sum() == 3
    assert counts_d[20 - 8] == 3
    assert counts_d.sum() == 3

    log_pc, log_pd = build_dispatch_prior(
        labels, time_col="times", target_col="A", alpha=0.5
    )
    assert log_pc[4] > log_pc[0]
    assert log_pd[20 - 8] > log_pd[0]
