from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DispatchResult:
    power: np.ndarray
    charge_start: int | None
    discharge_start: int | None
    predicted_spread: float
    top2_spread: float
    top5_spread_mean: float
    top5_spread_std: float
    top1_top2_gap: float
    top1_top5_mean_gap: float
    top_candidate_count: int


def rank_day_pairs(
    prices: np.ndarray,
    *,
    block_size: int = 8,
    top_k: int | None = None,
) -> list[tuple[float, int, int]]:
    if len(prices) != 96:
        raise ValueError(f"expected 96 prices, got {len(prices)}")

    prefix = np.concatenate([[0.0], np.cumsum(prices)])
    block_sum = prefix[block_size:] - prefix[:-block_size]

    candidates: list[tuple[float, int, int]] = []
    for tc in range(0, 81):
        for td in range(tc + block_size, 89):
            spread = float(block_sum[td] - block_sum[tc])
            candidates.append((spread, tc, td))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if top_k is None:
        return candidates
    return candidates[:top_k]


def optimize_day(
    prices: np.ndarray,
    *,
    tau: float = 0.0,
    block_size: int = 8,
    charge_power: float = -1000.0,
    discharge_power: float = 1000.0,
) -> DispatchResult:
    """Find the best one charge block and one later discharge block for one day."""
    candidates = rank_day_pairs(prices, block_size=block_size)
    best_spread, best_tc_raw, best_td_raw = candidates[0]
    top_spreads = np.asarray([item[0] for item in candidates[:5]], dtype=float)
    top2_spread = float(candidates[1][0]) if len(candidates) > 1 else best_spread
    top5_mean = float(top_spreads.mean())
    top5_std = float(top_spreads.std())
    top1_top2_gap = float(best_spread - top2_spread)
    top1_top5_mean_gap = float(best_spread - top5_mean)

    power = np.zeros(96, dtype=float)
    best_tc: int | None
    best_td: int | None
    if best_spread > tau:
        best_tc = best_tc_raw
        best_td = best_td_raw
        power[best_tc : best_tc + block_size] = charge_power
        power[best_td : best_td + block_size] = discharge_power
    else:
        best_tc = None
        best_td = None

    return DispatchResult(
        power=power,
        charge_start=best_tc,
        discharge_start=best_td,
        predicted_spread=best_spread,
        top2_spread=top2_spread,
        top5_spread_mean=top5_mean,
        top5_spread_std=top5_std,
        top1_top2_gap=top1_top2_gap,
        top1_top5_mean_gap=top1_top5_mean_gap,
        top_candidate_count=len(candidates),
    )
