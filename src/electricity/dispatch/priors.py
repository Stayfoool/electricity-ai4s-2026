"""Slot-level priors for dispatch optimization.

Computes oracle (charge_start, discharge_start) frequency tables from training
labels and converts them into log-prior arrays consumed by ``optimize_day``.

The priors capture systematic biases observed across the historical training
days and are used to nudge the dispatch optimizer away from systematically
over-represented mistakes (e.g., the model preferring afternoon charging
windows when oracle most often starts in the morning).

All inputs are restricted to data **strictly available at decision time**
(``train_end``), so producing the prior introduces no leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_BLOCK = 8
N_TC = 81  # tc in [0, 80]
N_TD = 81  # td in [BLOCK, 88], indexed at td - BLOCK


def _daily_oracle(prices: np.ndarray) -> tuple[int, int, float]:
    """Return (charge_start, discharge_start, spread) maximizing the day spread."""
    if len(prices) != 96:
        raise ValueError(f"expected 96 prices, got {len(prices)}")
    prefix = np.concatenate([[0.0], np.cumsum(prices)])
    block_sum = prefix[_BLOCK:] - prefix[:-_BLOCK]
    best_spread = -np.inf
    best_tc = 0
    best_td = _BLOCK
    for tc in range(0, N_TC):
        for td in range(tc + _BLOCK, 89):
            s = float(block_sum[td] - block_sum[tc])
            if s > best_spread:
                best_spread = s
                best_tc = tc
                best_td = td
    return best_tc, best_td, best_spread


def compute_oracle_slot_counts(
    labels: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    train_end: pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Count oracle (charge_start, discharge_start) frequencies over labelled days.

    Only days with all 96 slots and a positive oracle spread contribute to the
    counts (no-trade days carry no slot information).
    """
    df = labels[[time_col, target_col]].copy()
    if train_end is not None:
        df = df[df[time_col] <= train_end]
    df["date"] = df[time_col].dt.normalize()
    df["slot"] = df[time_col].dt.hour * 4 + df[time_col].dt.minute // 15

    counts_c = np.zeros(N_TC, dtype=float)
    counts_d = np.zeros(N_TD, dtype=float)
    for _, day in df.groupby("date", sort=False):
        if len(day) != 96:
            continue
        day = day.sort_values("slot")
        if not np.array_equal(day["slot"].to_numpy(), np.arange(96)):
            continue
        prices = day[target_col].to_numpy(dtype=float)
        tc, td, spread = _daily_oracle(prices)
        if spread <= 0:
            continue
        counts_c[tc] += 1.0
        counts_d[td - _BLOCK] += 1.0
    return counts_c, counts_d


def smoothed_log_prior(counts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Apply Laplace smoothing then take log to produce slot-level log-priors."""
    if alpha <= 0:
        raise ValueError("alpha must be positive for numerical stability")
    total = counts.sum() + alpha * len(counts)
    return np.log((counts + alpha) / total)


def build_dispatch_prior(
    labels: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    train_end: pd.Timestamp | None = None,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (log_prior_charge, log_prior_discharge) from training labels.

    Parameters
    ----------
    labels:
        Frame containing columns ``time_col`` and ``target_col``.
    train_end:
        If provided, only rows with ``time_col <= train_end`` are used.
    alpha:
        Laplace smoothing strength. Larger ``alpha`` flattens the prior.

    Returns
    -------
    (log_prior_charge, log_prior_discharge):
        ``log_prior_charge`` has length 81 (charge_start in [0, 80]).
        ``log_prior_discharge`` has length 81 (discharge_start in [BLOCK, 88]
        stored at index td - BLOCK).
    """
    counts_c, counts_d = compute_oracle_slot_counts(
        labels, time_col=time_col, target_col=target_col, train_end=train_end
    )
    return smoothed_log_prior(counts_c, alpha), smoothed_log_prior(counts_d, alpha)
