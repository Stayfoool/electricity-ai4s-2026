from __future__ import annotations

import numpy as np


def daily_profit(true_prices: np.ndarray, power: np.ndarray) -> float:
    if len(true_prices) != 96 or len(power) != 96:
        raise ValueError("true_prices and power must both have 96 points")
    return float(np.sum(true_prices * power))

