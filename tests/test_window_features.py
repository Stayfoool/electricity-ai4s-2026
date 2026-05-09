from __future__ import annotations

import numpy as np
import pandas as pd

from electricity.eval.window_backtest import optimize_window_day
from electricity.features.window_features import build_window_frame, window_feature_columns


def test_build_window_frame_creates_89_windows_for_complete_day() -> None:
    df = pd.DataFrame(
        {
            "times": pd.date_range("2025-01-01", periods=96, freq="15min"),
            "feat": np.arange(96, dtype=float),
            "target": np.arange(96, dtype=float),
        }
    )

    out = build_window_frame(
        df,
        time_col="times",
        base_feature_cols=["feat"],
        target_col="target",
    )

    assert len(out) == 89
    assert out.loc[0, "window_start"] == 0
    assert out.loc[0, "win_feat_mean"] == 3.5
    assert out.loc[0, "window_target_mean"] == 3.5
    assert out.loc[88, "window_start"] == 88
    assert "win_feat_max" in window_feature_columns(["feat"])


def test_optimize_window_day_matches_low_then_high_windows() -> None:
    window_means = np.ones(89)
    window_means[10] = -1.0
    window_means[30] = 2.0

    result = optimize_window_day(window_means)

    assert result.charge_start == 10
    assert result.discharge_start == 30
    assert result.predicted_spread == 24.0
    assert (result.power[10:18] == -1000).all()
    assert (result.power[30:38] == 1000).all()
