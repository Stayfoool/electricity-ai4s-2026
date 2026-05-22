from __future__ import annotations

import numpy as np
import pandas as pd

from electricity.eval.pair_window_backtest import _pair_point_feature_cols, optimize_pair_window_day
from electricity.features.pair_window_features import (
    PAIR_TARGET_COL,
    build_pair_window_frame,
    legal_pair_count,
    pair_window_feature_columns,
)
from electricity.features.window_features import build_window_frame


def test_legal_pair_count_matches_96_point_storage_constraint() -> None:
    assert legal_pair_count(block_size=8, points_per_day=96) == 3321


def test_build_pair_window_frame_creates_spread_target() -> None:
    df = pd.DataFrame(
        {
            "times": pd.date_range("2025-01-01", periods=96, freq="15min"),
            "feat": np.arange(96, dtype=float),
            "target": np.arange(96, dtype=float),
        }
    )
    windows = build_window_frame(
        df,
        time_col="times",
        base_feature_cols=["feat"],
        target_col="target",
    )

    out = build_pair_window_frame(windows, base_feature_cols=["feat"])

    assert len(out) == 3321
    first = out.iloc[0]
    assert first["charge_start"] == 0
    assert first["discharge_start"] == 8
    assert first[PAIR_TARGET_COL] == 64.0
    assert first["diff_win_feat_mean"] == 8.0
    assert "charge_win_feat_mean" in pair_window_feature_columns(["feat"])
    assert "discharge_win_feat_mean" in pair_window_feature_columns(["feat"])
    assert "diff_win_feat_mean" in pair_window_feature_columns(["feat"])


def test_optimize_pair_window_day_chooses_max_predicted_spread() -> None:
    pair_df = pd.DataFrame(
        {
            "charge_start": [0, 10],
            "discharge_start": [8, 30],
            "pred_pair_spread": [1.0, 5.0],
        }
    )

    result = optimize_pair_window_day(pair_df)

    assert result.charge_start == 10
    assert result.discharge_start == 30
    assert result.predicted_spread == 5.0
    assert (result.power[10:18] == -1000).all()
    assert (result.power[30:38] == 1000).all()


def test_pair_point_feature_cols_can_select_subset() -> None:
    cfg = {"model": {"pair_point_feature_cols": ["bid_space", "net_load"]}}

    assert _pair_point_feature_cols(cfg, ["hour", "bid_space", "net_load"]) == [
        "bid_space",
        "net_load",
    ]
