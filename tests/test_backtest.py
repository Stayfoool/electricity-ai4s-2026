from __future__ import annotations

import pandas as pd

from electricity.eval.backtest import _fold_train_frame, _target_col_for_mode


def test_fold_train_frame_uses_last_n_days() -> None:
    df = pd.DataFrame({"times": pd.date_range("2025-01-01", periods=10, freq="D")})

    out = _fold_train_frame(
        df,
        time_col="times",
        train_end=pd.Timestamp("2025-01-10 23:59:59"),
        train_window_days=3,
    )

    assert out["times"].min() == pd.Timestamp("2025-01-08")
    assert out["times"].max() == pd.Timestamp("2025-01-10")


def test_fold_train_frame_uses_all_past_when_window_is_none() -> None:
    df = pd.DataFrame({"times": pd.date_range("2025-01-01", periods=10, freq="D")})

    out = _fold_train_frame(
        df,
        time_col="times",
        train_end=pd.Timestamp("2025-01-10 23:59:59"),
        train_window_days=None,
    )

    assert len(out) == 10


def test_target_col_for_mode() -> None:
    assert _target_col_for_mode("absolute") == "target_absolute"
