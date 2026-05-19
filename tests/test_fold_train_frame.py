from __future__ import annotations

import pandas as pd

from electricity.eval.backtest import _fold_train_frame


def _frame(start: str, end: str) -> pd.DataFrame:
    times = pd.date_range(start, end, freq="15min")
    return pd.DataFrame({"times": times, "value": range(len(times))})


def test_fold_train_frame_default_forward_time() -> None:
    """Existing behavior: rows up to train_end (inclusive) when no extras."""
    df = _frame("2025-01-01", "2025-12-31 23:45")
    out = _fold_train_frame(
        df,
        time_col="times",
        train_end=pd.Timestamp("2025-08-31 23:59:59"),
        train_window_days=None,
    )
    assert out["times"].max() <= pd.Timestamp("2025-08-31 23:59:59")
    assert out["times"].min() == pd.Timestamp("2025-01-01")


def test_fold_train_frame_with_train_window_days() -> None:
    """train_window_days clips lower bound."""
    df = _frame("2025-01-01", "2025-12-31 23:45")
    out = _fold_train_frame(
        df,
        time_col="times",
        train_end=pd.Timestamp("2025-09-30 23:59:59"),
        train_window_days=180,
    )
    assert out["times"].max() <= pd.Timestamp("2025-09-30 23:59:59")
    assert out["times"].min() >= pd.Timestamp("2025-04-01")


def test_fold_train_frame_same_season_excludes_validation() -> None:
    """Jan-Feb 2025 fold: train spans Mar-Dec 2025, excludes Jan-Feb 2025."""
    df = _frame("2025-01-01", "2025-12-31 23:45")
    out = _fold_train_frame(
        df,
        time_col="times",
        train_end=pd.Timestamp("2025-12-31 23:59:59"),
        train_window_days=None,
        train_start=pd.Timestamp("2025-03-01"),
        valid_start=pd.Timestamp("2025-01-01"),
        valid_end=pd.Timestamp("2025-02-28 23:59:59"),
    )
    assert out["times"].min() >= pd.Timestamp("2025-03-01")
    assert out["times"].max() <= pd.Timestamp("2025-12-31 23:59:59")
    # Validation interval must not appear in training.
    in_valid = (out["times"] >= pd.Timestamp("2025-01-01")) & (
        out["times"] <= pd.Timestamp("2025-02-28 23:59:59")
    )
    assert not in_valid.any()


def test_fold_train_frame_excludes_valid_inside_train_window() -> None:
    """When train_window_days makes the window wrap around a validation window,
    the validation rows are still excluded."""
    df = _frame("2025-01-01", "2025-12-31 23:45")
    out = _fold_train_frame(
        df,
        time_col="times",
        train_end=pd.Timestamp("2025-12-31 23:59:59"),
        train_window_days=300,  # would include Jan-Feb
        valid_start=pd.Timestamp("2025-01-15"),
        valid_end=pd.Timestamp("2025-01-31 23:59:59"),
    )
    in_valid = (out["times"] >= pd.Timestamp("2025-01-15")) & (
        out["times"] <= pd.Timestamp("2025-01-31 23:59:59")
    )
    assert not in_valid.any()
