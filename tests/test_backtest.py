from __future__ import annotations

import pandas as pd
import pytest

from electricity.eval.backtest import _fold_train_frame, _maybe_build_prior, _target_col_for_mode
from electricity.models import validate_model_features


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


def test_validate_model_features_rejects_label_or_actual_columns() -> None:
    with pytest.raises(ValueError, match="leakage"):
        validate_model_features(["系统负荷预测值", "A"], target_col="A")
    with pytest.raises(ValueError, match="leakage"):
        validate_model_features(["风光总加实际值"], target_col="A")


def test_dispatch_prior_excludes_same_season_validation_interval() -> None:
    times = pd.date_range("2025-01-01", "2025-03-31 23:45:00", freq="15min")
    labels = pd.DataFrame({"times": times, "A": 0.0})
    labels["slot"] = labels["times"].dt.hour * 4 + labels["times"].dt.minute // 15
    labels["date"] = labels["times"].dt.normalize()

    # Jan validation days have oracle charge at slot 0 and discharge at slot 8.
    jan = labels["times"].dt.month == 1
    labels.loc[jan & labels["slot"].between(0, 7), "A"] = -10.0
    labels.loc[jan & labels["slot"].between(8, 15), "A"] = 10.0

    # Mar training days have oracle charge at slot 40 and discharge at slot 48.
    mar = labels["times"].dt.month == 3
    labels.loc[mar & labels["slot"].between(40, 47), "A"] = -10.0
    labels.loc[mar & labels["slot"].between(48, 55), "A"] = 10.0

    log_pc, _, _, _ = _maybe_build_prior(
        {"dispatch": {"prior": {"enabled": True, "alpha": 0.01}}},
        labels,
        time_col="times",
        target_col="A",
        train_end=pd.Timestamp("2025-03-31 23:59:59"),
        train_start=pd.Timestamp("2025-03-01 00:00:00"),
        valid_start=pd.Timestamp("2025-01-01 00:00:00"),
        valid_end=pd.Timestamp("2025-01-31 23:59:59"),
    )

    assert log_pc is not None
    assert int(log_pc.argmax()) == 40
