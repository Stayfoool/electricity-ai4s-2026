from __future__ import annotations

import pandas as pd
import pytest

from electricity.submit import _member_train_frame, _segment_boundaries, validate_submission


def _cfg() -> dict:
    return {
        "dispatch": {
            "points_per_day": 96,
            "block_size": 8,
            "charge_power": -1000,
            "discharge_power": 1000,
        }
    }


def _empty_submission() -> pd.DataFrame:
    times = pd.date_range("2026-01-01 00:00:00", periods=59 * 96, freq="15min")
    return pd.DataFrame(
        {
            "times": times.strftime("%Y-%m-%d %H:%M:%S"),
            "实时价格": 0.0,
            "power": 0.0,
        }
    )


def test_validate_submission_accepts_legal_no_trade_file() -> None:
    validate_submission(_empty_submission(), _cfg())


def test_validate_submission_rejects_partial_charge_block() -> None:
    df = _empty_submission()
    df.loc[0:6, "power"] = -1000.0

    with pytest.raises(ValueError, match="invalid charge block length"):
        validate_submission(df, _cfg())


def test_member_train_frame_uses_recent_window() -> None:
    df = pd.DataFrame({"times": pd.date_range("2025-01-01", periods=10, freq="D")})
    spec = {"model": {"train_window_days": 3}}

    out = _member_train_frame(df, spec, time_col="times")

    assert out["times"].min() == pd.Timestamp("2025-01-08")
    assert out["times"].max() == pd.Timestamp("2025-01-10")


def test_segment_boundaries_returns_none_for_plain_model() -> None:
    assert _segment_boundaries({"model": {}}) is None


def test_segment_boundaries_casts_values_to_int() -> None:
    spec = {"model": {"segment_boundaries": [0, "24", 96]}}
    assert _segment_boundaries(spec) == [0, 24, 96]
