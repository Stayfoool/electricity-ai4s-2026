from __future__ import annotations

import numpy as np
import pandas as pd

from electricity.features.forecast_error_augmentation import apply_forecast_error_augmentation


def _frame() -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=48, freq="h")
    actual_wind = np.linspace(1.0, 2.0, len(times))
    actual_solar = np.linspace(0.0, 1.0, len(times))
    actual_non_market = np.linspace(3.0, 4.0, len(times))
    wind_resid = np.tile([0.1, -0.2, 0.3, -0.4], 12)
    solar_resid = np.tile([-0.1, 0.2, -0.3, 0.4], 12)
    non_market_resid = np.tile([0.05, -0.05], 24)
    return pd.DataFrame(
        {
            "times": times,
            "风电实际值": actual_wind,
            "风电预测值": actual_wind + wind_resid,
            "光伏实际值": actual_solar,
            "光伏预测值": actual_solar + solar_resid,
            "风光总加实际值": actual_wind + actual_solar,
            "风光总加预测值": actual_wind + actual_solar + wind_resid + solar_resid,
            "非市场化机组实际值": actual_non_market,
            "非市场化机组预测值": actual_non_market + non_market_resid,
            "A": np.arange(len(times), dtype=float),
        }
    )


def test_forecast_error_augmentation_noop_when_disabled() -> None:
    df = _frame()

    out = apply_forecast_error_augmentation(df, {}, time_col="times")

    pd.testing.assert_frame_equal(out, df)


def test_forecast_error_augmentation_keeps_original_rows_and_recomputes_renewable() -> None:
    df = _frame()
    cfg = {
        "forecast_error_augmentation": {
            "enabled": True,
            "copies": 1,
            "shrink": 1.0,
            "bucket": "global",
            "min_bucket_samples": 1,
            "seed": 7,
            "channels": ["风电", "光伏", "非市场化机组"],
        }
    }

    out = apply_forecast_error_augmentation(df, cfg, time_col="times")

    assert len(out) == 2 * len(df)
    pd.testing.assert_frame_equal(out.iloc[: len(df)].reset_index(drop=True), df)
    aug = out.iloc[len(df) :].reset_index(drop=True)
    assert np.allclose(aug["风光总加预测值"], aug["风电预测值"] + aug["光伏预测值"])
    assert np.allclose(aug["A"], df["A"])
    assert not np.allclose(aug["风电预测值"], df["风电预测值"])


def test_forecast_error_augmentation_can_scope_hours_and_downweight_augmented_rows() -> None:
    df = _frame()
    df["sample_weight"] = 2.0
    cfg = {
        "model": {"sample_weight_col": "sample_weight"},
        "forecast_error_augmentation": {
            "enabled": True,
            "copies": 1,
            "shrink": 1.0,
            "bucket": "global",
            "min_bucket_samples": 1,
            "seed": 8,
            "channels": ["风电", "光伏"],
            "apply_hours": [0, 1],
            "augmented_weight_factor": 0.25,
        },
    }

    out = apply_forecast_error_augmentation(df, cfg, time_col="times")

    expected_extra = int(df["times"].dt.hour.isin([0, 1]).sum())
    assert len(out) == len(df) + expected_extra
    aug = out.iloc[len(df) :].reset_index(drop=True)
    assert set(aug["times"].dt.hour.unique()) <= {0, 1}
    assert np.allclose(aug["sample_weight"], 0.5)
