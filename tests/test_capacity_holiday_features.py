from __future__ import annotations

import numpy as np
import pandas as pd

from electricity.features import (
    CAPACITY_FEATURES,
    HOLIDAY_FEATURES,
    add_capacity_features,
    add_holiday_features,
    build_feature_frame,
    feature_columns,
)


def _synthetic_frame(start: str, days: int) -> pd.DataFrame:
    times = pd.date_range(start, periods=days * 96, freq="15min")
    rng = np.random.default_rng(42)
    n = len(times)
    return pd.DataFrame(
        {
            "times": times,
            "系统负荷预测值": 3.0 + 0.1 * rng.standard_normal(n),
            "风光总加预测值": 1.0 + 0.5 * rng.standard_normal(n),
            "联络线预测值": 0.15 + 0.02 * rng.standard_normal(n),
            "风电预测值": 0.7 + 0.3 * rng.standard_normal(n),
            "光伏预测值": 0.3 + 0.4 * rng.standard_normal(n),
            "水电预测值": 0.04 + 0.01 * rng.standard_normal(n),
            "非市场化机组预测值": 0.7 + 0.05 * rng.standard_normal(n),
        }
    )


def test_capacity_features_invariant_to_uniform_scaling() -> None:
    df = _synthetic_frame("2025-01-01", days=60)
    out_a = add_capacity_features(df, time_col="times")

    df2 = df.copy()
    for col in ["风电预测值", "光伏预测值", "风光总加预测值"]:
        df2[col] = df[col] * 1.30  # +30% capacity expansion
    out_b = add_capacity_features(df2, time_col="times")

    # Utilizations should be approximately equal after both arms reach
    # the same capacity-proxy regime (skip the warmup region).
    warmup = 96 * 30
    for col in ["wind_utilization", "solar_utilization", "renewable_utilization"]:
        a = out_a[col].iloc[warmup:].to_numpy()
        b = out_b[col].iloc[warmup:].to_numpy()
        assert np.allclose(a, b, atol=1e-6), (
            f"{col} not invariant to uniform scaling: max diff = {np.abs(a - b).max()}"
        )


def test_holiday_features_lunar_calendar() -> None:
    df = pd.DataFrame(
        {
            "times": pd.to_datetime(
                [
                    "2025-01-28 12:00",  # 2025 除夕 (中除夕)
                    "2025-02-04 12:00",  # 2025 假期最后一天
                    "2026-02-17 12:00",  # 2026 除夕
                    "2026-02-23 12:00",  # 2026 假期最后一天
                    "2025-06-15 12:00",  # 普通日
                    "2026-01-01 12:00",  # 元旦 (非春节)
                    "2026-02-14 12:00",  # 春节前 3 天
                    "2026-02-26 12:00",  # 春节后 3 天
                ]
            )
        }
    )
    out = add_holiday_features(df, time_col="times")

    assert out.loc[0, "is_spring_festival"] == 1
    assert out.loc[1, "is_spring_festival"] == 1
    assert out.loc[2, "is_spring_festival"] == 1
    assert out.loc[3, "is_spring_festival"] == 1
    assert out.loc[4, "is_spring_festival"] == 0
    assert out.loc[5, "is_spring_festival"] == 0

    # Days_to_spring_festival measures distance to nearest 除夕.
    assert out.loc[0, "days_to_spring_festival"] == 0
    assert out.loc[2, "days_to_spring_festival"] == 0
    assert out.loc[6, "is_pre_spring_window"] == 1
    assert out.loc[7, "is_post_spring_window"] == 1
    assert out.loc[6, "is_spring_festival"] == 0


def test_build_feature_frame_includes_capacity_and_holiday_columns() -> None:
    df = _synthetic_frame("2025-01-01", days=10)
    cfg = {
        "feature_sets": {"capacity": True, "holiday": True, "bid_space": True},
        "data": {"feature_cols": []},
    }
    out = build_feature_frame(df, cfg, time_col="times")
    for col in CAPACITY_FEATURES + HOLIDAY_FEATURES:
        assert col in out.columns, f"missing {col}"

    fc = feature_columns(cfg)
    for col in CAPACITY_FEATURES + HOLIDAY_FEATURES:
        assert col in fc, f"feature_columns missing {col}"


def test_capacity_features_have_no_nans() -> None:
    df = _synthetic_frame("2025-01-01", days=60)
    out = add_capacity_features(df, time_col="times")
    for col in CAPACITY_FEATURES:
        assert not out[col].isna().any(), f"{col} has NaN values"
