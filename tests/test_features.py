from __future__ import annotations

import pandas as pd
import pytest

from electricity.features import (
    NWP_CORE_FEATURES,
    NWP_FEATURES,
    NWP_INTERACTION_FEATURES,
    add_derived_features,
    build_feature_frame,
    feature_columns,
)


def test_add_time_features() -> None:
    df = pd.DataFrame({"times": pd.to_datetime(["2025-01-01 00:15:00"])})
    out = build_feature_frame(df, {"data": {"feature_cols": []}}, time_col="times")

    assert out.loc[0, "hour"] == 0
    assert out.loc[0, "minute"] == 15
    assert out.loc[0, "quarter"] == 1
    assert "quarter_sin" in out.columns


def test_add_derived_features() -> None:
    df = pd.DataFrame(
        {
            "times": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:15:00"]),
            "系统负荷预测值": [4.0, 8.0],
            "风光总加预测值": [1.0, 4.0],
            "联络线预测值": [0.4, 0.8],
            "风电预测值": [0.8, 3.0],
            "光伏预测值": [0.2, 1.0],
            "水电预测值": [0.5, 1.0],
            "非市场化机组预测值": [1.0, 2.0],
        }
    )

    out = add_derived_features(df)

    assert out.loc[0, "net_load"] == 2.5
    assert out.loc[1, "renewable_ratio"] == 0.5
    assert "net_load_day_rank_pct" in out.columns


def test_feature_columns_include_derived_only_when_enabled() -> None:
    cfg = {"data": {"feature_cols": ["系统负荷预测值"]}, "feature_sets": {"derived": False}}
    assert "net_load" not in feature_columns(cfg)

    cfg["feature_sets"]["derived"] = True
    assert "net_load" in feature_columns(cfg)


def test_feature_columns_support_feature_families() -> None:
    cfg = {
        "data": {"feature_cols": ["系统负荷预测值"]},
        "feature_sets": {"business": True, "rank": False, "deviation": False},
    }
    columns = feature_columns(cfg)
    assert "net_load" in columns
    assert "net_load_day_rank_pct" not in columns

    cfg["feature_sets"] = {"business": False, "rank": True, "deviation": False}
    columns = feature_columns(cfg)
    assert "net_load" not in columns
    assert "net_load_day_rank_pct" in columns


def test_build_feature_frame_can_merge_nwp_features(tmp_path) -> None:
    nwp_path = tmp_path / "nwp.csv"
    nwp = pd.DataFrame(
        {
            "times": ["2025-01-02 00:00:00"],
            **{feature: 1.0 for feature in NWP_FEATURES},
        }
    )
    nwp.to_csv(nwp_path, index=False)
    df = pd.DataFrame({"times": pd.to_datetime(["2025-01-02 00:00:00"])})
    cfg = {
        "data": {"feature_cols": []},
        "feature_sets": {"nwp": True},
        "nwp": {"feature_csv": str(nwp_path)},
    }

    out = build_feature_frame(df, cfg, time_col="times")

    assert out.loc[0, "nwp_ghi_mean"] == 1.0
    assert "nwp_wind_speed100_max" in feature_columns(cfg)


def test_build_feature_frame_supports_core_nwp_and_interactions(tmp_path) -> None:
    nwp_path = tmp_path / "nwp.csv"
    nwp = pd.DataFrame(
        {
            "times": ["2025-01-02 00:00:00"],
            **{feature: 2.0 for feature in NWP_FEATURES},
        }
    )
    nwp.to_csv(nwp_path, index=False)
    df = pd.DataFrame(
        {
            "times": pd.to_datetime(["2025-01-02 00:00:00"]),
            "风电预测值": [3.0],
            "光伏预测值": [5.0],
            "风光总加预测值": [8.0],
        }
    )
    cfg = {
        "data": {"feature_cols": []},
        "feature_sets": {"nwp": True, "nwp_interactions": True},
        "nwp": {"feature_csv": str(nwp_path), "mode": "core"},
    }

    out = build_feature_frame(df, cfg, time_col="times")
    columns = feature_columns(cfg)

    assert "nwp_sp_mean" not in columns
    assert set(NWP_CORE_FEATURES) <= set(columns)
    assert set(NWP_INTERACTION_FEATURES) <= set(columns)
    assert out.loc[0, "nwp_ghi_x_solar"] == 10.0
    assert out.loc[0, "nwp_wind_speed_x_wind"] == 6.0


def test_build_feature_frame_requires_nwp_path() -> None:
    df = pd.DataFrame({"times": pd.to_datetime(["2025-01-02 00:00:00"])})
    cfg = {"data": {"feature_cols": []}, "feature_sets": {"nwp": True}}

    with pytest.raises(ValueError, match="nwp.feature_csv"):
        build_feature_frame(df, cfg, time_col="times")
