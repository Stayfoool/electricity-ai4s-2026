from __future__ import annotations

import pytest

from electricity.features.time_features import feature_columns


def _base_cfg(feature_sets: dict) -> dict:
    return {
        "data": {
            "feature_cols": [
                "系统负荷预测值",
                "风光总加预测值",
                "联络线预测值",
                "风电预测值",
                "光伏预测值",
                "水电预测值",
                "非市场化机组预测值",
            ]
        },
        "feature_sets": feature_sets,
    }


def test_feature_columns_supports_explicit_business_subset() -> None:
    cols = feature_columns(_base_cfg({"business_features": ["net_load", "non_market_ratio"]}))

    assert "net_load" in cols
    assert "non_market_ratio" in cols
    assert "renewable_ratio" not in cols
    assert "tie_line_ratio" not in cols


def test_feature_columns_supports_full_group_override() -> None:
    cols = feature_columns(_base_cfg({"business": True}))

    assert "net_load" in cols
    assert "renewable_ratio" in cols
    assert "tie_line_ratio" in cols


def test_feature_columns_rejects_unknown_explicit_feature() -> None:
    with pytest.raises(ValueError, match="unsupported business_features"):
        feature_columns(_base_cfg({"business_features": ["bad_feature_name"]}))
