from __future__ import annotations

import pandas as pd
import pytest

from electricity.eval.segmented_utils import (
    segment_labels,
    segmented_feature_build_spec,
    segmented_feature_columns,
    slot_segment,
)


def test_slot_segment_uses_expected_six_segments() -> None:
    boundaries = [0, 24, 40, 56, 72, 88, 96]
    slots = pd.Series([0, 23, 24, 39, 40, 55, 56, 71, 72, 87, 88, 95])

    out = slot_segment(slots, boundaries).to_list()

    assert out == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert segment_labels(boundaries) == [
        "00_24",
        "24_40",
        "40_56",
        "56_72",
        "72_88",
        "88_96",
    ]


def test_slot_segment_requires_valid_boundaries() -> None:
    with pytest.raises(ValueError, match="start at 0 and end at 96"):
        slot_segment(pd.Series([0]), [1, 96])
    with pytest.raises(ValueError, match="strictly increasing"):
        slot_segment(pd.Series([0]), [0, 24, 24, 96])


def test_segmented_feature_columns_supports_per_segment_overrides() -> None:
    spec = {
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
        "feature_sets": {"derived": False},
        "model": {
            "segment_boundaries": [0, 24, 40, 56, 72, 88, 96],
            "segment_feature_sets": {
                "40_56": {"business": True},
                "72_88": {"business": True},
            },
        },
    }

    build_spec = segmented_feature_build_spec(spec)
    cols_by_segment = segmented_feature_columns(spec, [0, 24, 40, 56, 72, 88, 96])

    assert build_spec["feature_sets"]["business"] is True
    assert "net_load" not in cols_by_segment[0]
    assert "net_load" in cols_by_segment[2]
    assert "net_load" in cols_by_segment[4]


def test_segmented_feature_columns_supports_explicit_business_subsets() -> None:
    spec = {
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
        "feature_sets": {"business_features": ["net_load"]},
        "model": {
            "segment_boundaries": [0, 24, 40, 56, 72, 88, 96],
            "segment_feature_sets": {
                "40_56": {"business_features": ["net_load", "renewable_ratio"]},
                "72_88": {"business_features": ["net_load", "non_market_ratio"]},
            },
        },
    }

    build_spec = segmented_feature_build_spec(spec)
    cols_by_segment = segmented_feature_columns(spec, [0, 24, 40, 56, 72, 88, 96])

    assert build_spec["feature_sets"]["business_features"] == [
        "net_load",
        "renewable_ratio",
        "non_market_ratio",
    ]
    assert "net_load" in cols_by_segment[0]
    assert "renewable_ratio" not in cols_by_segment[0]
    assert "renewable_ratio" in cols_by_segment[2]
    assert "non_market_ratio" in cols_by_segment[4]
    assert "tie_line_ratio" not in cols_by_segment[4]
