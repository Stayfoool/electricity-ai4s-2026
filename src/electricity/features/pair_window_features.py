from __future__ import annotations

import numpy as np
import pandas as pd

from electricity.features.window_features import (
    WINDOW_TARGET_COL,
    window_aggregate_feature_columns,
)

PAIR_TARGET_COL = "pair_target_spread"

PAIR_TIME_FEATURES = [
    "charge_start",
    "discharge_start",
    "charge_end",
    "discharge_end",
    "charge_center",
    "discharge_center",
    "gap_slots",
    "gap_hours",
    "charge_start_hour",
    "discharge_start_hour",
    "pair_dayofweek",
    "pair_month",
    "pair_is_weekend",
    "charge_start_sin",
    "charge_start_cos",
    "discharge_start_sin",
    "discharge_start_cos",
    "gap_sin",
    "gap_cos",
]


def _selected_window_columns(
    base_feature_cols: list[str],
    *,
    agg_stats: list[str] | None = None,
) -> list[str]:
    if agg_stats is None:
        return window_aggregate_feature_columns(base_feature_cols)
    unsupported = sorted(set(agg_stats) - {"mean", "std", "min", "max"})
    if unsupported:
        raise ValueError(f"unsupported pair window aggregate stats: {unsupported}")
    return [f"win_{col}_{stat}" for col in base_feature_cols for stat in agg_stats]


def pair_window_feature_columns(
    base_feature_cols: list[str],
    *,
    agg_stats: list[str] | None = None,
) -> list[str]:
    window_cols = _selected_window_columns(base_feature_cols, agg_stats=agg_stats)
    return (
        [f"charge_{col}" for col in window_cols]
        + [f"discharge_{col}" for col in window_cols]
        + [f"diff_{col}" for col in window_cols]
        + PAIR_TIME_FEATURES
    )


def legal_pair_count(*, block_size: int = 8, points_per_day: int = 96) -> int:
    max_window_start = points_per_day - block_size
    max_charge_start = points_per_day - 2 * block_size
    total = 0
    for charge_start in range(max_charge_start + 1):
        total += max_window_start - (charge_start + block_size) + 1
    return total


def _legal_pair_starts(
    *,
    block_size: int,
    points_per_day: int,
) -> tuple[np.ndarray, np.ndarray]:
    charge_starts: list[int] = []
    discharge_starts: list[int] = []
    max_window_start = points_per_day - block_size
    max_charge_start = points_per_day - 2 * block_size
    for charge_start in range(max_charge_start + 1):
        for discharge_start in range(charge_start + block_size, max_window_start + 1):
            charge_starts.append(charge_start)
            discharge_starts.append(discharge_start)
    return (
        np.asarray(charge_starts, dtype=np.int16),
        np.asarray(discharge_starts, dtype=np.int16),
    )


def build_pair_window_frame(
    window_df: pd.DataFrame,
    *,
    base_feature_cols: list[str],
    agg_stats: list[str] | None = None,
    block_size: int = 8,
    points_per_day: int = 96,
) -> pd.DataFrame:
    window_cols = _selected_window_columns(base_feature_cols, agg_stats=agg_stats)
    required_cols = {"date", "window_start", *window_cols}
    missing = required_cols - set(window_df.columns)
    if missing:
        raise ValueError(f"window_df missing columns: {sorted(missing)}")

    expected_windows = points_per_day - block_size + 1
    has_target = WINDOW_TARGET_COL in window_df.columns
    charge_starts, discharge_starts = _legal_pair_starts(
        block_size=block_size,
        points_per_day=points_per_day,
    )
    pair_count = len(charge_starts)
    gap_slots = discharge_starts - charge_starts
    charge_centers = charge_starts + (block_size - 1) / 2
    discharge_centers = discharge_starts + (block_size - 1) / 2
    day_frames: list[pd.DataFrame] = []

    for date, group in window_df.groupby("date", sort=True):
        if len(group) != expected_windows:
            continue
        group = group.sort_values("window_start").reset_index(drop=True)
        starts = group["window_start"].to_numpy(dtype=int)
        if not np.array_equal(starts, np.arange(expected_windows)):
            continue

        feature_values = group[window_cols].to_numpy(dtype=np.float32)
        target_values = (
            None if not has_target else group[WINDOW_TARGET_COL].to_numpy(dtype=np.float32)
        )
        dayofweek = int(pd.Timestamp(date).dayofweek)
        month = int(pd.Timestamp(date).month)
        is_weekend = int(dayofweek >= 5)

        charge_features = feature_values[charge_starts]
        discharge_features = feature_values[discharge_starts]
        data: dict[str, object] = {
            "date": np.repeat(np.datetime64(pd.Timestamp(date)), pair_count),
            "charge_start": charge_starts,
            "discharge_start": discharge_starts,
            "charge_end": (charge_starts + block_size - 1).astype(np.int16),
            "discharge_end": (discharge_starts + block_size - 1).astype(np.int16),
            "charge_center": charge_centers.astype(np.float32),
            "discharge_center": discharge_centers.astype(np.float32),
            "gap_slots": gap_slots.astype(np.int16),
            "gap_hours": (gap_slots / 4).astype(np.float32),
            "charge_start_hour": (charge_starts / 4).astype(np.float32),
            "discharge_start_hour": (discharge_starts / 4).astype(np.float32),
            "pair_dayofweek": np.full(pair_count, dayofweek, dtype=np.int8),
            "pair_month": np.full(pair_count, month, dtype=np.int8),
            "pair_is_weekend": np.full(pair_count, is_weekend, dtype=np.int8),
            "charge_start_sin": np.sin(2 * np.pi * charge_starts / points_per_day).astype(
                np.float32
            ),
            "charge_start_cos": np.cos(2 * np.pi * charge_starts / points_per_day).astype(
                np.float32
            ),
            "discharge_start_sin": np.sin(
                2 * np.pi * discharge_starts / points_per_day
            ).astype(np.float32),
            "discharge_start_cos": np.cos(
                2 * np.pi * discharge_starts / points_per_day
            ).astype(np.float32),
            "gap_sin": np.sin(2 * np.pi * gap_slots / points_per_day).astype(np.float32),
            "gap_cos": np.cos(2 * np.pi * gap_slots / points_per_day).astype(np.float32),
        }
        for idx, col in enumerate(window_cols):
            charge_value = charge_features[:, idx]
            discharge_value = discharge_features[:, idx]
            data[f"charge_{col}"] = charge_value
            data[f"discharge_{col}"] = discharge_value
            data[f"diff_{col}"] = discharge_value - charge_value
        if target_values is not None:
            data[PAIR_TARGET_COL] = (
                (target_values[discharge_starts] - target_values[charge_starts]) * block_size
            ).astype(np.float32)
        day_frames.append(pd.DataFrame(data))

    if not day_frames:
        return pd.DataFrame()
    return pd.concat(day_frames, ignore_index=True)
