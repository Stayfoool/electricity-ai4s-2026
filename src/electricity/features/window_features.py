from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_TARGET_COL = "window_target_mean"

WINDOW_TIME_FEATURES = [
    "window_start",
    "window_end",
    "window_center",
    "window_start_hour",
    "window_center_hour",
    "window_dayofweek",
    "window_month",
    "window_is_weekend",
    "window_start_sin",
    "window_start_cos",
    "window_center_sin",
    "window_center_cos",
]

WINDOW_AGG_STATS = ["mean", "std", "min", "max"]


def window_aggregate_feature_columns(base_feature_cols: list[str]) -> list[str]:
    return [f"win_{col}_{stat}" for col in base_feature_cols for stat in WINDOW_AGG_STATS]


def window_feature_columns(base_feature_cols: list[str]) -> list[str]:
    return window_aggregate_feature_columns(base_feature_cols) + WINDOW_TIME_FEATURES


def build_window_frame(
    df: pd.DataFrame,
    *,
    time_col: str,
    base_feature_cols: list[str],
    target_col: str | None = None,
    block_size: int = 8,
    points_per_day: int = 96,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    work = df[[time_col, *base_feature_cols, *([target_col] if target_col else [])]].copy()
    work["date"] = work[time_col].dt.normalize()
    work["slot"] = work[time_col].dt.hour * 4 + work[time_col].dt.minute // 15

    for date, group in work.groupby("date", sort=True):
        if len(group) != points_per_day:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(points_per_day)):
            continue

        feature_values = group[base_feature_cols].to_numpy(dtype=float)
        target_values = None if target_col is None else group[target_col].to_numpy(dtype=float)
        for start in range(points_per_day - block_size + 1):
            end = start + block_size - 1
            center = start + (block_size - 1) / 2
            values = feature_values[start : start + block_size]
            row: dict[str, object] = {
                "date": date,
                "window_start": start,
                "window_end": end,
                "window_center": center,
                "window_start_hour": start / 4,
                "window_center_hour": center / 4,
                "window_dayofweek": date.dayofweek,
                "window_month": date.month,
                "window_is_weekend": int(date.dayofweek >= 5),
                "window_start_sin": np.sin(2 * np.pi * start / points_per_day),
                "window_start_cos": np.cos(2 * np.pi * start / points_per_day),
                "window_center_sin": np.sin(2 * np.pi * center / points_per_day),
                "window_center_cos": np.cos(2 * np.pi * center / points_per_day),
            }
            for col_idx, col in enumerate(base_feature_cols):
                series = values[:, col_idx]
                row[f"win_{col}_mean"] = float(np.mean(series))
                row[f"win_{col}_std"] = float(np.std(series))
                row[f"win_{col}_min"] = float(np.min(series))
                row[f"win_{col}_max"] = float(np.max(series))
            if target_values is not None:
                row[WINDOW_TARGET_COL] = float(np.mean(target_values[start : start + block_size]))
            rows.append(row)
    return pd.DataFrame(rows)
