from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from electricity.features import feature_columns
from electricity.models import best_iteration, predict_model, train_model


def slot_segment(slot: pd.Series, boundaries: list[int]) -> pd.Series:
    if boundaries[0] != 0 or boundaries[-1] != 96:
        raise ValueError("segment boundaries must start at 0 and end at 96")
    if sorted(boundaries) != boundaries or len(set(boundaries)) != len(boundaries):
        raise ValueError("segment boundaries must be strictly increasing")
    values = np.searchsorted(boundaries[1:], slot.to_numpy(dtype=int), side="right")
    return pd.Series(values, index=slot.index, dtype="int16")


def segment_labels(boundaries: list[int]) -> list[str]:
    return [
        f"{left:02d}_{right:02d}"
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def segment_boundaries(spec: dict, *, default: list[int] | None = None) -> list[int]:
    values = spec["model"].get("segment_boundaries", default or [0, 24, 40, 56, 72, 88, 96])
    return [int(x) for x in values]


def segment_feature_overrides(spec: dict) -> dict[str, dict]:
    return dict(spec["model"].get("segment_feature_sets", {}))


def segmented_feature_build_spec(spec: dict) -> dict:
    out = deepcopy(spec)
    union_sets = dict(out.get("feature_sets", {}))
    for override in segment_feature_overrides(spec).values():
        for key, value in override.items():
            if not value:
                continue
            if isinstance(value, bool):
                union_sets[key] = True
                continue
            if isinstance(value, list):
                base = union_sets.get(key, [])
                if isinstance(base, bool):
                    union_sets[key] = True if base else list(value)
                else:
                    merged = list(dict.fromkeys([*list(base), *list(value)]))
                    union_sets[key] = merged
                continue
            union_sets[key] = value
    out["feature_sets"] = union_sets
    return out


def segmented_feature_columns(
    spec: dict,
    boundaries: list[int],
) -> dict[int, list[str]]:
    labels = segment_labels(boundaries)
    overrides = segment_feature_overrides(spec)
    feature_cols_by_segment: dict[int, list[str]] = {}
    for segment_id, label in enumerate(labels):
        segment_spec = deepcopy(spec)
        feature_sets = dict(segment_spec.get("feature_sets", {}))
        feature_sets.update(overrides.get(label, {}))
        segment_spec["feature_sets"] = feature_sets
        feature_cols_by_segment[segment_id] = feature_columns(segment_spec)
    return feature_cols_by_segment


def add_slot_and_segment(
    df: pd.DataFrame,
    *,
    time_col: str,
    boundaries: list[int],
) -> pd.DataFrame:
    out = df.copy()
    out["slot"] = out[time_col].dt.hour * 4 + out[time_col].dt.minute // 15
    out["segment"] = slot_segment(out["slot"], boundaries)
    return out


def predict_segmented_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    feature_cols_by_segment: dict[int, list[str]],
    target_col: str,
    cfg: dict,
    boundaries: list[int],
) -> tuple[np.ndarray, list[int], list[int]]:
    preds = np.full(len(valid_df), np.nan, dtype=float)
    best_iterations: list[int] = []
    train_rows_by_segment: list[int] = []

    for segment_id, label in enumerate(segment_labels(boundaries)):
        train_seg = train_df[train_df["segment"] == segment_id].copy()
        valid_seg = valid_df[valid_df["segment"] == segment_id].copy()
        if train_seg.empty or valid_seg.empty:
            raise ValueError(f"empty train/valid segment={label}")
        feature_cols = feature_cols_by_segment[segment_id]

        model = train_model(
            train_seg,
            valid_seg,
            feature_cols=feature_cols,
            target_col=target_col,
            cfg=cfg,
        )
        preds[valid_seg.index.to_numpy()] = predict_model(model, valid_seg, feature_cols, cfg)
        best_iterations.append(best_iteration(model, cfg))
        train_rows_by_segment.append(len(train_seg))

    if np.isnan(preds).any():
        missing = int(np.isnan(preds).sum())
        raise ValueError(f"segmented prediction left {missing} rows without predictions")
    return preds, best_iterations, train_rows_by_segment
