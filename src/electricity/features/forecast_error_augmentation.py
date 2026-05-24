from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from electricity.features.time_features import add_bid_space_feature, add_derived_features

DEFAULT_AUGMENT_CHANNELS = ["风电", "光伏", "非市场化机组"]
FORECAST_SUFFIX = "预测值"
ACTUAL_SUFFIX = "实际值"


@dataclass(frozen=True)
class _Pool:
    by_key: dict[tuple[int, ...], np.ndarray]
    by_hour: dict[int, np.ndarray]
    global_indices: np.ndarray


def _forecast_col(channel: str) -> str:
    return f"{channel}{FORECAST_SUFFIX}"


def _actual_col(channel: str) -> str:
    return f"{channel}{ACTUAL_SUFFIX}"


def _bucket_columns(mode: str) -> list[str]:
    if mode == "month_hour":
        return ["__month", "__hour"]
    if mode == "month_slot":
        return ["__month", "__slot"]
    if mode == "hour":
        return ["__hour"]
    if mode == "global":
        return []
    raise ValueError(f"unsupported forecast_error_augmentation.bucket={mode!r}")


def _validate_channels(df: pd.DataFrame, channels: list[str]) -> None:
    missing: list[str] = []
    for channel in channels:
        for col in (_forecast_col(channel), _actual_col(channel)):
            if col not in df.columns:
                missing.append(col)
    if missing:
        raise ValueError(f"forecast error augmentation missing columns: {sorted(missing)}")


def _with_bucket_columns(df: pd.DataFrame, *, time_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = out[time_col]
    out["__month"] = ts.dt.month.astype(int)
    out["__hour"] = ts.dt.hour.astype(int)
    out["__slot"] = (ts.dt.hour * 4 + ts.dt.minute // 15).astype(int)
    return out


def _build_pool(work: pd.DataFrame, *, bucket_cols: list[str], valid_mask: pd.Series) -> _Pool:
    valid_indices = np.flatnonzero(valid_mask.to_numpy())
    by_key: dict[tuple[int, ...], np.ndarray] = {}
    if bucket_cols:
        grouped = work.loc[valid_mask].groupby(bucket_cols, sort=False, observed=True)
        for key, group in grouped:
            normalized_key = key if isinstance(key, tuple) else (key,)
            by_key[tuple(int(x) for x in normalized_key)] = group.index.to_numpy(dtype=int)

    by_hour = {
        int(hour): group.index.to_numpy(dtype=int)
        for hour, group in work.loc[valid_mask].groupby("__hour", sort=False, observed=True)
    }
    return _Pool(by_key=by_key, by_hour=by_hour, global_indices=valid_indices.astype(int))


def _sample_donor_indices(
    work: pd.DataFrame,
    *,
    pool: _Pool,
    bucket_cols: list[str],
    min_bucket_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(pool.global_indices) == 0:
        raise ValueError("forecast error augmentation has no valid residual rows to sample")

    sampled = np.empty(len(work), dtype=int)
    hours = work["__hour"].to_numpy(dtype=int)
    bucket_values = work[bucket_cols].to_numpy(dtype=int) if bucket_cols else None
    for pos in range(len(work)):
        candidates: np.ndarray | None = None
        if bucket_values is not None:
            key = tuple(int(x) for x in bucket_values[pos])
            bucket = pool.by_key.get(key)
            if bucket is not None and len(bucket) >= min_bucket_samples:
                candidates = bucket
        if candidates is None and bucket_cols:
            hour_bucket = pool.by_hour.get(int(hours[pos]))
            if hour_bucket is not None and len(hour_bucket) >= min_bucket_samples:
                candidates = hour_bucket
        if candidates is None:
            candidates = pool.global_indices
        sampled[pos] = int(rng.choice(candidates))
    return sampled


def _recompute_dependent_features(
    df: pd.DataFrame,
    *,
    time_col: str,
    recompute_renewable_total: bool,
) -> pd.DataFrame:
    out = df.copy()
    if recompute_renewable_total and {"风电预测值", "光伏预测值"} <= set(out.columns):
        out["风光总加预测值"] = out["风电预测值"] + out["光伏预测值"]
    if "bid_space" in out.columns:
        out = add_bid_space_feature(out)
    derived_cols = {
        "net_load",
        "renewable_ratio",
        "wind_ratio",
        "solar_ratio",
        "hydro_ratio",
        "tie_line_ratio",
        "non_market_ratio",
        "wind_solar_balance",
        "net_load_day_mean",
        "net_load_day_dev",
        "renewable_ratio_day_mean",
        "renewable_ratio_day_dev",
        "net_load_day_rank_pct",
        "renewable_ratio_day_rank_pct",
        "load_day_rank_pct",
        "solar_day_rank_pct",
        "wind_day_rank_pct",
    }
    if derived_cols & set(out.columns):
        out = add_derived_features(out, time_col=time_col)
    return out


def _scope_mask(work: pd.DataFrame, aug_cfg: dict) -> pd.Series:
    mask = pd.Series(True, index=work.index)
    months = aug_cfg.get("apply_months")
    if months:
        mask &= work["__month"].isin([int(x) for x in months])
    hours = aug_cfg.get("apply_hours")
    if hours:
        mask &= work["__hour"].isin([int(x) for x in hours])
    slots = aug_cfg.get("apply_slots")
    if slots:
        mask &= work["__slot"].isin([int(x) for x in slots])
    return mask


def apply_forecast_error_augmentation(
    train_df: pd.DataFrame,
    cfg: dict,
    *,
    time_col: str = "times",
) -> pd.DataFrame:
    """Create robust training rows by resampling historical forecast errors.

    For each copied training row, the actual boundary value is kept fixed while
    the forecast residual is replaced by a sampled historical residual. This
    simulates alternative official forecasts for the same realized system state,
    without touching validation rows or price labels.
    """
    aug_cfg = cfg.get("forecast_error_augmentation") or {}
    if not aug_cfg.get("enabled", False):
        return train_df

    copies = int(aug_cfg.get("copies", 1))
    if copies <= 0:
        return train_df
    shrink = float(aug_cfg.get("shrink", 0.5))
    if not 0.0 <= shrink <= 1.0:
        raise ValueError("forecast_error_augmentation.shrink must be in [0, 1]")

    channels = list(aug_cfg.get("channels", DEFAULT_AUGMENT_CHANNELS))
    _validate_channels(train_df, channels)

    bucket_mode = str(aug_cfg.get("bucket", "month_hour"))
    bucket_cols = _bucket_columns(bucket_mode)
    min_bucket_samples = int(aug_cfg.get("min_bucket_samples", 24))
    recompute_renewable_total = bool(aug_cfg.get("recompute_renewable_total", True))
    seed = int(aug_cfg.get("seed", 2026))
    augmented_weight_factor = float(aug_cfg.get("augmented_weight_factor", 1.0))
    if augmented_weight_factor < 0:
        raise ValueError("forecast_error_augmentation.augmented_weight_factor must be >= 0")

    work = _with_bucket_columns(train_df.reset_index(drop=True), time_col=time_col)
    residual_cols: list[str] = []
    for channel in channels:
        res_col = f"__resid_{channel}"
        residual_cols.append(res_col)
        work[res_col] = work[_forecast_col(channel)] - work[_actual_col(channel)]

    valid_mask = work[residual_cols].notna().all(axis=1)
    pool = _build_pool(work, bucket_cols=bucket_cols, valid_mask=valid_mask)
    residual_matrix = work[residual_cols].to_numpy(dtype=float)
    current_residuals = residual_matrix.copy()
    scope_mask = _scope_mask(work, aug_cfg)
    scope_indices = np.flatnonzero(scope_mask.to_numpy())
    if len(scope_indices) == 0:
        return train_df

    frames = [train_df.copy()]
    for copy_idx in range(copies):
        rng = np.random.default_rng(seed + copy_idx)
        donor_idx = _sample_donor_indices(
            work,
            pool=pool,
            bucket_cols=bucket_cols,
            min_bucket_samples=min_bucket_samples,
            rng=rng,
        )
        sampled_residuals = residual_matrix[donor_idx]
        augmented = train_df.copy()
        for col_idx, channel in enumerate(channels):
            forecast_col = _forecast_col(channel)
            actual_col = _actual_col(channel)
            mixed_residual = (
                (1.0 - shrink) * current_residuals[:, col_idx]
                + shrink * sampled_residuals[:, col_idx]
            )
            values = augmented[forecast_col].to_numpy(dtype=float)
            actual_values = augmented[actual_col].to_numpy(dtype=float)
            values[scope_indices] = actual_values[scope_indices] + mixed_residual[scope_indices]
            augmented[forecast_col] = values
        augmented = _recompute_dependent_features(
            augmented,
            time_col=time_col,
            recompute_renewable_total=recompute_renewable_total,
        )
        if augmented_weight_factor != 1.0:
            weight_col = cfg.get("model", {}).get("sample_weight_col")
            if weight_col and weight_col in augmented.columns:
                augmented[weight_col] = (
                    augmented[weight_col].to_numpy(dtype=float) * augmented_weight_factor
                )
        frames.append(augmented.iloc[scope_indices].copy())

    return pd.concat(frames, ignore_index=True)
