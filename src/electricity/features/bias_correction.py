"""Forecast bias correction via (hour, month) lookup tables.

Motivation
----------
EDA on 2025 boundary data shows that ``forecast - actual`` for renewable and
non-market channels has *strongly time-of-day-dependent* bias (e.g. solar at
noon over-predicts by +0.4, at night ~0). This non-uniform offset distorts the
intraday rank of net load forecasts, which directly causes the model to pick
wrong charge / discharge slots. Within-day rank correlation between forecast
and actual net load is only spearman ~0.92 (median) and dispatch slot
agreement is ~50%.

A constant per-day offset (``D``) does *not* affect dispatch (rank invariant),
so we explicitly strip it out before fitting the lookup table. The resulting
``bias_table`` captures only the dispatch-relevant *time-of-day* component
(``W``).

Notes
-----
* The table is fitted strictly on rows whose ``time_col`` is <= ``train_end``.
* Channels with low bias (系统负荷, 联络线, 水电) are not corrected.
* ``shrink`` lets callers downscale the correction (0.0 = no correction,
  1.0 = full correction). Useful when the bias regime in the test period is
  uncertain (e.g. forecast system upgrade between 2025 train and 2026 test).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Channels with EDA bias > 0.07 (in p.u.) — worth correcting.
CORRECTABLE_CHANNELS: tuple[str, ...] = (
    "风光总加",
    "风电",
    "光伏",
    "非市场化机组",
)

# Feature columns the bias correction module exposes downstream.
BIAS_CORRECTED_FORECAST_COLUMNS: tuple[str, ...] = tuple(
    f"{c}预测值_debiased" for c in CORRECTABLE_CHANNELS
)
NET_LOAD_FEATURES: tuple[str, ...] = (
    "net_load_fct_explicit",
    "net_load_fct_debiased",
)

ALL_BIAS_FEATURES: tuple[str, ...] = (
    *BIAS_CORRECTED_FORECAST_COLUMNS,
    *NET_LOAD_FEATURES,
)


def fit_bias_table(
    df: pd.DataFrame,
    *,
    channels: Sequence[str] = CORRECTABLE_CHANNELS,
    time_col: str = "times",
    train_end: pd.Timestamp | str | None = None,
    remove_daily_mean: bool = True,
    min_samples_per_bucket: int = 5,
) -> pd.DataFrame:
    """Fit a (channel, hour, month) bias lookup table.

    Parameters
    ----------
    df
        Frame containing ``<channel>实际值`` and ``<channel>预测值`` columns.
    channels
        Channels to compute bias for.
    time_col
        Timestamp column.
    train_end
        Inclusive cutoff. Rows strictly after ``train_end`` are excluded so the
        fit is leak-free for backtest folds. ``None`` uses all rows.
    remove_daily_mean
        If True, subtract the per-day mean of the residual before grouping by
        (hour, month). This isolates the within-day, dispatch-relevant
        component ``W`` from the dispatch-neutral daily offset ``D``.
    min_samples_per_bucket
        Buckets with fewer samples fall back to the channel's monthly mean to
        avoid noisy edge buckets.

    Returns
    -------
    DataFrame with columns ``[channel, hour, month, bias, n]`` (long form).
    """
    if train_end is not None:
        cutoff = pd.Timestamp(train_end)
        work = df[df[time_col] <= cutoff].copy()
    else:
        work = df.copy()

    if work.empty:
        return pd.DataFrame(columns=["channel", "hour", "month", "bias", "n"])

    times = work[time_col]
    work["__hour"] = times.dt.hour
    work["__month"] = times.dt.month
    work["__date"] = times.dt.normalize()

    out_rows = []
    for channel in channels:
        actual_col = f"{channel}实际值"
        forecast_col = f"{channel}预测值"
        if actual_col not in work.columns or forecast_col not in work.columns:
            raise KeyError(
                f"missing columns for channel '{channel}': "
                f"need both {actual_col!r} and {forecast_col!r}"
            )
        err = work[forecast_col] - work[actual_col]
        if remove_daily_mean:
            daily_mean = err.groupby(work["__date"]).transform("mean")
            err = err - daily_mean

        grouped = err.groupby([work["__hour"], work["__month"]]).agg(["mean", "count"])
        grouped = grouped.reset_index().rename(
            columns={"__hour": "hour", "__month": "month", "mean": "bias", "count": "n"}
        )
        grouped["channel"] = channel

        # Fallback for sparse buckets: use channel's per-month mean.
        month_mean = err.groupby(work["__month"]).mean()
        weak = grouped["n"] < min_samples_per_bucket
        if weak.any():
            grouped.loc[weak, "bias"] = grouped.loc[weak, "month"].map(month_mean).fillna(0.0)

        out_rows.append(grouped[["channel", "hour", "month", "bias", "n"]])

    return pd.concat(out_rows, ignore_index=True)


def apply_bias_correction(
    df: pd.DataFrame,
    bias_table: pd.DataFrame,
    *,
    time_col: str = "times",
    shrink: float = 1.0,
    add_net_load: bool = True,
    mode: str = "augment",
) -> pd.DataFrame:
    """Add ``<channel>预测值_debiased`` columns and optional net-load features.

    Parameters
    ----------
    df
        Frame with ``<channel>预测值`` columns. Actual columns are NOT required
        (works on test data).
    bias_table
        Output of :func:`fit_bias_table`.
    time_col
        Timestamp column.
    shrink
        Scale factor on the correction. 0.0 disables correction (debiased ==
        forecast). 1.0 applies the full lookup. Values in (0, 1) hedge against
        bias regime drift between train and test.
    add_net_load
        If True, also add explicit ``net_load_fct_explicit`` (with 联络线
        subtracted, matching the EDA definition) and ``net_load_fct_debiased``.

    Returns
    -------
    Copy of ``df`` with new columns appended.
    """
    out = df.copy()
    times = out[time_col]
    out["__hour"] = times.dt.hour
    out["__month"] = times.dt.month

    if mode not in ("augment", "replace"):
        raise ValueError(f"mode must be 'augment' or 'replace', got {mode!r}")

    debiased_cols: dict[str, np.ndarray] = {}
    for channel in bias_table["channel"].unique():
        sub = (
            bias_table[bias_table["channel"] == channel][["hour", "month", "bias"]]
            .rename(columns={"hour": "__hour", "month": "__month"})
        )
        merged = out.merge(sub, on=["__hour", "__month"], how="left")
        forecast_values = merged[f"{channel}预测值"].to_numpy()
        bias_values = merged["bias"].fillna(0.0).to_numpy()
        debiased = forecast_values - shrink * bias_values
        debiased_cols[channel] = debiased
        if mode == "augment":
            out[f"{channel}预测值_debiased"] = debiased
        else:  # replace
            out[f"{channel}预测值"] = debiased

    if add_net_load:
        # net_load = 系统负荷 - 风光 - 水电 - 非市场化 - 联络线 (EDA-correct definition)
        load = out["系统负荷预测值"].to_numpy()
        ren_raw = (
            df["风光总加预测值"].to_numpy() if mode == "replace" else out["风光总加预测值"].to_numpy()
        )
        non_mkt_raw = (
            df["非市场化机组预测值"].to_numpy()
            if mode == "replace"
            else out["非市场化机组预测值"].to_numpy()
        )
        hydro = out["水电预测值"].to_numpy()
        tie = out["联络线预测值"].to_numpy()

        ren_db = debiased_cols.get("风光总加", ren_raw)
        non_mkt_db = debiased_cols.get("非市场化机组", non_mkt_raw)

        if mode == "augment":
            out["net_load_fct_explicit"] = load - ren_raw - hydro - non_mkt_raw - tie
            out["net_load_fct_debiased"] = load - ren_db - hydro - non_mkt_db - tie
        else:  # replace: only one consolidated column
            out["net_load_fct"] = load - ren_db - hydro - non_mkt_db - tie

    return out.drop(columns=["__hour", "__month"])


def add_bias_correction_features(
    df: pd.DataFrame,
    cfg: dict,
    *,
    time_col: str = "times",
    train_end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Convenience wrapper used by feature pipelines.

    Reads ``cfg['feature_sets']['bias_correction']`` and the optional
    ``cfg['bias_correction']`` block::

        bias_correction:
          channels: [风光总加, 风电, 光伏, 非市场化机组]   # default = CORRECTABLE_CHANNELS
          shrink: 1.0                                       # 0.0..1.0
          remove_daily_mean: true
          min_samples_per_bucket: 5
          add_net_load: true

    The bias table is fitted on rows ``<= train_end`` (if provided) to avoid
    leakage; otherwise on all rows in ``df``.
    """
    feature_sets = cfg.get("feature_sets", {})
    if not feature_sets.get("bias_correction", False):
        return df

    bc_cfg = cfg.get("bias_correction", {}) or {}
    channels = tuple(bc_cfg.get("channels", CORRECTABLE_CHANNELS))
    shrink = float(bc_cfg.get("shrink", 1.0))
    remove_daily_mean = bool(bc_cfg.get("remove_daily_mean", True))
    min_samples = int(bc_cfg.get("min_samples_per_bucket", 5))
    add_net_load = bool(bc_cfg.get("add_net_load", True))
    mode = str(bc_cfg.get("mode", "augment"))

    table = fit_bias_table(
        df,
        channels=channels,
        time_col=time_col,
        train_end=train_end,
        remove_daily_mean=remove_daily_mean,
        min_samples_per_bucket=min_samples,
    )
    return apply_bias_correction(
        df,
        table,
        time_col=time_col,
        shrink=shrink,
        add_net_load=add_net_load,
        mode=mode,
    )


def apply_fold_bias_correction(
    frame: pd.DataFrame,
    spec: dict,
    *,
    time_col: str = "times",
    train_end: pd.Timestamp | str | None,
) -> pd.DataFrame:
    """Fold-aware wrapper. Drops any pre-existing debiased columns then
    fits a fresh bias table on rows ``<= train_end`` and applies it.

    No-op (returns ``frame`` unchanged) when
    ``spec['feature_sets']['bias_correction']`` is falsy.
    """
    feature_sets = spec.get("feature_sets", {}) or {}
    if not feature_sets.get("bias_correction", False):
        return frame

    drop_cols = [c for c in BIAS_CORRECTED_FORECAST_COLUMNS if c in frame.columns]
    drop_cols += [c for c in NET_LOAD_FEATURES if c in frame.columns]
    if "net_load_fct" in frame.columns:
        drop_cols.append("net_load_fct")
    work = frame.drop(columns=drop_cols) if drop_cols else frame.copy()
    return add_bias_correction_features(work, spec, time_col=time_col, train_end=train_end)


def bias_correction_feature_columns(cfg: dict) -> list[str]:
    """Return columns added by bias correction, conditional on cfg flags."""
    feature_sets = cfg.get("feature_sets", {})
    if not feature_sets.get("bias_correction", False):
        return []

    bc_cfg = cfg.get("bias_correction", {}) or {}
    channels: Iterable[str] = bc_cfg.get("channels", CORRECTABLE_CHANNELS)
    add_net_load = bool(bc_cfg.get("add_net_load", True))
    mode = str(bc_cfg.get("mode", "augment"))

    if mode == "replace":
        # Replace mode keeps original forecast names (already in feature_cols);
        # only the net_load column is new.
        return ["net_load_fct"] if add_net_load else []

    columns = [f"{c}预测值_debiased" for c in channels]
    if add_net_load:
        columns += list(NET_LOAD_FEATURES)
    return columns
