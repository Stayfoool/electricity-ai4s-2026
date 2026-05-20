from __future__ import annotations

import numpy as np
import pandas as pd

TIME_FEATURES = [
    "hour",
    "minute",
    "quarter",
    "dayofweek",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "quarter_sin",
    "quarter_cos",
]

BUSINESS_FEATURES = [
    "net_load",
    "renewable_ratio",
    "wind_ratio",
    "solar_ratio",
    "hydro_ratio",
    "tie_line_ratio",
    "non_market_ratio",
    "wind_solar_balance",
]

DEVIATION_FEATURES = [
    "net_load_day_mean",
    "net_load_day_dev",
    "renewable_ratio_day_mean",
    "renewable_ratio_day_dev",
]

RANK_FEATURES = [
    "net_load_day_rank_pct",
    "renewable_ratio_day_rank_pct",
    "load_day_rank_pct",
    "solar_day_rank_pct",
    "wind_day_rank_pct",
]

BID_SPACE_FEATURES = ["bid_space"]

CAPACITY_FEATURES = [
    "wind_utilization",
    "solar_utilization",
    "renewable_utilization",
    "bid_space_local_dev",
]

HOLIDAY_FEATURES = [
    "is_spring_festival",
    "days_to_spring_festival",
    "is_pre_spring_window",
    "is_post_spring_window",
]

DERIVED_FEATURES = BUSINESS_FEATURES + DEVIATION_FEATURES + RANK_FEATURES

# Spring Festival official holiday windows (除夕 to last day of holiday).
# Used to compute holiday features that respect the lunar calendar shift.
SPRING_FESTIVAL_WINDOWS: list[tuple[str, str]] = [
    # 2024: Feb 9 (除夕) - Feb 17
    ("2024-02-09", "2024-02-17"),
    # 2025: Jan 28 (除夕) - Feb 4
    ("2025-01-28", "2025-02-04"),
    # 2026: Feb 17 (除夕) - Feb 23
    ("2026-02-17", "2026-02-23"),
    # 2027: Feb 6 (除夕) - Feb 12 (anchor for future years)
    ("2027-02-06", "2027-02-12"),
]

NWP_FEATURES = [
    "nwp_ghi_mean",
    "nwp_ghi_std",
    "nwp_ghi_min",
    "nwp_ghi_max",
    "nwp_sp_mean",
    "nwp_sp_std",
    "nwp_sp_min",
    "nwp_sp_max",
    "nwp_t2m_mean",
    "nwp_t2m_std",
    "nwp_t2m_min",
    "nwp_t2m_max",
    "nwp_tcc_mean",
    "nwp_tcc_std",
    "nwp_tcc_min",
    "nwp_tcc_max",
    "nwp_tp_mean",
    "nwp_tp_std",
    "nwp_tp_min",
    "nwp_tp_max",
    "nwp_u100_mean",
    "nwp_u100_std",
    "nwp_u100_min",
    "nwp_u100_max",
    "nwp_v100_mean",
    "nwp_v100_std",
    "nwp_v100_min",
    "nwp_v100_max",
    "nwp_wind_speed100_mean",
    "nwp_wind_speed100_std",
    "nwp_wind_speed100_min",
    "nwp_wind_speed100_max",
]

NWP_CORE_FEATURES = [
    "nwp_ghi_mean",
    "nwp_ghi_std",
    "nwp_ghi_max",
    "nwp_tcc_mean",
    "nwp_tcc_std",
    "nwp_tcc_max",
    "nwp_wind_speed100_mean",
    "nwp_wind_speed100_std",
    "nwp_wind_speed100_max",
]

NWP_INTERACTION_FEATURES = [
    "nwp_ghi_x_solar",
    "nwp_tcc_x_solar",
    "nwp_wind_speed_x_wind",
    "nwp_ghi_x_wind_solar",
    "nwp_wind_speed_x_wind_solar",
]

FEATURE_GROUPS = {
    "business_features": BUSINESS_FEATURES,
    "deviation_features": DEVIATION_FEATURES,
    "rank_features": RANK_FEATURES,
}


def _selected_group_features(
    feature_sets: dict,
    *,
    list_key: str,
    full_key: str,
    allowed: list[str],
) -> list[str]:
    if feature_sets.get("derived", False):
        return list(allowed)
    if feature_sets.get(full_key, False):
        return list(allowed)

    explicit = feature_sets.get(list_key, [])
    if not explicit:
        return []

    explicit_list = list(explicit)
    invalid = sorted(set(explicit_list) - set(allowed))
    if invalid:
        raise ValueError(f"unsupported {list_key}: {invalid}")
    return [name for name in allowed if name in explicit_list]


def selected_derived_features(feature_sets: dict) -> list[str]:
    columns: list[str] = []
    columns += _selected_group_features(
        feature_sets,
        list_key="business_features",
        full_key="business",
        allowed=BUSINESS_FEATURES,
    )
    columns += _selected_group_features(
        feature_sets,
        list_key="deviation_features",
        full_key="deviation",
        allowed=DEVIATION_FEATURES,
    )
    columns += _selected_group_features(
        feature_sets,
        list_key="rank_features",
        full_key="rank",
        allowed=RANK_FEATURES,
    )
    return columns


def add_time_features(df: pd.DataFrame, *, time_col: str = "times") -> pd.DataFrame:
    out = df.copy()
    ts = out[time_col]
    quarter = ts.dt.hour * 4 + ts.dt.minute // 15
    out["hour"] = ts.dt.hour
    out["minute"] = ts.dt.minute
    out["quarter"] = quarter
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["quarter_sin"] = np.sin(2 * np.pi * quarter / 96)
    out["quarter_cos"] = np.cos(2 * np.pi * quarter / 96)
    return out


def add_derived_features(df: pd.DataFrame, *, time_col: str = "times") -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6
    load = out["系统负荷预测值"].clip(lower=eps)
    renewable = out["风光总加预测值"]
    wind = out["风电预测值"]
    solar = out["光伏预测值"]
    hydro = out["水电预测值"]

    out["net_load"] = out["系统负荷预测值"] - renewable - hydro
    out["renewable_ratio"] = renewable / load
    out["wind_ratio"] = wind / load
    out["solar_ratio"] = solar / load
    out["hydro_ratio"] = hydro / load
    out["tie_line_ratio"] = out["联络线预测值"] / load
    out["non_market_ratio"] = out["非市场化机组预测值"] / load
    out["wind_solar_balance"] = wind - solar

    date = out[time_col].dt.normalize()
    out["net_load_day_mean"] = out.groupby(date)["net_load"].transform("mean")
    out["net_load_day_dev"] = out["net_load"] - out["net_load_day_mean"]
    out["net_load_day_rank_pct"] = out.groupby(date)["net_load"].rank(pct=True)
    out["renewable_ratio_day_mean"] = out.groupby(date)["renewable_ratio"].transform("mean")
    out["renewable_ratio_day_dev"] = out["renewable_ratio"] - out["renewable_ratio_day_mean"]
    out["renewable_ratio_day_rank_pct"] = out.groupby(date)["renewable_ratio"].rank(pct=True)
    out["load_day_rank_pct"] = out.groupby(date)["系统负荷预测值"].rank(pct=True)
    out["solar_day_rank_pct"] = out.groupby(date)["光伏预测值"].rank(pct=True)
    out["wind_day_rank_pct"] = out.groupby(date)["风电预测值"].rank(pct=True)
    return out


def add_nwp_features(df: pd.DataFrame, cfg: dict, *, time_col: str = "times") -> pd.DataFrame:
    path = cfg.get("nwp", {}).get("feature_csv")
    if not path:
        raise ValueError("feature_sets.nwp=true requires nwp.feature_csv")

    nwp_columns = nwp_feature_columns(cfg)
    raw_nwp_columns = sorted(set(nwp_columns) & set(NWP_FEATURES))
    nwp = pd.read_csv(path, parse_dates=[time_col])
    missing = sorted(set(raw_nwp_columns) - set(nwp.columns))
    if missing:
        raise ValueError(f"NWP feature file is missing columns: {missing}")
    out = df.merge(nwp[[time_col, *raw_nwp_columns]], on=time_col, how="left")
    if cfg.get("feature_sets", {}).get("nwp_interactions", False):
        out = add_nwp_interaction_features(out)
    return out


def add_nwp_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["nwp_ghi_x_solar"] = out["nwp_ghi_mean"] * out["光伏预测值"]
    out["nwp_tcc_x_solar"] = out["nwp_tcc_mean"] * out["光伏预测值"]
    out["nwp_wind_speed_x_wind"] = out["nwp_wind_speed100_mean"] * out["风电预测值"]
    out["nwp_ghi_x_wind_solar"] = out["nwp_ghi_mean"] * out["风光总加预测值"]
    out["nwp_wind_speed_x_wind_solar"] = (
        out["nwp_wind_speed100_mean"] * out["风光总加预测值"]
    )
    return out


def nwp_feature_columns(cfg: dict) -> list[str]:
    feature_sets = cfg.get("feature_sets", {})
    if not feature_sets.get("nwp", False):
        return []

    mode = cfg.get("nwp", {}).get("mode", "full")
    if mode == "full":
        columns = list(NWP_FEATURES)
    elif mode == "core":
        columns = list(NWP_CORE_FEATURES)
    else:
        raise ValueError(f"unsupported nwp.mode={mode}")

    if feature_sets.get("nwp_interactions", False):
        columns += NWP_INTERACTION_FEATURES
    return columns


def add_bid_space_feature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bid_space"] = (
        out["系统负荷预测值"]
        - out["联络线预测值"]
        - out["非市场化机组预测值"]
        - out["风光总加预测值"]
        - out["水电预测值"]
    )
    return out


def add_capacity_features(
    df: pd.DataFrame,
    *,
    time_col: str = "times",
    window_days: int = 30,
) -> pd.DataFrame:
    """Add capacity-normalized utilization features.

    Rationale: train data is 2025, test is 2026-01/02 where wind+solar
    installed capacity is roughly +30%. Raw forecast values shift up by
    that amount; the model overestimates renewable scarcity. Normalizing
    by a rolling-window capacity proxy makes the features invariant to
    the absolute capacity level.

    Computation must be done on a frame that includes BOTH train and test
    sorted by ``time_col`` so the rolling window can look backward from
    test rows into the training tail.
    """
    out = df.sort_values(time_col).reset_index(drop=True).copy()
    n = len(out)
    if n == 0:
        return out

    # 96 slots per day; window in slots.
    window_slots = int(window_days * 96)
    min_window = min(int(7 * 96), window_slots)

    # Capacity proxy = rolling p99 (close to max but robust to outliers).
    def _rolling_q99(series: pd.Series) -> pd.Series:
        return series.rolling(window=window_slots, min_periods=min_window).quantile(0.99)

    wind_cap = _rolling_q99(out["风电预测值"])
    solar_cap = _rolling_q99(out["光伏预测值"])
    total_cap = _rolling_q99(out["风光总加预测值"])

    # For early rows (before min_window), fall back to expanding p99 from the start
    # to avoid NaNs that LightGBM may treat oddly.
    wind_exp = out["风电预测值"].expanding(min_periods=1).quantile(0.99)
    solar_exp = out["光伏预测值"].expanding(min_periods=1).quantile(0.99)
    total_exp = out["风光总加预测值"].expanding(min_periods=1).quantile(0.99)
    wind_cap = wind_cap.fillna(wind_exp)
    solar_cap = solar_cap.fillna(solar_exp)
    total_cap = total_cap.fillna(total_exp)

    eps = 1e-3
    out["wind_utilization"] = out["风电预测值"] / wind_cap.clip(lower=eps)
    out["solar_utilization"] = out["光伏预测值"] / solar_cap.clip(lower=eps)
    out["renewable_utilization"] = out["风光总加预测值"] / total_cap.clip(lower=eps)

    # bid_space local deviation: bid_space minus its rolling mean over the same
    # window. Captures how the current row compares to the *recent* baseline,
    # which is invariant to long-run capacity shifts.
    if "bid_space" not in out.columns:
        out = add_bid_space_feature(out)
    bid_mean = (
        out["bid_space"]
        .rolling(window=window_slots, min_periods=min_window)
        .mean()
    )
    bid_mean = bid_mean.fillna(out["bid_space"].expanding(min_periods=1).mean())
    out["bid_space_local_dev"] = out["bid_space"] - bid_mean

    return out


def _nearest_spring_festival_eve(timestamps: pd.Series) -> pd.Series:
    """Return the nearest Spring Festival 除夕 date for each timestamp."""
    eves = pd.to_datetime([start for start, _ in SPRING_FESTIVAL_WINDOWS])
    eve_values = eves.to_numpy()
    ts = timestamps.dt.normalize().to_numpy()
    # For each timestamp pick eve minimizing |ts - eve|.
    diffs = np.abs(ts[:, None] - eve_values[None, :])
    idx = diffs.argmin(axis=1)
    return pd.Series(eve_values[idx], index=timestamps.index)


def add_holiday_features(
    df: pd.DataFrame,
    *,
    time_col: str = "times",
    pre_post_window_days: int = 3,
) -> pd.DataFrame:
    """Add Spring Festival aware holiday features.

    The Lunar New Year date shifts each year (2025 Jan 28, 2026 Feb 17, etc.),
    so a model that only sees Gregorian time features cannot generalize from
    training 2025 Spring Festival to predicting 2026 Spring Festival. These
    features encode festival proximity in a year-invariant way.
    """
    out = df.copy()
    ts = out[time_col]
    eves = _nearest_spring_festival_eve(ts)
    day_diff = (ts.dt.normalize() - eves).dt.days
    out["days_to_spring_festival"] = day_diff.astype(int)

    in_window = pd.Series(False, index=out.index)
    for start, end in SPRING_FESTIVAL_WINDOWS:
        mask = (ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end) + pd.Timedelta(days=1))
        in_window = in_window | mask
    out["is_spring_festival"] = in_window.astype(int)

    pre_window = (day_diff >= -pre_post_window_days) & (day_diff < 0)
    out["is_pre_spring_window"] = pre_window.astype(int)

    # Find the holiday-end day for each row's nearest eve to define post-window.
    eve_to_end = {pd.Timestamp(s): pd.Timestamp(e) for s, e in SPRING_FESTIVAL_WINDOWS}
    ends = eves.map(eve_to_end)
    end_diff = (ts.dt.normalize() - ends).dt.days
    post_window = (end_diff > 0) & (end_diff <= pre_post_window_days)
    out["is_post_spring_window"] = post_window.astype(int)

    return out


def build_feature_frame(df: pd.DataFrame, cfg: dict, *, time_col: str = "times") -> pd.DataFrame:
    out = add_time_features(df, time_col=time_col)
    feature_sets = cfg.get("feature_sets", {})
    if feature_sets.get("bid_space", False):
        out = add_bid_space_feature(out)
    if selected_derived_features(feature_sets):
        out = add_derived_features(out, time_col=time_col)
    if feature_sets.get("capacity", False):
        out = add_capacity_features(out, time_col=time_col)
    if feature_sets.get("holiday", False):
        out = add_holiday_features(out, time_col=time_col)
    if feature_sets.get("nwp", False):
        out = add_nwp_features(out, cfg, time_col=time_col)
    return out


def feature_columns(cfg: dict) -> list[str]:
    columns = list(cfg["data"]["feature_cols"]) + TIME_FEATURES
    feature_sets = cfg.get("feature_sets", {})
    if feature_sets.get("bid_space", False):
        columns += BID_SPACE_FEATURES
    columns += selected_derived_features(feature_sets)
    if feature_sets.get("capacity", False):
        columns += CAPACITY_FEATURES
    if feature_sets.get("holiday", False):
        columns += HOLIDAY_FEATURES
    if feature_sets.get("nwp", False):
        columns += nwp_feature_columns(cfg)
    if feature_sets.get("bias_correction", False):
        # Imported lazily to avoid cycles.
        from electricity.features.bias_correction import bias_correction_feature_columns

        columns += bias_correction_feature_columns(cfg)
    return columns
