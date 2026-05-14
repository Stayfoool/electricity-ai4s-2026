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

BID_SPACE_FEATURES = [
    "bid_space",
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

DERIVED_FEATURES = BUSINESS_FEATURES + DEVIATION_FEATURES + RANK_FEATURES

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
    "bid_space_features": BID_SPACE_FEATURES,
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


def selected_bid_space_features(feature_sets: dict) -> list[str]:
    if not feature_sets.get("bid_space", False) and not feature_sets.get(
        "bid_space_features"
    ):
        return []

    explicit = feature_sets.get("bid_space_features", [])
    if feature_sets.get("bid_space", False):
        return list(BID_SPACE_FEATURES)
    if isinstance(explicit, bool):
        return list(BID_SPACE_FEATURES) if explicit else []

    explicit_list = list(explicit)
    invalid = sorted(set(explicit_list) - set(BID_SPACE_FEATURES))
    if invalid:
        raise ValueError(f"unsupported bid_space_features: {invalid}")
    return [name for name in BID_SPACE_FEATURES if name in explicit_list]


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

    out["bid_space"] = (
        out["系统负荷预测值"]
        - renewable
        - out["联络线预测值"]
        - hydro
        - out["非市场化机组预测值"]
    )
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


def build_feature_frame(df: pd.DataFrame, cfg: dict, *, time_col: str = "times") -> pd.DataFrame:
    out = add_time_features(df, time_col=time_col)
    feature_sets = cfg.get("feature_sets", {})
    if selected_derived_features(feature_sets):
        out = add_derived_features(out, time_col=time_col)
    if feature_sets.get("nwp", False):
        out = add_nwp_features(out, cfg, time_col=time_col)
    return out


def feature_columns(cfg: dict) -> list[str]:
    columns = list(cfg["data"]["feature_cols"]) + TIME_FEATURES
    feature_sets = cfg.get("feature_sets", {})
    columns += selected_bid_space_features(feature_sets)
    columns += selected_derived_features(feature_sets)
    if feature_sets.get("nwp", False):
        columns += nwp_feature_columns(cfg)
    return columns
