from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from electricity.features.bid_space import ACTUAL_COLS, PRED_COLS
from electricity.features.weather import CORE_WEATHER_FEATURES, merge_weather_renewable

WEATHER_CORRECTION_FEATURES = [
    "weather_renewable_error_pred",
    "weather_renewable_corrected",
    "weather_bid_space_corrected",
    "weather_bid_space_delta",
    "weather_correction_abs",
]

WEATHER_CORRECTION_MODEL_FEATURES = [
    "month",
    "hour",
    "renewable_forecast",
    "wind_forecast",
    "solar_forecast",
    "ghi_mean",
    "ghi_max",
    "tcc_mean",
    "tp_mean",
    "wind_speed_mean",
    "wind_speed_max",
    "u100_mean",
    "v100_mean",
    "t2m_mean",
    "sp_mean",
]

WEATHER_CORRECTION_PARAMS = {
    "objective": "huber",
    "metric": "l1",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 4,
    "force_col_wise": True,
    "deterministic": True,
    "seed": 2026,
    "verbosity": -1,
}


def weather_correction_feature_columns(cfg: dict) -> list[str]:
    if not cfg.get("feature_sets", {}).get("weather_correction", False):
        return []
    return list(WEATHER_CORRECTION_FEATURES)


def _train_weather_correction(train: pd.DataFrame) -> lgb.Booster:
    cutoff = train["times"].quantile(0.85)
    fit = train[train["times"] <= cutoff]
    valid = train[train["times"] > cutoff]
    if len(valid) < 200:
        fit = train
        valid = train
    train_set = lgb.Dataset(
        fit[WEATHER_CORRECTION_MODEL_FEATURES],
        label=fit["renewable_error"],
        feature_name=WEATHER_CORRECTION_MODEL_FEATURES,
    )
    valid_set = lgb.Dataset(
        valid[WEATHER_CORRECTION_MODEL_FEATURES],
        label=valid["renewable_error"],
        feature_name=WEATHER_CORRECTION_MODEL_FEATURES,
        reference=train_set,
    )
    return lgb.train(
        dict(WEATHER_CORRECTION_PARAMS),
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )


def _prepare_hourly_weather_frame(cfg: dict) -> pd.DataFrame:
    hourly = merge_weather_renewable(cfg).rename(columns={"target_hour": "times"})
    hourly["bid_space"] = (
        hourly[PRED_COLS["load"]]
        - hourly["renewable_forecast"]
        - hourly[PRED_COLS["tie_line"]]
        - hourly[PRED_COLS["hydro"]]
        - hourly[PRED_COLS["non_market"]]
    )
    hourly["bid_space_actual"] = (
        hourly[ACTUAL_COLS["load"]]
        - hourly["renewable_actual"]
        - hourly[ACTUAL_COLS["tie_line"]]
        - hourly[ACTUAL_COLS["hydro"]]
        - hourly[ACTUAL_COLS["non_market"]]
    )
    return hourly


def _predict_and_build_features(
    model: lgb.Booster,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    best_iteration = (
        model.best_iteration if model.best_iteration and model.best_iteration > 0 else None
    )
    out = hourly[["times", "renewable_forecast", "bid_space", *CORE_WEATHER_FEATURES]].copy()
    pred_error = np.asarray(
        model.predict(hourly[WEATHER_CORRECTION_MODEL_FEATURES], num_iteration=best_iteration)
    )
    out["weather_renewable_error_pred"] = pred_error
    out["weather_renewable_corrected"] = out["renewable_forecast"] - pred_error
    out["weather_bid_space_corrected"] = out["bid_space"] + pred_error
    out["weather_bid_space_delta"] = pred_error
    out["weather_correction_abs"] = np.abs(pred_error)
    return out[["times", *WEATHER_CORRECTION_FEATURES]]


def apply_fold_weather_correction(
    df: pd.DataFrame,
    cfg: dict,
    *,
    time_col: str,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    if not cfg.get("feature_sets", {}).get("weather_correction", False):
        return df

    hourly = _prepare_hourly_weather_frame(cfg)
    train_hourly = hourly[hourly["times"] <= train_end.floor("h")].copy()
    if train_hourly.empty:
        raise ValueError("empty train split for weather correction")
    model = _train_weather_correction(train_hourly)
    hourly_features = _predict_and_build_features(model, hourly)
    out = df.copy()
    out["__weather_hour"] = out[time_col].dt.floor("h")
    out = out.merge(hourly_features, left_on="__weather_hour", right_on="times", how="left")
    out = out.drop(columns=["__weather_hour", "times_y"]).rename(columns={"times_x": time_col})
    missing = out[WEATHER_CORRECTION_FEATURES].isna().any(axis=1)
    if missing.any():
        # The first training day can be missing because the previous-day NWP file is absent.
        for col in WEATHER_CORRECTION_FEATURES:
            out[col] = out[col].fillna(0.0)
    return out
