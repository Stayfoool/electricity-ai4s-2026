from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from electricity.dispatch import optimize_day
from electricity.eval.metrics import daily_profit
from electricity.features.bid_space import ACTUAL_COLS, PRED_COLS, markdown_table
from electricity.features.weather import merge_weather_renewable

CONFIG_PATH = Path("configs/base.yaml")
REPORTS_DIR = Path("reports")
OUT_MD = REPORTS_DIR / "weather_renewable_correction_backtest.md"

FEATURE_COLS = [
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

LGB_PARAMS = {
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


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def month_hour_mean_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    global_mean = float(train["renewable_error"].mean())
    mh = train.groupby(["month", "hour"], observed=True)["renewable_error"].mean()
    month = train.groupby("month", observed=True)["renewable_error"].mean()
    preds: list[float] = []
    for row in valid[["month", "hour"]].itertuples(index=False):
        key = (row.month, row.hour)
        if key in mh.index:
            preds.append(float(mh.loc[key]))
        elif row.month in month.index:
            preds.append(float(month.loc[row.month]))
        else:
            preds.append(global_mean)
    return np.asarray(preds, dtype=float)


def lgb_predict(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[np.ndarray, int]:
    cutoff = train["target_hour"].quantile(0.85)
    fit = train[train["target_hour"] <= cutoff]
    es = train[train["target_hour"] > cutoff]
    if len(es) < 200:
        fit = train
        es = train
    train_set = lgb.Dataset(fit[FEATURE_COLS], label=fit["renewable_error"])
    valid_set = lgb.Dataset(es[FEATURE_COLS], label=es["renewable_error"], reference=train_set)
    model = lgb.train(
        dict(LGB_PARAMS),
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    best_iter = model.best_iteration if model.best_iteration and model.best_iteration > 0 else None
    return np.asarray(model.predict(valid[FEATURE_COLS], num_iteration=best_iter)), int(
        best_iter or 500
    )


def add_corrections(valid: pd.DataFrame, pred_error: np.ndarray, prefix: str) -> pd.DataFrame:
    out = valid.copy()
    out[f"{prefix}_pred_error"] = pred_error
    out[f"{prefix}_renewable_corrected"] = out["renewable_forecast"] - pred_error
    out[f"{prefix}_renewable_error"] = (
        out[f"{prefix}_renewable_corrected"] - out["renewable_actual"]
    )
    out[f"{prefix}_bid_space_fct"] = (
        out[PRED_COLS["load"]]
        - out[f"{prefix}_renewable_corrected"]
        - out[PRED_COLS["tie_line"]]
        - out[PRED_COLS["hydro"]]
        - out[PRED_COLS["non_market"]]
    )
    out[f"{prefix}_bid_space_error"] = out[f"{prefix}_bid_space_fct"] - out["bid_space_act"]
    return out


def add_bid_space(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bid_space_fct"] = (
        out[PRED_COLS["load"]]
        - out["renewable_forecast"]
        - out[PRED_COLS["tie_line"]]
        - out[PRED_COLS["hydro"]]
        - out[PRED_COLS["non_market"]]
    )
    out["bid_space_act"] = (
        out[ACTUAL_COLS["load"]]
        - out["renewable_actual"]
        - out[ACTUAL_COLS["tie_line"]]
        - out[ACTUAL_COLS["hydro"]]
        - out[ACTUAL_COLS["non_market"]]
    )
    out["bid_space_error"] = out["bid_space_fct"] - out["bid_space_act"]
    return out


def _mae(s: pd.Series) -> float:
    return float(s.abs().mean())


def _rmse(s: pd.Series) -> float:
    return float(np.sqrt(np.mean(s**2)))


def summarize_fold(df: pd.DataFrame, *, fold_name: str, best_iter: int) -> dict[str, object]:
    return {
        "fold": fold_name,
        "hours": len(df),
        "days": df["valid_date"].nunique(),
        "orig_renewable_mae": _mae(df["renewable_error"]),
        "mh_renewable_mae": _mae(df["mh_renewable_error"]),
        "lgb_renewable_mae": _mae(df["lgb_renewable_error"]),
        "orig_bid_space_mae": _mae(df["bid_space_error"]),
        "mh_bid_space_mae": _mae(df["mh_bid_space_error"]),
        "lgb_bid_space_mae": _mae(df["lgb_bid_space_error"]),
        "orig_bid_space_rmse": _rmse(df["bid_space_error"]),
        "mh_bid_space_rmse": _rmse(df["mh_bid_space_error"]),
        "lgb_bid_space_rmse": _rmse(df["lgb_bid_space_error"]),
        "lgb_best_iteration": best_iter,
    }


def window_means(values: np.ndarray, block_size: int = 8) -> np.ndarray:
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    return (prefix[block_size:] - prefix[:-block_size]) / block_size


def rank_window_pairs(window_values: np.ndarray) -> list[tuple[float, int, int]]:
    if len(window_values) != 89:
        raise ValueError(f"expected 89 window values, got {len(window_values)}")
    out: list[tuple[float, int, int]] = []
    for charge_start in range(0, 81):
        for discharge_start in range(charge_start + 8, 89):
            spread = float(window_values[discharge_start] - window_values[charge_start])
            out.append((spread, charge_start, discharge_start))
    out.sort(key=lambda item: item[0], reverse=True)
    return out


def pair_profit(price: np.ndarray, charge_start: int, discharge_start: int) -> float:
    power = np.zeros(96, dtype=float)
    power[charge_start : charge_start + 8] = -1000.0
    power[discharge_start : discharge_start + 8] = 1000.0
    return daily_profit(price, power)


def daily_dispatch_summary(df: pd.DataFrame, *, fold_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in df.groupby("valid_date", sort=True):
        day15 = group.sort_values("target_hour")
        # This script works hourly. Expand each hourly value to four 15-minute points for
        # comparable 8-slot dispatch diagnostics on the original 96-point price curve proxy.
        if len(day15) != 24:
            continue
        price_hour = day15["price_hour"].to_numpy(dtype=float)
        price = np.repeat(price_hour, 4)
        oracle = optimize_day(price)
        oracle_profit = daily_profit(price, oracle.power)
        price_win = window_means(price)
        price_pair = rank_window_pairs(price_win)[0]
        for prefix, col in [
            ("orig", "bid_space_fct"),
            ("mh", "mh_bid_space_fct"),
            ("lgb", "lgb_bid_space_fct"),
        ]:
            signal = np.repeat(day15[col].to_numpy(dtype=float), 4)
            pair = rank_window_pairs(window_means(signal))[0]
            profit = pair_profit(price, pair[1], pair[2])
            rows.append(
                {
                    "date": pd.Timestamp(date).date().isoformat(),
                    "fold": fold_name,
                    "method": prefix,
                    "profit": profit,
                    "oracle_profit": oracle_profit,
                    "regret": oracle_profit - profit,
                    "charge_start": pair[1],
                    "discharge_start": pair[2],
                    "oracle_charge_start": price_pair[1],
                    "oracle_discharge_start": price_pair[2],
                    "charge_gap": pair[1] - price_pair[1],
                    "discharge_gap": pair[2] - price_pair[2],
                    "predicted_spread": pair[0],
                    "oracle_spread": price_pair[0],
                }
            )
    return pd.DataFrame(rows)


def add_hourly_price(cfg: dict, hourly: pd.DataFrame) -> pd.DataFrame:
    # Reuse the same loader through merge_weather_renewable's train frame dependencies,
    # but keep this local to avoid changing production data APIs.
    from electricity.data import load_train_frame

    train = load_train_frame(cfg).copy()
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    train["target_hour"] = train[time_col].dt.floor("h")
    price_hour = train.groupby("target_hour", as_index=False)[target_col].mean()
    price_hour = price_hour.rename(columns={target_col: "price_hour"})
    return hourly.merge(price_hour, on="target_hour", how="inner")


def run_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    hourly = add_hourly_price(cfg, merge_weather_renewable(cfg))
    hourly = add_bid_space(hourly)
    fold_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    for fold in cfg["folds"]:
        train_end = pd.Timestamp(fold["train_end"]).floor("h")
        valid_start = pd.Timestamp(fold["valid_start"]).floor("h")
        valid_end = pd.Timestamp(fold["valid_end"]).floor("h")
        train = hourly[hourly["target_hour"] <= train_end].copy()
        valid = hourly[
            (hourly["target_hour"] >= valid_start) & (hourly["target_hour"] <= valid_end)
        ].copy()
        mh_pred = month_hour_mean_predict(train, valid)
        lgb_pred, best_iter = lgb_predict(train, valid)
        valid = add_corrections(valid, mh_pred, "mh")
        valid = add_corrections(valid, lgb_pred, "lgb")
        valid["fold"] = fold["name"]
        fold_rows.append(summarize_fold(valid, fold_name=fold["name"], best_iter=best_iter))
        pred_frames.append(valid)
        daily_frames.append(daily_dispatch_summary(valid, fold_name=fold["name"]))
    return (
        pd.DataFrame(fold_rows),
        pd.concat(pred_frames, ignore_index=True),
        pd.concat(daily_frames, ignore_index=True),
    )


def daily_method_summary(daily: pd.DataFrame) -> pd.DataFrame:
    return (
        daily.groupby("method", as_index=False)
        .agg(
            days=("date", "count"),
            mean_profit=("profit", "mean"),
            mean_oracle_profit=("oracle_profit", "mean"),
            mean_regret=("regret", "mean"),
            loss_days=("profit", lambda s: int((s < 0).sum())),
            mean_abs_charge_gap=("charge_gap", lambda s: s.abs().mean()),
            mean_abs_discharge_gap=("discharge_gap", lambda s: s.abs().mean()),
        )
        .sort_values("mean_profit", ascending=False)
    )


def write_report(fold_summary: pd.DataFrame, preds: pd.DataFrame, daily: pd.DataFrame) -> None:
    method_summary = daily_method_summary(daily)
    mae_rows = []
    for method, err_col, bid_col in [
        ("orig", "renewable_error", "bid_space_error"),
        ("month_hour_mean", "mh_renewable_error", "mh_bid_space_error"),
        ("lgb_weather", "lgb_renewable_error", "lgb_bid_space_error"),
    ]:
        mae_rows.append(
            {
                "method": method,
                "renewable_mae": _mae(preds[err_col]),
                "renewable_rmse": _rmse(preds[err_col]),
                "bid_space_mae": _mae(preds[bid_col]),
                "bid_space_rmse": _rmse(preds[bid_col]),
            }
        )
    overall = pd.DataFrame(mae_rows)
    lines = [
        "# Weather Renewable Correction Backtest",
        "",
        "This experiment predicts `renewable_error = forecast - actual` using only past "
        "fold data, then subtracts the predicted error from renewable forecast.",
        "",
        "Methods:",
        "",
        "- `orig`: original renewable forecast.",
        "- `month_hour_mean`: historical month x hour mean renewable-error correction.",
        "- `lgb_weather`: LightGBM correction using NWP and renewable forecast features.",
        "",
        "## Overall Error Summary",
        "",
        markdown_table(overall, floatfmt=".4f"),
        "",
        "## Fold Error Summary",
        "",
        markdown_table(fold_summary, floatfmt=".4f"),
        "",
        "## Hourly Bid-Space Dispatch Proxy Summary",
        "",
        "This is only a proxy: hourly bid-space signals are repeated to 15-minute slots, "
        "then legal 8-slot pairs are selected. It tests directionality, not final champion impact.",
        "",
        markdown_table(method_summary, floatfmt=".4f"),
        "",
        "## Readout",
        "",
        "- Use this to decide whether corrected renewable improves bid-space accuracy.",
        "- Do not promote to champion unless fold-level bid-space error and profit proxy improve.",
        "- If useful, next step is adding corrected renewable/bid_space as model or "
        "reranker features.",
        "",
        "## Artifacts",
        "",
        "- `reports/weather_renewable_correction_folds.csv`",
        "- `reports/weather_renewable_correction_predictions.csv`",
        "- `reports/weather_renewable_correction_daily_dispatch.csv`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    fold_summary, preds, daily = run_backtest()
    fold_summary.to_csv(REPORTS_DIR / "weather_renewable_correction_folds.csv", index=False)
    preds.to_csv(REPORTS_DIR / "weather_renewable_correction_predictions.csv", index=False)
    daily.to_csv(REPORTS_DIR / "weather_renewable_correction_daily_dispatch.csv", index=False)
    write_report(fold_summary, preds, daily)
    print(fold_summary.to_string(index=False))
    print(daily_method_summary(daily).to_string(index=False))
    print(OUT_MD)


if __name__ == "__main__":
    main()
