from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from electricity.dispatch import optimize_day
from electricity.eval.metrics import daily_profit
from electricity.features.bid_space import ACTUAL_COLS, PRED_COLS, markdown_table
from electricity.features.weather import bjt_valid_times, hourly_renewable_frame, nc_dir

CONFIG_PATH = Path("configs/ensemble_champion_segmented6_prior.yaml")
REPORTS_DIR = Path("reports")
TOP_CELLS_PATH = REPORTS_DIR / "weather_spatial_top_cells.csv"
OUT_FOLDS = REPORTS_DIR / "weather_spatial_correction_folds.csv"
OUT_DAILY = REPORTS_DIR / "weather_spatial_correction_daily_dispatch.csv"
OUT_FEATURES = REPORTS_DIR / "weather_spatial_selected_features.csv"
OUT_IMPORTANCE = REPORTS_DIR / "weather_spatial_correction_importance.csv"
OUT_MD = REPORTS_DIR / "weather_spatial_correction.md"

BASE_FEATURES = ["month", "hour", "renewable_forecast", "wind_forecast", "solar_forecast"]
TARGETS_FOR_SELECTION = [
    "renewable_error_mh_resid",
    "wind_error_mh_resid",
    "solar_error_mh_resid",
    "renewable_error",
    "wind_error",
    "solar_error",
]
SCOPES_FOR_SELECTION = [
    "winter_daylight",
    "winter_like_01_02_11_12",
    "jan_feb",
    "midday_10_17",
    "daylight_07_18",
    "all",
]
SCOPE_PRIORITY = {name: i for i, name in enumerate(SCOPES_FOR_SELECTION)}
TARGET_PRIORITY = {name: i for i, name in enumerate(TARGETS_FOR_SELECTION)}
MAX_SPATIAL_FEATURES = 30

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


@dataclass(frozen=True)
class CellFeature:
    weather_var: str
    lat: float
    lon: float
    lat_idx: int
    lon_idx: int

    @property
    def name(self) -> str:
        lat = f"{self.lat:.2f}".replace(".", "p")
        lon = f"{self.lon:.2f}".replace(".", "p")
        return f"sp_{self.weather_var}_lat{lat}_lon{lon}"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fold_defs(cfg: dict) -> list[dict[str, object]]:
    return [
        {
            "name": fold["name"],
            "train_end": pd.Timestamp(fold["train_end"]),
            "valid_start": pd.Timestamp(fold["valid_start"]),
            "valid_end": pd.Timestamp(fold["valid_end"]),
        }
        for fold in cfg["folds"]
    ]


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


def select_cells_for_fold(
    top_cells: pd.DataFrame,
    fold_name: str,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
) -> list[CellFeature]:
    cand = top_cells[
        (top_cells["fold"] == fold_name)
        & top_cells["target"].isin(TARGETS_FOR_SELECTION)
        & top_cells["scope"].isin(SCOPES_FOR_SELECTION)
        & (top_cells["rank"] <= 10)
    ].copy()
    if cand.empty:
        raise ValueError(f"no spatial cells for fold={fold_name}")
    cand["scope_priority"] = cand["scope"].map(SCOPE_PRIORITY)
    cand["target_priority"] = cand["target"].map(TARGET_PRIORITY)
    cand = cand.sort_values(
        ["scope_priority", "target_priority", "abs_pearson", "rank"],
        ascending=[True, True, False, True],
    )

    selected: list[CellFeature] = []
    seen: set[tuple[str, float, float]] = set()
    for row in cand.itertuples(index=False):
        key = (str(row.weather_var), round(float(row.lat), 4), round(float(row.lon), 4))
        if key in seen:
            continue
        lat_idx = int(np.argmin(np.abs(lat_values - float(row.lat))))
        lon_idx = int(np.argmin(np.abs(lon_values - float(row.lon))))
        selected.append(
            CellFeature(
                weather_var=str(row.weather_var),
                lat=float(lat_values[lat_idx]),
                lon=float(lon_values[lon_idx]),
                lat_idx=lat_idx,
                lon_idx=lon_idx,
            )
        )
        seen.add(key)
        if len(selected) >= MAX_SPATIAL_FEATURES:
            break
    return selected


def extract_spatial_features(cfg: dict, cells: list[CellFeature]) -> pd.DataFrame:
    paths = sorted(nc_dir(cfg).glob("*.nc"))
    if not paths:
        raise FileNotFoundError("no NWP nc files found")
    rows: list[dict[str, object]] = []
    for i, path in enumerate(paths, start=1):
        if i == 1 or i % 50 == 0 or i == len(paths):
            print(f"extracting spatial features {i}/{len(paths)} {path.name}", flush=True)
        with xr.open_dataset(path) as ds:
            valid_times = bjt_valid_times(ds)
            data = ds["data"].isel(time=0)
            u100 = data.sel(channel="u100").to_numpy()
            v100 = data.sel(channel="v100").to_numpy()
            channel_cache = {
                "ghi": data.sel(channel="ghi").to_numpy(),
                "tcc": data.sel(channel="tcc").to_numpy(),
                "u100": u100,
                "v100": v100,
                "wind_speed": np.sqrt(u100**2 + v100**2),
            }
            for lead_idx, ts in enumerate(valid_times):
                row: dict[str, object] = {"target_hour": pd.Timestamp(ts)}
                for cell in cells:
                    arr = channel_cache[cell.weather_var]
                    row[cell.name] = float(arr[lead_idx, cell.lat_idx, cell.lon_idx])
                rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("target_hour").sort_values("target_hour")


def train_lgb(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, int, pd.DataFrame]:
    cutoff = train["target_hour"].quantile(0.85)
    fit = train[train["target_hour"] <= cutoff]
    es = train[train["target_hour"] > cutoff]
    if len(es) < 200:
        fit = train
        es = train
    train_set = lgb.Dataset(fit[feature_cols], label=fit[target_col], feature_name=feature_cols)
    valid_set = lgb.Dataset(
        es[feature_cols],
        label=es[target_col],
        feature_name=feature_cols,
        reference=train_set,
    )
    model = lgb.train(
        dict(LGB_PARAMS),
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    best_iter = model.best_iteration if model.best_iteration and model.best_iteration > 0 else 500
    pred = np.asarray(model.predict(valid[feature_cols], num_iteration=best_iter), dtype=float)
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    return pred, int(best_iter), importance


def add_correction(valid: pd.DataFrame, pred_error: np.ndarray, prefix: str) -> pd.DataFrame:
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


def _mae(s: pd.Series) -> float:
    return float(s.abs().mean())


def _rmse(s: pd.Series) -> float:
    return float(np.sqrt(np.mean(s**2)))


def summarize_fold(
    df: pd.DataFrame,
    *,
    fold_name: str,
    raw_iter: int,
    resid_iter: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "fold": fold_name,
        "hours": len(df),
        "days": df["valid_date"].nunique(),
        "raw_best_iteration": raw_iter,
        "resid_best_iteration": resid_iter,
    }
    for prefix, err_col, bid_col in [
        ("orig", "renewable_error", "bid_space_error"),
        ("mh", "mh_renewable_error", "mh_bid_space_error"),
        ("spatial_raw", "spatial_raw_renewable_error", "spatial_raw_bid_space_error"),
        ("spatial_resid", "spatial_resid_renewable_error", "spatial_resid_bid_space_error"),
    ]:
        row[f"{prefix}_renewable_mae"] = _mae(df[err_col])
        row[f"{prefix}_renewable_rmse"] = _rmse(df[err_col])
        row[f"{prefix}_bid_space_mae"] = _mae(df[bid_col])
        row[f"{prefix}_bid_space_rmse"] = _rmse(df[bid_col])
    return row


def window_means(values: np.ndarray, block_size: int = 8) -> np.ndarray:
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    return (prefix[block_size:] - prefix[:-block_size]) / block_size


def rank_window_pairs(window_values: np.ndarray) -> list[tuple[float, int, int]]:
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
        day = group.sort_values("target_hour")
        if len(day) != 24:
            continue
        price = np.repeat(day["price_hour"].to_numpy(dtype=float), 4)
        oracle = optimize_day(price)
        oracle_profit = daily_profit(price, oracle.power)
        for prefix, col in [
            ("orig", "bid_space_fct"),
            ("mh", "mh_bid_space_fct"),
            ("spatial_raw", "spatial_raw_bid_space_fct"),
            ("spatial_resid", "spatial_resid_bid_space_fct"),
        ]:
            signal = np.repeat(day[col].to_numpy(dtype=float), 4)
            spread, charge_start, discharge_start = rank_window_pairs(window_means(signal))[0]
            profit = pair_profit(price, charge_start, discharge_start)
            rows.append(
                {
                    "date": pd.Timestamp(date).date().isoformat(),
                    "fold": fold_name,
                    "method": prefix,
                    "profit": profit,
                    "oracle_profit": oracle_profit,
                    "regret": oracle_profit - profit,
                    "loss_day": int(profit < -1e-9),
                    "charge_start": charge_start,
                    "discharge_start": discharge_start,
                    "predicted_spread": spread,
                }
            )
    return pd.DataFrame(rows)


def write_report(folds: pd.DataFrame, daily: pd.DataFrame, importance: pd.DataFrame) -> None:
    methods = ["orig", "mh", "spatial_raw", "spatial_resid"]
    err_rows = []
    for method in methods:
        err_rows.append(
            {
                "method": method,
                "renewable_mae": folds[f"{method}_renewable_mae"].mean(),
                "renewable_rmse": folds[f"{method}_renewable_rmse"].mean(),
                "bid_space_mae": folds[f"{method}_bid_space_mae"].mean(),
                "bid_space_rmse": folds[f"{method}_bid_space_rmse"].mean(),
            }
        )
    err_summary = pd.DataFrame(err_rows)
    daily_summary = daily.groupby("method", observed=True).agg(
        days=("date", "nunique"),
        mean_profit=("profit", "mean"),
        mean_oracle_profit=("oracle_profit", "mean"),
        mean_regret=("regret", "mean"),
        loss_days=("loss_day", "sum"),
    ).reset_index()
    top_importance = importance.sort_values("gain", ascending=False).head(40)
    lines = [
        "# Weather Spatial Correction Backtest",
        "",
        "## Goal",
        "",
        "Use fold-safe spatially selected NWP grid cells to correct renewable forecast error, "
        "then compare renewable and bid-space reconstruction against the old coarse approach.",
        "",
        "## Methods",
        "",
        "- `orig`: original renewable forecast.",
        "- `mh`: month x hour mean renewable-error correction.",
        "- `spatial_raw`: LightGBM predicts raw renewable forecast error from selected cells.",
        "- `spatial_resid`: month-hour mean plus LightGBM residual correction.",
        "",
        "## Error Summary",
        "",
        markdown_table(err_summary, floatfmt=".4f"),
        "",
        "## Fold Error Summary",
        "",
        markdown_table(folds, floatfmt=".4f"),
        "",
        "## Hourly Bid-Space Dispatch Proxy",
        "",
        markdown_table(daily_summary, floatfmt=".4f"),
        "",
        "## Top Feature Importance",
        "",
        markdown_table(top_importance, floatfmt=".4f"),
        "",
        "## Readout",
        "",
        "- This still evaluates a bid-space proxy, not the final champion price model.",
        "- Promote only if spatial correction improves bid-space error and does not worsen "
        "the dispatch proxy.",
        "- If `spatial_resid` beats `spatial_raw`, weather adds signal beyond month/hour bias.",
        "",
        "## Artifacts",
        "",
        f"- `{OUT_FOLDS}`",
        f"- `{OUT_DAILY}`",
        f"- `{OUT_FEATURES}`",
        f"- `{OUT_IMPORTANCE}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    top_cells = pd.read_csv(TOP_CELLS_PATH)
    folds = fold_defs(cfg)
    hourly = add_bid_space(hourly_renewable_frame(cfg))
    hourly["price_hour"] = hourly["renewable_error"] * 0.0
    # Use hourly mean of true price for proxy dispatch diagnostics.
    price = pd.read_csv(
        Path(cfg["paths"]["material_dir"]) / cfg["data"]["train_label"],
        parse_dates=[cfg["data"]["time_col"]],
    )
    price["target_hour"] = price[cfg["data"]["time_col"]].dt.floor("h")
    price_hour = price.groupby("target_hour", observed=True)[cfg["data"]["target_col"]].mean()
    hourly = hourly.drop(columns=["price_hour"]).merge(
        price_hour.rename("price_hour"), on="target_hour", how="left"
    )

    with xr.open_dataset(sorted(nc_dir(cfg).glob("*.nc"))[0]) as ds:
        lat_values = ds["lat"].values
        lon_values = ds["lon"].values

    fold_rows: list[dict[str, object]] = []
    daily_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    importance_rows: list[pd.DataFrame] = []

    for fold in folds:
        fold_name = str(fold["name"])
        print(f"running_fold={fold_name}", flush=True)
        cells = select_cells_for_fold(top_cells, fold_name, lat_values, lon_values)
        for idx, cell in enumerate(cells, start=1):
            feature_rows.append(
                {
                    "fold": fold_name,
                    "feature_rank": idx,
                    "feature": cell.name,
                    "weather_var": cell.weather_var,
                    "lat": cell.lat,
                    "lon": cell.lon,
                }
            )
        spatial = extract_spatial_features(cfg, cells)
        frame = hourly.merge(spatial, on="target_hour", how="left")
        spatial_cols = [cell.name for cell in cells]
        for col in spatial_cols:
            frame[col] = frame[col].fillna(frame[col].median())

        train = frame[frame["target_hour"] <= fold["train_end"]].copy()
        valid = frame[
            (frame["target_hour"] >= fold["valid_start"])
            & (frame["target_hour"] <= fold["valid_end"])
        ].copy()
        if train.empty or valid.empty:
            raise ValueError(f"empty split for fold={fold_name}")

        mh_train_pred = month_hour_mean_predict(train, train)
        mh_valid_pred = month_hour_mean_predict(train, valid)
        train["renewable_error_mh_resid"] = train["renewable_error"] - mh_train_pred

        raw_pred, raw_iter, raw_imp = train_lgb(
            train,
            valid,
            BASE_FEATURES + spatial_cols,
            "renewable_error",
        )
        resid_pred, resid_iter, resid_imp = train_lgb(
            train,
            valid,
            BASE_FEATURES + spatial_cols,
            "renewable_error_mh_resid",
        )
        raw_imp.insert(0, "model", "spatial_raw")
        raw_imp.insert(0, "fold", fold_name)
        resid_imp.insert(0, "model", "spatial_resid")
        resid_imp.insert(0, "fold", fold_name)
        importance_rows.extend([raw_imp, resid_imp])

        valid = add_correction(valid, mh_valid_pred, "mh")
        valid = add_correction(valid, raw_pred, "spatial_raw")
        valid = add_correction(valid, mh_valid_pred + resid_pred, "spatial_resid")
        fold_rows.append(
            summarize_fold(
                valid,
                fold_name=fold_name,
                raw_iter=raw_iter,
                resid_iter=resid_iter,
            )
        )
        daily_rows.append(daily_dispatch_summary(valid, fold_name=fold_name))

    folds_df = pd.DataFrame(fold_rows)
    daily_df = pd.concat(daily_rows, ignore_index=True)
    features_df = pd.DataFrame(feature_rows)
    importance_df = pd.concat(importance_rows, ignore_index=True)
    folds_df.to_csv(OUT_FOLDS, index=False)
    daily_df.to_csv(OUT_DAILY, index=False)
    features_df.to_csv(OUT_FEATURES, index=False)
    importance_df.to_csv(OUT_IMPORTANCE, index=False)
    write_report(folds_df, daily_df, importance_df)
    print(markdown_table(folds_df, floatfmt=".4f"))
    print(f"markdown_path={OUT_MD}")


if __name__ == "__main__":
    main()
