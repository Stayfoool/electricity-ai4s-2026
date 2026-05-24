from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from electricity.data import load_train_frame
from electricity.dispatch import DispatchResult, optimize_day
from electricity.eval.backtest import FoldResult, _fold_bounds, _fold_train_frame
from electricity.eval.metrics import daily_profit
from electricity.features import build_feature_frame, feature_columns
from electricity.features.pair_window_features import (
    PAIR_TARGET_COL,
    build_pair_window_frame,
    pair_window_feature_columns,
)
from electricity.features.window_features import build_window_frame
from electricity.models import best_iteration, predict_model, train_model


def optimize_pair_window_day(
    pair_df: pd.DataFrame,
    *,
    tau: float = 0.0,
    block_size: int = 8,
    points_per_day: int = 96,
    charge_power: float = -1000.0,
    discharge_power: float = 1000.0,
) -> DispatchResult:
    if pair_df.empty:
        raise ValueError("pair_df must not be empty")
    if "pred_pair_spread" not in pair_df.columns:
        raise ValueError("pair_df missing pred_pair_spread")

    sorted_pairs = pair_df.sort_values("pred_pair_spread", ascending=False)
    best = sorted_pairs.iloc[0]
    best_spread = float(best["pred_pair_spread"])
    best_tc: int | None = int(best["charge_start"])
    best_td: int | None = int(best["discharge_start"])
    top_spreads = sorted_pairs["pred_pair_spread"].head(5).to_numpy(dtype=float)
    top2_spread = float(top_spreads[1]) if len(top_spreads) > 1 else best_spread
    top5_mean = float(top_spreads.mean())
    top5_std = float(top_spreads.std())

    power = np.zeros(points_per_day, dtype=float)
    if best_spread > tau:
        power[best_tc : best_tc + block_size] = charge_power
        power[best_td : best_td + block_size] = discharge_power
    else:
        best_tc = None
        best_td = None

    return DispatchResult(
        power=power,
        charge_start=best_tc,
        discharge_start=best_td,
        predicted_spread=best_spread,
        top2_spread=top2_spread,
        top5_spread_mean=top5_mean,
        top5_spread_std=top5_std,
        top1_top2_gap=float(best_spread - top2_spread),
        top1_top5_mean_gap=float(best_spread - top5_mean),
        top_candidate_count=len(pair_df),
    )


def _evaluate_pair_dispatch(
    valid_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    tau: float,
    block_size: int,
    points_per_day: int,
) -> dict:
    by_day = valid_df[[time_col, target_col]].copy()
    by_day["date"] = by_day[time_col].dt.normalize()
    by_day["slot"] = by_day[time_col].dt.hour * 4 + by_day[time_col].dt.minute // 15

    pred_by_day = {date: group.copy() for date, group in pair_df.groupby("date", sort=True)}

    profits: list[float] = []
    oracle_profits: list[float] = []
    regrets: list[float] = []
    abs_charge_gaps: list[float] = []
    abs_discharge_gaps: list[float] = []
    trade_days = 0
    loss_days = 0
    evaluated_days = 0
    day_rows: list[dict] = []

    for date, group in by_day.groupby("date", sort=True):
        if len(group) != points_per_day or date not in pred_by_day:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(points_per_day)):
            continue

        true_prices = group[target_col].to_numpy(dtype=float)
        result = optimize_pair_window_day(
            pred_by_day[date],
            tau=tau,
            block_size=block_size,
            points_per_day=points_per_day,
        )
        oracle = optimize_day(true_prices, tau=0.0, block_size=block_size)
        profit = daily_profit(true_prices, result.power)
        oracle_profit = daily_profit(true_prices, oracle.power)
        traded = int(np.count_nonzero(result.power) > 0)
        regret = oracle_profit - profit
        if profit < 0:
            loss_days += 1
        if result.charge_start is not None and oracle.charge_start is not None:
            abs_charge_gaps.append(float(abs(result.charge_start - oracle.charge_start)))
        if result.discharge_start is not None and oracle.discharge_start is not None:
            abs_discharge_gaps.append(float(abs(result.discharge_start - oracle.discharge_start)))

        profits.append(profit)
        oracle_profits.append(oracle_profit)
        regrets.append(regret)
        trade_days += traded
        evaluated_days += 1
        day_rows.append(
            {
                "date": date.date().isoformat(),
                "profit": profit,
                "oracle_profit": oracle_profit,
                "traded": traded,
                "charge_start": result.charge_start,
                "discharge_start": result.discharge_start,
                "predicted_spread": result.predicted_spread,
                "oracle_charge_start": oracle.charge_start,
                "oracle_discharge_start": oracle.discharge_start,
                "regret": regret,
                "profit_ratio_day": profit / oracle_profit if oracle_profit > 0 else 0.0,
                "charge_start_gap": (
                    None
                    if result.charge_start is None or oracle.charge_start is None
                    else result.charge_start - oracle.charge_start
                ),
                "discharge_start_gap": (
                    None
                    if result.discharge_start is None or oracle.discharge_start is None
                    else result.discharge_start - oracle.discharge_start
                ),
            }
        )

    avg_profit = float(np.mean(profits)) if profits else 0.0
    oracle_avg = float(np.mean(oracle_profits)) if oracle_profits else 0.0
    return {
        "avg_profit": avg_profit,
        "oracle_avg_profit": oracle_avg,
        "oracle_ratio": avg_profit / oracle_avg if oracle_avg > 0 else 0.0,
        "avg_regret": float(np.mean(regrets)) if regrets else 0.0,
        "loss_days": loss_days,
        "mean_abs_charge_gap": float(np.mean(abs_charge_gaps)) if abs_charge_gaps else 0.0,
        "mean_abs_discharge_gap": float(np.mean(abs_discharge_gaps)) if abs_discharge_gaps else 0.0,
        "trade_days": trade_days,
        "valid_days": evaluated_days,
        "trade_days_ratio": trade_days / evaluated_days if evaluated_days else 0.0,
        "day_rows": day_rows,
    }


def _build_pair_frame(
    point_df: pd.DataFrame,
    *,
    time_col: str,
    base_feature_cols: list[str],
    target_col: str,
    pair_agg_stats: list[str] | None,
    block_size: int,
    points_per_day: int,
) -> pd.DataFrame:
    windows = build_window_frame(
        point_df,
        time_col=time_col,
        base_feature_cols=base_feature_cols,
        target_col=target_col,
        block_size=block_size,
        points_per_day=points_per_day,
    )
    return build_pair_window_frame(
        windows,
        base_feature_cols=base_feature_cols,
        agg_stats=pair_agg_stats,
        block_size=block_size,
        points_per_day=points_per_day,
    )


def _pair_point_feature_cols(cfg: dict, available_cols: list[str]) -> list[str]:
    configured = cfg["model"].get("pair_point_feature_cols")
    if configured is None:
        return available_cols
    selected = list(configured)
    missing = sorted(set(selected) - set(available_cols))
    if missing:
        raise ValueError(f"model.pair_point_feature_cols contains unavailable columns: {missing}")
    return selected


def run_pair_window_backtest(cfg: dict, *, config_path: str) -> Path:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    train_window_days = cfg["model"].get("train_window_days")
    block_size = int(cfg["dispatch"]["block_size"])
    points_per_day = int(cfg["dispatch"]["points_per_day"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_point_feature_cols = feature_columns(cfg)
    point_feature_cols = _pair_point_feature_cols(cfg, all_point_feature_cols)
    pair_agg_stats = cfg["model"].get("pair_agg_stats")
    pair_feature_cols = pair_window_feature_columns(
        point_feature_cols,
        agg_stats=pair_agg_stats,
    )
    df = build_feature_frame(load_train_frame(cfg), cfg, time_col=time_col)

    mlflow.set_tracking_uri(str(Path(cfg["paths"]["mlruns_dir"]).resolve()))
    mlflow.set_experiment(cfg["model"]["name"])

    fold_results: list[FoldResult] = []
    all_day_rows: list[dict] = []

    with mlflow.start_run(run_name=f"{cfg['model']['name']}_rolling"):
        mlflow.log_param("config", config_path)
        mlflow.log_param("model", cfg["model"]["name"])
        mlflow.log_param("target_mode", "pair_spread")
        mlflow.log_param("train_window_days", train_window_days or "all_past")
        mlflow.log_param(
            "pair_agg_stats",
            ",".join(pair_agg_stats or ["mean", "std", "min", "max"]),
        )
        mlflow.log_param("all_point_feature_count", len(all_point_feature_cols))
        mlflow.log_param("point_feature_count", len(point_feature_cols))
        mlflow.log_param("pair_feature_count", len(pair_feature_cols))

        for fold in cfg["folds"]:
            print(f"running_fold={fold['name']}", flush=True)
            bounds = _fold_bounds(fold)
            train_end = bounds.train_end
            valid_start = bounds.valid_start
            valid_end = bounds.valid_end

            train_points = _fold_train_frame(
                df,
                time_col=time_col,
                train_end=train_end,
                train_window_days=train_window_days,
                train_start=bounds.train_start,
                valid_start=valid_start,
                valid_end=valid_end,
            )
            valid_points = df[(df[time_col] >= valid_start) & (df[time_col] <= valid_end)].copy()
            if train_points.empty or valid_points.empty:
                raise ValueError(f"empty train/valid split for fold={fold['name']}")

            train_pairs = _build_pair_frame(
                train_points,
                time_col=time_col,
                base_feature_cols=point_feature_cols,
                target_col=target_col,
                pair_agg_stats=pair_agg_stats,
                block_size=block_size,
                points_per_day=points_per_day,
            )
            valid_pairs = _build_pair_frame(
                valid_points,
                time_col=time_col,
                base_feature_cols=point_feature_cols,
                target_col=target_col,
                pair_agg_stats=pair_agg_stats,
                block_size=block_size,
                points_per_day=points_per_day,
            )
            if train_pairs.empty or valid_pairs.empty:
                raise ValueError(f"empty pair frame for fold={fold['name']}")

            actual_train_start = train_points[time_col].min()
            actual_train_end = train_points[time_col].max()
            print(
                "train_range="
                f"{actual_train_start.strftime('%Y-%m-%d')}.."
                f"{actual_train_end.strftime('%Y-%m-%d')} "
                f"train_pairs={len(train_pairs)} valid_pairs={len(valid_pairs)}",
                flush=True,
            )

            model = train_model(
                train_pairs,
                valid_pairs,
                feature_cols=pair_feature_cols,
                target_col=PAIR_TARGET_COL,
                cfg=cfg,
            )
            valid_pairs["pred_pair_spread"] = predict_model(
                model, valid_pairs, pair_feature_cols, cfg
            )
            y_true = valid_pairs[PAIR_TARGET_COL].to_numpy(dtype=float)
            y_pred = valid_pairs["pred_pair_spread"].to_numpy(dtype=float)
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

            dispatch_metrics = _evaluate_pair_dispatch(
                valid_points,
                valid_pairs,
                time_col=time_col,
                target_col=target_col,
                tau=float(cfg["dispatch"]["tau"]),
                block_size=block_size,
                points_per_day=points_per_day,
            )

            result = FoldResult(
                fold=fold["name"],
                train_rows=len(train_pairs),
                valid_rows=len(valid_pairs),
                valid_days=dispatch_metrics["valid_days"],
                train_window_days=train_window_days,
                train_start=actual_train_start.strftime("%Y-%m-%d %H:%M:%S"),
                train_end=actual_train_end.strftime("%Y-%m-%d %H:%M:%S"),
                target_mode="pair_spread",
                mae=mae,
                rmse=rmse,
                curve_z_rmse=0.0,
                mean_abs_charge_gap=dispatch_metrics["mean_abs_charge_gap"],
                mean_abs_discharge_gap=dispatch_metrics["mean_abs_discharge_gap"],
                avg_profit=dispatch_metrics["avg_profit"],
                oracle_avg_profit=dispatch_metrics["oracle_avg_profit"],
                oracle_ratio=dispatch_metrics["oracle_ratio"],
                avg_regret=dispatch_metrics["avg_regret"],
                loss_days=dispatch_metrics["loss_days"],
                trade_days=dispatch_metrics["trade_days"],
                trade_days_ratio=dispatch_metrics["trade_days_ratio"],
                best_iteration=best_iteration(model, cfg),
            )
            fold_results.append(result)
            for row in dispatch_metrics["day_rows"]:
                row["fold"] = fold["name"]
                all_day_rows.append(row)

            for key in ["mae", "rmse", "avg_profit", "oracle_ratio"]:
                mlflow.log_metric(f"{fold['name']}_{key}", getattr(result, key))

        summary = pd.DataFrame([r.__dict__ for r in fold_results])
        day_detail = pd.DataFrame(all_day_rows)
        summary_path = reports_dir / f"backtest_{cfg['model']['name']}.csv"
        detail_path = reports_dir / f"backtest_{cfg['model']['name']}_daily.csv"
        summary.to_csv(summary_path, index=False)
        day_detail.to_csv(detail_path, index=False)
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(detail_path))

    print(summary.to_string(index=False))
    print(f"summary_path={summary_path}")
    print(f"daily_path={detail_path}")
    return summary_path
