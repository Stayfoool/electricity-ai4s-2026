from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from electricity.data import load_train_frame
from electricity.dispatch import optimize_day
from electricity.eval.metrics import daily_profit
from electricity.eval.segmented_utils import (
    add_slot_and_segment,
    predict_segmented_fold,
    segment_boundaries,
    segmented_feature_build_spec,
    segmented_feature_columns,
)
from electricity.features import build_feature_frame, feature_columns
from electricity.models import best_iteration, predict_model, train_model


@dataclass(frozen=True)
class FoldResult:
    fold: str
    train_rows: int
    valid_rows: int
    valid_days: int
    train_window_days: int | None
    train_start: str
    train_end: str
    target_mode: str
    mae: float
    rmse: float
    curve_z_rmse: float
    mean_abs_charge_gap: float
    mean_abs_discharge_gap: float
    avg_profit: float
    oracle_avg_profit: float
    oracle_ratio: float
    avg_regret: float
    loss_days: int
    trade_days: int
    trade_days_ratio: float
    best_iteration: int


def _add_target_variants(df: pd.DataFrame, *, time_col: str, target_col: str) -> pd.DataFrame:
    out = df.copy()
    date = out[time_col].dt.normalize()
    daily_mean = out.groupby(date)[target_col].transform("mean")
    daily_std = out.groupby(date)[target_col].transform("std").replace(0.0, np.nan)
    out["target_absolute"] = out[target_col]
    out["target_centered"] = out[target_col] - daily_mean
    out["target_zscore"] = (out[target_col] - daily_mean) / daily_std.fillna(1.0)
    return out


def _target_col_for_mode(target_mode: str) -> str:
    if target_mode not in {"absolute", "centered", "zscore"}:
        raise ValueError(f"unsupported target_mode={target_mode}")
    return f"target_{target_mode}"


def _daily_curve_z_rmse(
    df: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
) -> float:
    work = df[[time_col, target_col, "pred"]].copy()
    work["date"] = work[time_col].dt.normalize()
    work["slot"] = work[time_col].dt.hour * 4 + work[time_col].dt.minute // 15

    errors: list[float] = []
    for _, group in work.groupby("date", sort=True):
        if len(group) != 96:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            continue

        true = group[target_col].to_numpy(dtype=float)
        pred = group["pred"].to_numpy(dtype=float)
        true_std = np.std(true)
        pred_std = np.std(pred)
        if true_std == 0 or pred_std == 0:
            continue
        true_z = (true - np.mean(true)) / true_std
        pred_z = (pred - np.mean(pred)) / pred_std
        errors.append(float(np.mean((true_z - pred_z) ** 2)))
    return float(np.sqrt(np.mean(errors))) if errors else 0.0


def _evaluate_dispatch(
    valid_df: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    tau: float,
) -> dict:
    by_day = valid_df[[time_col, target_col, "pred"]].copy()
    by_day["date"] = by_day[time_col].dt.normalize()
    by_day["slot"] = by_day[time_col].dt.hour * 4 + by_day[time_col].dt.minute // 15

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
        if len(group) != 96:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            continue

        true_prices = group[target_col].to_numpy(dtype=float)
        pred_prices = group["pred"].to_numpy(dtype=float)
        result = optimize_day(pred_prices, tau=tau)
        oracle = optimize_day(true_prices, tau=0.0)
        profit = daily_profit(true_prices, result.power)
        oracle_profit = daily_profit(true_prices, oracle.power)
        traded = int(np.count_nonzero(result.power) > 0)
        regret = oracle_profit - profit
        if profit < 0:
            loss_days += 1
        if result.charge_start is not None and oracle.charge_start is not None:
            abs_charge_gaps.append(float(abs(result.charge_start - oracle.charge_start)))
        if result.discharge_start is not None and oracle.discharge_start is not None:
            abs_discharge_gaps.append(
                float(abs(result.discharge_start - oracle.discharge_start))
            )

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
                "top2_spread": result.top2_spread,
                "top5_spread_mean": result.top5_spread_mean,
                "top5_spread_std": result.top5_spread_std,
                "top1_top2_gap": result.top1_top2_gap,
                "top1_top5_mean_gap": result.top1_top5_mean_gap,
                "top_candidate_count": result.top_candidate_count,
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
        "mean_abs_discharge_gap": (
            float(np.mean(abs_discharge_gaps)) if abs_discharge_gaps else 0.0
        ),
        "trade_days": trade_days,
        "valid_days": evaluated_days,
        "trade_days_ratio": trade_days / evaluated_days if evaluated_days else 0.0,
        "day_rows": day_rows,
    }


def _tau_values(cfg: dict) -> list[float]:
    grid = cfg["dispatch"]["tau_grid"]
    start = float(grid["start"])
    stop = float(grid["stop"])
    step = float(grid["step"])
    values: list[float] = []
    current = start
    while current <= stop + step / 2:
        values.append(round(current, 10))
        current += step
    return values


def _fold_train_frame(
    df: pd.DataFrame,
    *,
    time_col: str,
    train_end: pd.Timestamp,
    train_window_days: int | None,
) -> pd.DataFrame:
    train_df = df[df[time_col] <= train_end].copy()
    if train_window_days is not None:
        train_start = train_end - pd.Timedelta(days=train_window_days) + pd.Timedelta(seconds=1)
        train_df = train_df[train_df[time_col] >= train_start].copy()
    return train_df


def run_backtest(cfg: dict, *, config_path: str) -> Path:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    target_mode = cfg["model"].get("target_mode", "absolute")
    train_target_col = _target_col_for_mode(target_mode)
    train_window_days = cfg["model"].get("train_window_days")
    feature_cols = feature_columns(cfg)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = _add_target_variants(
        build_feature_frame(load_train_frame(cfg), cfg, time_col=time_col),
        time_col=time_col,
        target_col=target_col,
    )
    mlflow.set_tracking_uri(str(Path(cfg["paths"]["mlruns_dir"]).resolve()))
    mlflow.set_experiment(cfg["model"]["name"])

    fold_results: list[FoldResult] = []
    all_day_rows: list[dict] = []

    with mlflow.start_run(run_name=f"{cfg['model']['name']}_rolling"):
        mlflow.log_param("config", config_path)
        mlflow.log_param("model", cfg["model"]["name"])
        mlflow.log_param("target_mode", target_mode)
        mlflow.log_param("train_window_days", train_window_days or "all_past")
        mlflow.log_param("feature_count", len(feature_cols))

        for fold in cfg["folds"]:
            print(f"running_fold={fold['name']}", flush=True)
            train_end = pd.Timestamp(fold["train_end"])
            valid_start = pd.Timestamp(fold["valid_start"])
            valid_end = pd.Timestamp(fold["valid_end"])

            train_df = _fold_train_frame(
                df,
                time_col=time_col,
                train_end=train_end,
                train_window_days=train_window_days,
            )
            valid_df = df[(df[time_col] >= valid_start) & (df[time_col] <= valid_end)].copy()
            if train_df.empty or valid_df.empty:
                raise ValueError(f"empty train/valid split for fold={fold['name']}")
            actual_train_start = train_df[time_col].min()
            actual_train_end = train_df[time_col].max()
            print(
                "train_range="
                f"{actual_train_start.strftime('%Y-%m-%d')}.."
                f"{actual_train_end.strftime('%Y-%m-%d')}",
                flush=True,
            )

            model = train_model(
                train_df,
                valid_df,
                feature_cols=feature_cols,
                target_col=train_target_col,
                cfg=cfg,
            )
            valid_df["pred"] = predict_model(model, valid_df, feature_cols, cfg)
            y_true = valid_df[train_target_col].to_numpy()
            y_pred = valid_df["pred"].to_numpy()
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            curve_z_rmse = _daily_curve_z_rmse(
                valid_df,
                time_col=time_col,
                target_col=target_col,
            )
            dispatch_metrics = _evaluate_dispatch(
                valid_df,
                time_col=time_col,
                target_col=target_col,
                tau=float(cfg["dispatch"]["tau"]),
            )

            result = FoldResult(
                fold=fold["name"],
                train_rows=len(train_df),
                valid_rows=len(valid_df),
                valid_days=dispatch_metrics["valid_days"],
                train_window_days=train_window_days,
                train_start=actual_train_start.strftime("%Y-%m-%d %H:%M:%S"),
                train_end=actual_train_end.strftime("%Y-%m-%d %H:%M:%S"),
                target_mode=target_mode,
                mae=mae,
                rmse=rmse,
                curve_z_rmse=curve_z_rmse,
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

            for key in ["mae", "rmse", "curve_z_rmse", "avg_profit", "oracle_ratio"]:
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


def run_tau_search(cfg: dict, *, config_path: str) -> Path:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    target_mode = cfg["model"].get("target_mode", "absolute")
    train_target_col = _target_col_for_mode(target_mode)
    train_window_days = cfg["model"].get("train_window_days")
    feature_cols = feature_columns(cfg)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = _add_target_variants(
        build_feature_frame(load_train_frame(cfg), cfg, time_col=time_col),
        time_col=time_col,
        target_col=target_col,
    )
    mlflow.set_tracking_uri(str(Path(cfg["paths"]["mlruns_dir"]).resolve()))
    mlflow.set_experiment(f"{cfg['model']['name']}_tau")

    rows: list[dict] = []
    tau_values = _tau_values(cfg)

    with mlflow.start_run(run_name=f"{cfg['model']['name']}_tau_search"):
        mlflow.log_param("config", config_path)
        mlflow.log_param("model", cfg["model"]["name"])
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("tau_values", ",".join(str(v) for v in tau_values))

        for fold in cfg["folds"]:
            print(f"running_fold={fold['name']}", flush=True)
            train_end = pd.Timestamp(fold["train_end"])
            valid_start = pd.Timestamp(fold["valid_start"])
            valid_end = pd.Timestamp(fold["valid_end"])

            train_df = _fold_train_frame(
                df,
                time_col=time_col,
                train_end=train_end,
                train_window_days=train_window_days,
            )
            valid_df = df[(df[time_col] >= valid_start) & (df[time_col] <= valid_end)].copy()
            if train_df.empty or valid_df.empty:
                raise ValueError(f"empty train/valid split for fold={fold['name']}")

            model = train_model(
                train_df,
                valid_df,
                feature_cols=feature_cols,
                target_col=train_target_col,
                cfg=cfg,
            )
            valid_df["pred"] = predict_model(model, valid_df, feature_cols, cfg)

            for tau in tau_values:
                metrics = _evaluate_dispatch(
                    valid_df,
                    time_col=time_col,
                    target_col=target_col,
                    tau=tau,
                )
                rows.append(
                    {
                        "fold": fold["name"],
                        "tau": tau,
                        "avg_profit": metrics["avg_profit"],
                        "oracle_avg_profit": metrics["oracle_avg_profit"],
                        "oracle_ratio": metrics["oracle_ratio"],
                        "trade_days": metrics["trade_days"],
                        "valid_days": metrics["valid_days"],
                        "trade_days_ratio": metrics["trade_days_ratio"],
                    }
                )

        by_fold = pd.DataFrame(rows)
        summary = (
            by_fold.groupby("tau", as_index=False)
            .agg(
                avg_profit_mean=("avg_profit", "mean"),
                avg_profit_min=("avg_profit", "min"),
                oracle_ratio_mean=("oracle_ratio", "mean"),
                oracle_ratio_min=("oracle_ratio", "min"),
                trade_days_ratio_mean=("trade_days_ratio", "mean"),
            )
            .sort_values(["avg_profit_mean", "avg_profit_min"], ascending=False)
        )
        detail_path = reports_dir / f"tau_search_{cfg['model']['name']}_by_fold.csv"
        summary_path = reports_dir / f"tau_search_{cfg['model']['name']}.csv"
        by_fold.to_csv(detail_path, index=False)
        summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(detail_path))
        if not summary.empty:
            best = summary.iloc[0]
            mlflow.log_metric("best_tau", float(best["tau"]))
            mlflow.log_metric("best_avg_profit_mean", float(best["avg_profit_mean"]))
            mlflow.log_metric("best_oracle_ratio_mean", float(best["oracle_ratio_mean"]))

    print(summary.to_string(index=False))
    print(f"summary_path={summary_path}")
    print(f"detail_path={detail_path}")
    return summary_path


def _model_spec_frame(
    df: pd.DataFrame,
    spec: dict,
    *,
    time_col: str,
    target_col: str,
) -> pd.DataFrame:
    build_spec = segmented_feature_build_spec(spec)
    return _add_target_variants(
        build_feature_frame(df, build_spec, time_col=time_col),
        time_col=time_col,
        target_col=target_col,
    )


def _predict_member_fold(
    *,
    spec: dict,
    frame: pd.DataFrame,
    feature_cols: list[str],
    train_target_col: str,
    time_col: str,
    fold: dict,
) -> tuple[np.ndarray, pd.DataFrame, str, int]:
    train_window_days = spec["model"].get("train_window_days")
    train_end = pd.Timestamp(fold["train_end"])
    valid_start = pd.Timestamp(fold["valid_start"])
    valid_end = pd.Timestamp(fold["valid_end"])

    train_df = _fold_train_frame(
        frame,
        time_col=time_col,
        train_end=train_end,
        train_window_days=train_window_days,
    )
    valid_df = frame[(frame[time_col] >= valid_start) & (frame[time_col] <= valid_end)].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(
            f"empty train/valid split for fold={fold['name']} member={spec['model']['name']}"
        )

    actual_train_start = train_df[time_col].min()
    actual_train_end = train_df[time_col].max()
    train_range = (
        f"{spec['model']['name']}:{actual_train_start:%Y-%m-%d}.."
        f"{actual_train_end:%Y-%m-%d}"
    )

    boundaries = spec["model"].get("segment_boundaries")
    if boundaries is not None:
        valid_df = valid_df.reset_index(drop=True)
        boundary_values = segment_boundaries(spec)
        feature_cols_by_segment = segmented_feature_columns(spec, boundary_values)
        train_df = add_slot_and_segment(
            train_df,
            time_col=time_col,
            boundaries=boundary_values,
        )
        valid_df = add_slot_and_segment(
            valid_df,
            time_col=time_col,
            boundaries=boundary_values,
        )
        preds, best_iterations, _ = predict_segmented_fold(
            train_df,
            valid_df,
            feature_cols_by_segment=feature_cols_by_segment,
            target_col=train_target_col,
            cfg=spec,
            boundaries=boundary_values,
        )
        best_iter = int(np.mean(best_iterations))
        return preds, valid_df, train_range, best_iter

    model = train_model(
        train_df,
        valid_df,
        feature_cols=feature_cols,
        target_col=train_target_col,
        cfg=spec,
    )
    preds = predict_model(model, valid_df, feature_cols, spec)
    return preds, valid_df, train_range, best_iteration(model, spec)


def run_ensemble_backtest(cfg: dict, *, config_path: str) -> Path:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    specs = cfg["ensemble"]["members"]
    if len(specs) < 2:
        raise ValueError("ensemble requires at least two members")
    weights = np.array([float(spec.get("weight", 1.0)) for spec in specs], dtype=float)
    weights = weights / weights.sum()

    raw_df = load_train_frame(cfg)
    member_frames = [
        _model_spec_frame(raw_df, spec, time_col=time_col, target_col=target_col)
        for spec in specs
    ]
    member_feature_cols = [feature_columns(spec) for spec in specs]
    member_target_cols = [
        _target_col_for_mode(spec["model"].get("target_mode", "absolute")) for spec in specs
    ]

    mlflow.set_tracking_uri(str(Path(cfg["paths"]["mlruns_dir"]).resolve()))
    mlflow.set_experiment(cfg["model"]["name"])

    fold_results: list[FoldResult] = []
    all_day_rows: list[dict] = []

    with mlflow.start_run(run_name=f"{cfg['model']['name']}_rolling"):
        mlflow.log_param("config", config_path)
        mlflow.log_param("model", cfg["model"]["name"])
        mlflow.log_param("ensemble_members", ",".join(spec["model"]["name"] for spec in specs))
        mlflow.log_param("ensemble_weights", ",".join(str(w) for w in weights))

        for fold in cfg["folds"]:
            print(f"running_fold={fold['name']}", flush=True)
            preds: list[np.ndarray] = []
            train_ranges: list[str] = []
            best_iterations: list[int] = []
            valid_df_for_eval: pd.DataFrame | None = None

            for spec, frame, feature_cols, train_target_col in zip(
                specs,
                member_frames,
                member_feature_cols,
                member_target_cols,
                strict=True,
            ):
                pred, valid_df, train_range, best_iter = _predict_member_fold(
                    spec=spec,
                    frame=frame,
                    feature_cols=feature_cols,
                    train_target_col=train_target_col,
                    time_col=time_col,
                    fold=fold,
                )
                preds.append(pred)
                train_ranges.append(train_range)
                best_iterations.append(best_iter)
                if valid_df_for_eval is None:
                    valid_df_for_eval = valid_df[[time_col, target_col]].copy()

            if valid_df_for_eval is None:
                raise ValueError(f"no validation frame for fold={fold['name']}")

            print("train_ranges=" + ";".join(train_ranges), flush=True)
            stacked_preds = np.vstack(preds)
            valid_df_for_eval["pred"] = np.average(stacked_preds, axis=0, weights=weights)

            y_true = valid_df_for_eval[target_col].to_numpy()
            y_pred = valid_df_for_eval["pred"].to_numpy()
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            curve_z_rmse = _daily_curve_z_rmse(
                valid_df_for_eval,
                time_col=time_col,
                target_col=target_col,
            )
            dispatch_metrics = _evaluate_dispatch(
                valid_df_for_eval,
                time_col=time_col,
                target_col=target_col,
                tau=float(cfg["dispatch"]["tau"]),
            )

            result = FoldResult(
                fold=fold["name"],
                train_rows=0,
                valid_rows=len(valid_df_for_eval),
                valid_days=dispatch_metrics["valid_days"],
                train_window_days=None,
                train_start=";".join(train_ranges),
                train_end=";".join(train_ranges),
                target_mode="ensemble",
                mae=mae,
                rmse=rmse,
                curve_z_rmse=curve_z_rmse,
                mean_abs_charge_gap=dispatch_metrics["mean_abs_charge_gap"],
                mean_abs_discharge_gap=dispatch_metrics["mean_abs_discharge_gap"],
                avg_profit=dispatch_metrics["avg_profit"],
                oracle_avg_profit=dispatch_metrics["oracle_avg_profit"],
                oracle_ratio=dispatch_metrics["oracle_ratio"],
                avg_regret=dispatch_metrics["avg_regret"],
                loss_days=dispatch_metrics["loss_days"],
                trade_days=dispatch_metrics["trade_days"],
                trade_days_ratio=dispatch_metrics["trade_days_ratio"],
                best_iteration=int(np.mean(best_iterations)),
            )
            fold_results.append(result)
            for row in dispatch_metrics["day_rows"]:
                row["fold"] = fold["name"]
                all_day_rows.append(row)

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
