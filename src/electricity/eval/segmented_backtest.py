from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from electricity.data import load_train_frame
from electricity.eval.backtest import (
    FoldResult,
    _add_target_variants,
    _daily_curve_z_rmse,
    _evaluate_dispatch,
    _fold_train_frame,
    _target_col_for_mode,
)
from electricity.eval.segmented_utils import (
    add_slot_and_segment,
    predict_segmented_fold,
    segment_boundaries,
    segment_labels,
    segmented_feature_build_spec,
    segmented_feature_columns,
)
from electricity.features import (
    apply_fold_bias_correction,
    build_feature_frame,
    feature_columns,
)


def run_segmented_backtest(cfg: dict, *, config_path: str) -> Path:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    target_mode = cfg["model"].get("target_mode", "absolute")
    train_target_col = _target_col_for_mode(target_mode)
    train_window_days = cfg["model"].get("train_window_days")
    boundaries = segment_boundaries(cfg)
    build_spec = segmented_feature_build_spec(cfg)
    feature_cols = feature_columns(build_spec)
    feature_cols_by_segment = segmented_feature_columns(cfg, boundaries)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = add_slot_and_segment(
        _add_target_variants(
            build_feature_frame(load_train_frame(cfg), build_spec, time_col=time_col),
            time_col=time_col,
            target_col=target_col,
        ),
        time_col=time_col,
        boundaries=boundaries,
    )
    mlflow.set_tracking_uri(str(Path(cfg["paths"]["mlruns_dir"]).resolve()))
    mlflow.set_experiment(cfg["model"]["name"])

    fold_results: list[FoldResult] = []
    all_day_rows: list[dict] = []
    segment_rows: list[dict] = []

    with mlflow.start_run(run_name=f"{cfg['model']['name']}_rolling"):
        mlflow.log_param("config", config_path)
        mlflow.log_param("model", cfg["model"]["name"])
        mlflow.log_param("target_mode", target_mode)
        mlflow.log_param("train_window_days", train_window_days or "all_past")
        mlflow.log_param("segment_boundaries", ",".join(str(x) for x in boundaries))
        mlflow.log_param("feature_count", len(feature_cols))

        for fold in cfg["folds"]:
            print(f"running_fold={fold['name']}", flush=True)
            train_end = pd.Timestamp(fold["train_end"])
            valid_start = pd.Timestamp(fold["valid_start"])
            valid_end = pd.Timestamp(fold["valid_end"])

            fold_df = apply_fold_bias_correction(
                df, build_spec, time_col=time_col, train_end=train_end
            )
            train_df = _fold_train_frame(
                fold_df,
                time_col=time_col,
                train_end=train_end,
                train_window_days=train_window_days,
            )
            valid_df = fold_df[
                (fold_df[time_col] >= valid_start) & (fold_df[time_col] <= valid_end)
            ].copy()
            valid_df = valid_df.reset_index(drop=True)
            if train_df.empty or valid_df.empty:
                raise ValueError(f"empty train/valid split for fold={fold['name']}")
            actual_train_start = train_df[time_col].min()
            actual_train_end = train_df[time_col].max()
            print(
                "train_range="
                f"{actual_train_start.strftime('%Y-%m-%d')}.."
                f"{actual_train_end.strftime('%Y-%m-%d')} "
                f"segments={len(boundaries) - 1}",
                flush=True,
            )

            preds, best_iterations, train_rows_by_segment = predict_segmented_fold(
                train_df,
                valid_df,
                feature_cols_by_segment=feature_cols_by_segment,
                target_col=train_target_col,
                cfg=cfg,
                boundaries=boundaries,
            )
            valid_df["pred"] = preds
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
                target_mode=f"segmented_{target_mode}",
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
            for label, train_rows, iteration in zip(
                segment_labels(boundaries),
                train_rows_by_segment,
                best_iterations,
                strict=True,
            ):
                segment_rows.append(
                    {
                        "fold": fold["name"],
                        "segment": label,
                        "train_rows": train_rows,
                        "best_iteration": iteration,
                    }
                )

            for key in ["mae", "rmse", "curve_z_rmse", "avg_profit", "oracle_ratio"]:
                mlflow.log_metric(f"{fold['name']}_{key}", getattr(result, key))

        summary = pd.DataFrame([r.__dict__ for r in fold_results])
        day_detail = pd.DataFrame(all_day_rows)
        segment_detail = pd.DataFrame(segment_rows)
        summary_path = reports_dir / f"backtest_{cfg['model']['name']}.csv"
        detail_path = reports_dir / f"backtest_{cfg['model']['name']}_daily.csv"
        segment_path = reports_dir / f"backtest_{cfg['model']['name']}_segments.csv"
        summary.to_csv(summary_path, index=False)
        day_detail.to_csv(detail_path, index=False)
        segment_detail.to_csv(segment_path, index=False)
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(detail_path))
        mlflow.log_artifact(str(segment_path))

    print(summary.to_string(index=False))
    print(f"summary_path={summary_path}")
    print(f"daily_path={detail_path}")
    print(f"segment_path={segment_path}")
    return summary_path
