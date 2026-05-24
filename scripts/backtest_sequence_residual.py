from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

from electricity.data import load_train_frame
from electricity.dispatch import optimize_day
from electricity.eval.backtest import (
    _fold_bounds,
    _fold_train_frame,
    _maybe_build_prior,
    _target_col_for_mode,
)
from electricity.eval.metrics import daily_profit
from electricity.eval.segmented_utils import (
    add_slot_and_segment,
    predict_segmented_fold,
    segment_boundaries,
    segmented_feature_build_spec,
    segmented_feature_columns,
)
from electricity.features import (
    apply_fold_bias_correction,
    apply_fold_weather_correction,
    build_feature_frame,
    feature_columns,
)
from electricity.models import predict_model, train_model
from electricity.models.sequence import (
    build_mlp_residual_model,
    flatten_daily_features,
    make_daily_sequence_frame,
)

REPORTS_DIR = Path("reports")
DEFAULT_CONFIG = Path("configs/sequence_mlp_residual_5fold.yaml")


@dataclass(frozen=True)
class FoldPrediction:
    fold: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    train_df: pd.DataFrame
    valid_df: pd.DataFrame


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _add_target_variants(df: pd.DataFrame, *, time_col: str, target_col: str) -> pd.DataFrame:
    out = df.copy()
    date = out[time_col].dt.normalize()
    daily_mean = out.groupby(date)[target_col].transform("mean")
    daily_std = out.groupby(date)[target_col].transform("std").replace(0.0, np.nan)
    out["target_absolute"] = out[target_col]
    out["target_centered"] = out[target_col] - daily_mean
    out["target_zscore"] = (out[target_col] - daily_mean) / daily_std.fillna(1.0)
    return out


def _prepare_ensemble_specs(cfg: dict) -> list[dict]:
    specs = []
    for raw_spec in cfg["ensemble"]["members"]:
        spec = dict(raw_spec)
        spec.setdefault("paths", cfg["paths"])
        if "sample_weighting" in cfg:
            spec.setdefault("sample_weighting", cfg["sample_weighting"])
        if "weekly_relative_features" in cfg:
            spec.setdefault("weekly_relative_features", cfg["weekly_relative_features"])
        if "forecast_error_augmentation" in cfg:
            spec.setdefault("forecast_error_augmentation", cfg["forecast_error_augmentation"])
        specs.append(spec)
    return specs


def _model_spec_frame(
    raw_df: pd.DataFrame,
    spec: dict,
    *,
    time_col: str,
    target_col: str,
) -> pd.DataFrame:
    frame = build_feature_frame(raw_df, spec, time_col=time_col)
    return _add_target_variants(frame, time_col=time_col, target_col=target_col)


def _predict_single_member(
    spec: dict,
    raw_df: pd.DataFrame,
    fold: dict,
    *,
    time_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_mode = spec["model"].get("target_mode", "absolute")
    train_target_col = _target_col_for_mode(target_mode)
    train_window_days = spec["model"].get("train_window_days")
    boundaries = spec["model"].get("segment_boundaries")
    build_spec = segmented_feature_build_spec(spec)
    cols = feature_columns(build_spec)

    frame = _model_spec_frame(raw_df, spec, time_col=time_col, target_col=target_col)
    if boundaries is not None:
        boundary_values = segment_boundaries(spec)
        frame = add_slot_and_segment(frame, time_col=time_col, boundaries=boundary_values)
        feature_cols_by_segment = segmented_feature_columns(spec, boundary_values)

    bounds = _fold_bounds(fold)
    fold_frame = apply_fold_bias_correction(
        frame,
        spec,
        time_col=time_col,
        train_end=bounds.train_end,
        train_start=bounds.train_start,
        valid_start=bounds.valid_start,
        valid_end=bounds.valid_end,
    )
    fold_frame = apply_fold_weather_correction(
        fold_frame,
        spec,
        time_col=time_col,
        train_end=bounds.train_end,
        train_start=bounds.train_start,
        valid_start=bounds.valid_start,
        valid_end=bounds.valid_end,
    )
    train_df = _fold_train_frame(
        fold_frame,
        time_col=time_col,
        train_end=bounds.train_end,
        train_window_days=train_window_days,
        train_start=bounds.train_start,
        valid_start=bounds.valid_start,
        valid_end=bounds.valid_end,
    )
    valid_df = fold_frame[
        (fold_frame[time_col] >= bounds.valid_start) & (fold_frame[time_col] <= bounds.valid_end)
    ].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(f"empty train/valid split for fold={fold['name']}")

    if boundaries is not None:
        preds, _, _ = predict_segmented_fold(
            train_df,
            valid_df.reset_index(drop=True),
            feature_cols_by_segment=feature_cols_by_segment,
            target_col=train_target_col,
            cfg=spec,
            boundaries=boundary_values,
        )
    else:
        model = train_model(
            train_df,
            valid_df,
            feature_cols=cols,
            target_col=train_target_col,
            cfg=spec,
        )
        preds = predict_model(model, valid_df, cols, spec)
    out = valid_df[[time_col, target_col]].copy()
    out["pred"] = preds
    return train_df[[time_col, target_col]].copy(), out


def build_anchor_fold_predictions(cfg: dict) -> list[FoldPrediction]:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    raw_df = load_train_frame(cfg)
    specs = _prepare_ensemble_specs(cfg)
    weights = np.asarray([float(spec.get("weight", 1.0)) for spec in specs], dtype=float)
    weights = weights / weights.sum()

    raw_labels = raw_df[[time_col, target_col]].copy()
    rows: list[FoldPrediction] = []
    for fold in cfg["folds"]:
        bounds = _fold_bounds(fold)
        member_valids: list[pd.DataFrame] = []
        train_mask = raw_labels[time_col] <= bounds.train_end
        if bounds.train_start is not None:
            train_mask &= raw_labels[time_col] >= bounds.train_start
        if bounds.valid_start is not None and bounds.valid_end is not None:
            train_mask &= (raw_labels[time_col] < bounds.valid_start) | (
                raw_labels[time_col] > bounds.valid_end
            )
        anchor_train = raw_labels[train_mask].copy()

        for spec in specs:
            _, valid_member = _predict_single_member(
                spec,
                raw_df,
                fold,
                time_col=time_col,
                target_col=target_col,
            )
            member_valids.append(valid_member)

        valid = member_valids[0][[time_col, target_col]].copy()
        stacked = np.vstack([frame["pred"].to_numpy(dtype=float) for frame in member_valids])
        valid["anchor_pred"] = np.average(stacked, axis=0, weights=weights)
        rows.append(
            FoldPrediction(
                fold=str(fold["name"]),
                train_start=anchor_train[time_col].min(),
                train_end=anchor_train[time_col].max(),
                valid_start=bounds.valid_start,
                valid_end=bounds.valid_end,
                train_df=anchor_train,
                valid_df=valid,
            )
        )
    return rows


def add_sequence_features(df: pd.DataFrame, cfg: dict, *, time_col: str) -> pd.DataFrame:
    feature_cfg = cfg.get("sequence_residual", {}).get("feature_sets", {})
    spec = {
        **cfg,
        "feature_sets": {
            "derived": bool(feature_cfg.get("business", True)),
            "bid_space": bool(feature_cfg.get("bid_space", True)),
        },
    }
    return build_feature_frame(df, spec, time_col=time_col)


def sequence_feature_columns(cfg: dict) -> list[str]:
    base_cols = list(cfg["data"]["feature_cols"])
    cols = [
        *base_cols,
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
    feature_cfg = cfg.get("sequence_residual", {}).get("feature_sets", {})
    if feature_cfg.get("bid_space", True):
        cols.append("bid_space")
    if feature_cfg.get("business", True):
        cols += [
            "net_load",
            "renewable_ratio",
            "wind_ratio",
            "solar_ratio",
            "hydro_ratio",
            "tie_line_ratio",
            "non_market_ratio",
            "wind_solar_balance",
            "net_load_day_dev",
            "renewable_ratio_day_dev",
            "net_load_day_rank_pct",
            "renewable_ratio_day_rank_pct",
        ]
    return cols


def _daily_curve_z_rmse(
    df: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    pred_col: str,
) -> float:
    work = df[[time_col, target_col, pred_col]].copy()
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
        pred = group[pred_col].to_numpy(dtype=float)
        true_std = np.std(true)
        pred_std = np.std(pred)
        if true_std == 0 or pred_std == 0:
            continue
        true_z = (true - np.mean(true)) / true_std
        pred_z = (pred - np.mean(pred)) / pred_std
        errors.append(float(np.mean((true_z - pred_z) ** 2)))
    return float(np.sqrt(np.mean(errors))) if errors else 0.0


def evaluate_predictions(
    df: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    pred_col: str,
    cfg: dict,
    train_labels: pd.DataFrame,
    fold: str,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> tuple[dict, list[dict]]:
    work = df[[time_col, target_col, pred_col]].copy()
    work["date"] = work[time_col].dt.normalize()
    work["slot"] = work[time_col].dt.hour * 4 + work[time_col].dt.minute // 15
    log_pc, log_pd, lc_w, ld_w = _maybe_build_prior(
        cfg,
        train_labels,
        time_col=time_col,
        target_col=target_col,
        train_end=train_end,
        train_start=train_start,
        valid_start=valid_start,
        valid_end=valid_end,
    )

    profits: list[float] = []
    oracle_profits: list[float] = []
    regrets: list[float] = []
    abs_charge_gaps: list[float] = []
    abs_discharge_gaps: list[float] = []
    loss_days = 0
    day_rows: list[dict] = []
    for date, group in work.groupby("date", sort=True):
        if len(group) != 96:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            continue
        true_prices = group[target_col].to_numpy(dtype=float)
        pred_prices = group[pred_col].to_numpy(dtype=float)
        result = optimize_day(
            pred_prices,
            tau=float(cfg["dispatch"]["tau"]),
            log_prior_charge=log_pc,
            log_prior_discharge=log_pd,
            lambda_charge=lc_w,
            lambda_discharge=ld_w,
        )
        oracle = optimize_day(true_prices, tau=0.0)
        profit = daily_profit(true_prices, result.power)
        oracle_profit = daily_profit(true_prices, oracle.power)
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
        day_rows.append(
            {
                "date": date.date().isoformat(),
                "fold": fold,
                "profit": profit,
                "oracle_profit": oracle_profit,
                "traded": int(np.count_nonzero(result.power) > 0),
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

    y_true = work[target_col].to_numpy(dtype=float)
    y_pred = work[pred_col].to_numpy(dtype=float)
    avg_profit = float(np.mean(profits)) if profits else 0.0
    oracle_avg = float(np.mean(oracle_profits)) if oracle_profits else 0.0
    summary = {
        "fold": fold,
        "valid_rows": len(work),
        "valid_days": len(profits),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "curve_z_rmse": _daily_curve_z_rmse(
            work.rename(columns={pred_col: "pred"}),
            time_col=time_col,
            target_col=target_col,
            pred_col="pred",
        ),
        "mean_abs_charge_gap": float(np.mean(abs_charge_gaps)) if abs_charge_gaps else 0.0,
        "mean_abs_discharge_gap": (
            float(np.mean(abs_discharge_gaps)) if abs_discharge_gaps else 0.0
        ),
        "avg_profit": avg_profit,
        "oracle_avg_profit": oracle_avg,
        "oracle_ratio": avg_profit / oracle_avg if oracle_avg > 0 else 0.0,
        "avg_regret": float(np.mean(regrets)) if regrets else 0.0,
        "loss_days": loss_days,
    }
    return summary, day_rows


def fold_name_for_train_day(date: pd.Timestamp, cfg: dict) -> str | None:
    for fold in cfg["folds"]:
        bounds = _fold_bounds(fold)
        if bounds.valid_start <= date <= bounds.valid_end:
            return str(fold["name"])
    return None


def write_report(
    *,
    cfg: dict,
    overall: pd.DataFrame,
    by_fold: pd.DataFrame,
    blend_summary: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    model_name = cfg["sequence_residual"]["name"]
    best = overall.iloc[0]
    lines = [
        "# Sequence Residual Experiment",
        "",
        "## Scope",
        "",
        f"- Model: `{model_name}`.",
        "- Goal: learn residuals on top of current champion predictions, not replace champion.",
        "- Implementation: daily sequence MLP using sklearn, no PyTorch/GPU dependency.",
        "- Input: 96-point daily curves from official forecast features plus safe derived features.",
        "- Target: `true_price - anchor_pred`.",
        "- Fusion: `final_pred = anchor_pred + weight * residual_pred`.",
        "- Validation: fold-safe 5-fold; each fold scaler/model is fitted only on its train period.",
        "",
        "## Overall Weight Search",
        "",
        markdown_table(overall),
        "",
        "## Fold Summary",
        "",
        markdown_table(by_fold, limit=60),
        "",
        "## Residual Fit Summary",
        "",
        markdown_table(blend_summary),
        "",
        "## Decision",
        "",
    ]
    delta = float(best["profit_delta_vs_anchor"])
    loss_delta = float(best["loss_day_delta_vs_anchor"])
    if delta > 100 and loss_delta <= 0:
        lines += [
            "- Clears the local promotion threshold.",
            "- Next: add test-like risk view and consider submit-time implementation.",
        ]
    elif delta > 30 and loss_delta <= 0:
        lines += [
            "- Positive diagnostic signal, but below submit threshold.",
            "- Next: try a small TCN or gated residual before considering submission.",
        ]
    elif delta > 0:
        lines += [
            "- Tiny positive signal, but risk metrics are not strong enough.",
            "- Keep as diagnostic only.",
        ]
    else:
        lines += [
            "- Does not beat anchor after fold-safe blending.",
            "- Reject this standalone MLP residual configuration.",
        ]
    lines += [
        "",
        "## Artifacts",
        "",
        *[f"- `{path}`" for path in paths.values()],
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame, *, limit: int = 30, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_无数据_"
    block = df.head(limit).copy()
    cols = list(block.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in block.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float | np.floating):
                cells.append("" if pd.isna(value) else format(float(value), floatfmt))
            else:
                cells.append("" if pd.isna(value) else str(value).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    model_name = cfg["sequence_residual"]["name"]
    REPORTS_DIR.mkdir(exist_ok=True)

    print("building_anchor_predictions", flush=True)
    fold_predictions = build_anchor_fold_predictions(cfg)
    raw_df = load_train_frame(cfg)
    feature_df = add_sequence_features(raw_df, cfg, time_col=time_col)
    feature_cols = sequence_feature_columns(cfg)
    missing = sorted(set(feature_cols) - set(feature_df.columns))
    if missing:
        raise ValueError(f"missing sequence features: {missing}")

    all_fold_frames: list[pd.DataFrame] = []
    residual_fit_rows: list[dict] = []
    train_labels = raw_df[[time_col, target_col]].copy()
    weights = [float(x) for x in cfg["sequence_residual"].get("blend_weights", [0.0])]

    for fp in fold_predictions:
        print(f"sequence_fold={fp.fold}", flush=True)
        train = fp.train_df.merge(feature_df[[time_col, *feature_cols]], on=time_col, how="left")
        train["anchor_pred"] = 0.0
        train["residual"] = 0.0

        # Use available out-of-fold anchor predictions for train days when possible.
        oof = pd.concat(
            [
                item.valid_df[[time_col, "anchor_pred"]].assign(
                    residual=lambda d, item=item: item.valid_df[target_col].to_numpy(dtype=float)
                    - d["anchor_pred"].to_numpy(dtype=float)
                )
                for item in fold_predictions
                if item.fold != fp.fold
            ],
            ignore_index=True,
        )
        train = train.drop(columns=["anchor_pred", "residual"]).merge(oof, on=time_col, how="inner")
        train_seq = make_daily_sequence_frame(
            train,
            time_col=time_col,
            target_col=target_col,
            base_pred_col="anchor_pred",
            residual_col="residual",
            feature_cols=feature_cols,
        )

        valid = fp.valid_df.merge(feature_df[[time_col, *feature_cols]], on=time_col, how="left")
        valid["residual"] = valid[target_col] - valid["anchor_pred"]
        valid_seq = make_daily_sequence_frame(
            valid,
            time_col=time_col,
            target_col=target_col,
            base_pred_col="anchor_pred",
            residual_col="residual",
            feature_cols=feature_cols,
        )
        if train_seq.features.shape[0] == 0 or valid_seq.features.shape[0] == 0:
            raise ValueError(f"empty sequence train/valid for fold={fp.fold}")

        mlp_cfg = cfg["sequence_residual"].get("mlp", {})
        model = build_mlp_residual_model(
            hidden_layer_sizes=tuple(int(x) for x in mlp_cfg.get("hidden_layer_sizes", [64, 32])),
            alpha=float(mlp_cfg.get("alpha", 0.01)),
            learning_rate_init=float(mlp_cfg.get("learning_rate_init", 0.001)),
            max_iter=int(mlp_cfg.get("max_iter", 1500)),
            random_state=int(mlp_cfg.get("random_state", 2026)),
        )
        x_train = flatten_daily_features(train_seq.features)
        y_train = train_seq.residual
        x_valid = flatten_daily_features(valid_seq.features)
        model.fit(x_train, y_train)
        valid_resid_pred = model.predict(x_valid)
        train_resid_pred = model.predict(x_train)
        residual_fit_rows.append(
            {
                "fold": fp.fold,
                "train_days": train_seq.features.shape[0],
                "valid_days": valid_seq.features.shape[0],
                "train_residual_rmse": float(
                    np.sqrt(np.mean((train_resid_pred - y_train) ** 2))
                ),
                "valid_residual_rmse": float(
                    np.sqrt(np.mean((valid_resid_pred - valid_seq.residual) ** 2))
                ),
                "valid_residual_mae": float(
                    np.mean(np.abs(valid_resid_pred - valid_seq.residual))
                ),
            }
        )

        complete_times = np.concatenate(valid_seq.times)
        complete_index = pd.DataFrame({time_col: complete_times})
        valid_out = complete_index.merge(
            valid[[time_col, target_col, "anchor_pred"]],
            on=time_col,
            how="left",
        )
        valid_out = valid_out.sort_values(time_col).reset_index(drop=True)
        valid_out["fold"] = fp.fold
        valid_out["residual_pred"] = valid_resid_pred.reshape(-1)
        valid_out["residual_true"] = valid_out[target_col] - valid_out["anchor_pred"]
        valid_out["train_start"] = fp.train_start
        valid_out["train_end"] = fp.train_end
        valid_out["valid_start"] = fp.valid_start
        valid_out["valid_end"] = fp.valid_end
        all_fold_frames.append(valid_out)

    predictions = pd.concat(all_fold_frames, ignore_index=True)
    residual_fit = pd.DataFrame(residual_fit_rows)
    prediction_path = REPORTS_DIR / f"predictions_{model_name}.csv"
    residual_fit_path = REPORTS_DIR / f"{model_name}_residual_fit.csv"
    predictions.to_csv(prediction_path, index=False)
    residual_fit.to_csv(residual_fit_path, index=False)

    summary_rows: list[dict] = []
    day_rows_all: list[dict] = []
    for weight in weights:
        pred_col = f"pred_w{weight:g}".replace(".", "p")
        predictions[pred_col] = predictions["anchor_pred"] + weight * predictions["residual_pred"]
        for fp in fold_predictions:
            subset = predictions[predictions["fold"] == fp.fold].copy()
            summary, day_rows = evaluate_predictions(
                subset,
                time_col=time_col,
                target_col=target_col,
                pred_col=pred_col,
                cfg=cfg,
                train_labels=train_labels,
                fold=fp.fold,
                train_start=fp.train_start,
                train_end=fp.train_end,
                valid_start=fp.valid_start,
                valid_end=fp.valid_end,
            )
            summary["blend_weight"] = weight
            summary_rows.append(summary)
            for row in day_rows:
                row["blend_weight"] = weight
                day_rows_all.append(row)

    by_fold = pd.DataFrame(summary_rows)
    daily = pd.DataFrame(day_rows_all)
    anchor = by_fold[by_fold["blend_weight"] == 0.0][
        ["fold", "avg_profit", "loss_days", "avg_regret"]
    ].rename(
        columns={
            "avg_profit": "anchor_avg_profit",
            "loss_days": "anchor_loss_days",
            "avg_regret": "anchor_avg_regret",
        }
    )
    by_fold = by_fold.merge(anchor, on="fold", how="left")
    by_fold["profit_delta_vs_anchor"] = by_fold["avg_profit"] - by_fold["anchor_avg_profit"]
    by_fold["loss_day_delta_vs_anchor"] = by_fold["loss_days"] - by_fold["anchor_loss_days"]
    by_fold["regret_delta_vs_anchor"] = by_fold["avg_regret"] - by_fold["anchor_avg_regret"]

    overall = (
        by_fold.groupby("blend_weight", as_index=False)
        .agg(
            mean_profit=("avg_profit", "mean"),
            min_profit=("avg_profit", "min"),
            mean_rmse=("rmse", "mean"),
            mean_curve_z_rmse=("curve_z_rmse", "mean"),
            loss_days=("loss_days", "sum"),
            mean_regret=("avg_regret", "mean"),
            mean_charge_gap=("mean_abs_charge_gap", "mean"),
            mean_discharge_gap=("mean_abs_discharge_gap", "mean"),
            profit_delta_vs_anchor=("profit_delta_vs_anchor", "mean"),
            loss_day_delta_vs_anchor=("loss_day_delta_vs_anchor", "sum"),
            regret_delta_vs_anchor=("regret_delta_vs_anchor", "mean"),
        )
        .sort_values(["mean_profit", "min_profit"], ascending=False)
    )

    summary_path = REPORTS_DIR / f"backtest_{model_name}.csv"
    daily_path = REPORTS_DIR / f"backtest_{model_name}_daily.csv"
    overall_path = REPORTS_DIR / f"{model_name}_overall.csv"
    report_path = REPORTS_DIR / f"{model_name}_experiment.md"
    by_fold.to_csv(summary_path, index=False)
    daily.to_csv(daily_path, index=False)
    overall.to_csv(overall_path, index=False)
    paths = {
        "summary": summary_path,
        "daily": daily_path,
        "overall": overall_path,
        "predictions": prediction_path,
        "residual_fit": residual_fit_path,
        "report": report_path,
    }
    write_report(
        cfg=cfg,
        overall=overall,
        by_fold=by_fold.sort_values(["blend_weight", "fold"]),
        blend_summary=residual_fit,
        paths=paths,
    )
    print(overall.to_string(index=False))
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
