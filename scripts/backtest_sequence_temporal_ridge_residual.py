from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backtest_sequence_residual import (
    REPORTS_DIR,
    add_sequence_features,
    build_anchor_fold_predictions,
    evaluate_predictions,
    load_config,
    markdown_table,
    sequence_feature_columns,
)
from electricity.data import load_train_frame
from electricity.models.sequence import (
    build_temporal_ridge_residual_model,
    make_daily_sequence_frame,
    temporal_context_samples,
)


DEFAULT_CONFIG = Path("configs/sequence_temporal_ridge_residual_5fold.yaml")


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


def write_report(
    *,
    cfg: dict,
    overall: pd.DataFrame,
    by_fold: pd.DataFrame,
    residual_fit: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    model_name = cfg["sequence_residual"]["name"]
    best_nonzero = overall[overall["blend_weight"] > 0].head(1)
    lines = [
        "# Sequence Temporal Ridge Residual Experiment",
        "",
        "## Scope",
        "",
        f"- Model: `{model_name}`.",
        "- Goal: test a TCN-lite idea without adding PyTorch: local temporal context over the 96-point feature curve.",
        "- Implementation: slot-level ridge residual model with neighboring slots as features.",
        "- Anchor: current champion fold-safe OOF predictions.",
        "- Target: `true_price - anchor_pred`.",
        "- Fusion: `final_pred = anchor_pred + weight * residual_pred`.",
        "- Validation: same 5-fold setup and same champion dispatch prior.",
        "",
        "## Overall Weight Search",
        "",
        markdown_table(overall),
        "",
        "## Fold Summary",
        "",
        markdown_table(by_fold.sort_values(["blend_weight", "fold"]), limit=80),
        "",
        "## Residual Fit Summary",
        "",
        markdown_table(residual_fit),
        "",
        "## Decision",
        "",
    ]
    if best_nonzero.empty:
        lines.append("- No nonzero blend was evaluated.")
    else:
        best = best_nonzero.iloc[0]
        delta = float(best["profit_delta_vs_anchor"])
        loss_delta = int(best["loss_day_delta_vs_anchor"])
        if delta >= 100 and loss_delta <= 0:
            lines += [
                "- Clears the local promotion threshold.",
                "- Next: run online risk dashboard before considering submit generation.",
            ]
        elif delta > 0 and loss_delta <= 0:
            lines += [
                f"- Positive but below submit threshold: best nonzero delta is {delta:.1f}/day.",
                "- Keep as diagnostic; do not submit unless combined with another validated gate.",
            ]
        else:
            lines += [
                "- Does not improve the champion anchor safely.",
                "- Reject this standalone temporal residual.",
            ]
    lines += [
        "",
        "## Artifacts",
        "",
        *[f"- `{path}`" for path in paths.values()],
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        raise ValueError(f"missing temporal sequence features: {missing}")

    train_labels = raw_df[[time_col, target_col]].copy()
    ridge_cfg = cfg["sequence_residual"].get("temporal_ridge", {})
    alpha = float(ridge_cfg.get("alpha", 100.0))
    offsets = tuple(int(x) for x in ridge_cfg.get("offsets", [-8, -4, -2, -1, 0, 1, 2, 4, 8]))
    weights = [float(x) for x in cfg["sequence_residual"].get("blend_weights", [0.0])]

    all_fold_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    for fp in fold_predictions:
        print(f"temporal_ridge_fold={fp.fold}", flush=True)
        train = fp.train_df.merge(feature_df[[time_col, *feature_cols]], on=time_col, how="left")

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
        train = train.merge(oof, on=time_col, how="inner")
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

        model = build_temporal_ridge_residual_model(alpha=alpha)
        x_train = temporal_context_samples(train_seq.features, offsets=offsets)
        y_train = train_seq.residual.reshape(-1)
        x_valid = temporal_context_samples(valid_seq.features, offsets=offsets)
        y_valid = valid_seq.residual.reshape(-1)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        valid_pred = model.predict(x_valid)
        fit_rows.append(
            {
                "fold": fp.fold,
                "train_days": train_seq.features.shape[0],
                "valid_days": valid_seq.features.shape[0],
                "alpha": alpha,
                "offsets": ",".join(str(x) for x in offsets),
                "train_residual_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
                "valid_residual_rmse": float(np.sqrt(mean_squared_error(y_valid, valid_pred))),
                "valid_residual_mae": float(mean_absolute_error(y_valid, valid_pred)),
            }
        )

        complete_times = np.concatenate(valid_seq.times)
        valid_out = pd.DataFrame({time_col: complete_times}).merge(
            valid[[time_col, target_col, "anchor_pred"]],
            on=time_col,
            how="left",
        )
        valid_out = valid_out.sort_values(time_col).reset_index(drop=True)
        valid_out["fold"] = fp.fold
        valid_out["residual_pred"] = valid_pred
        valid_out["residual_true"] = valid_out[target_col] - valid_out["anchor_pred"]
        valid_out["train_start"] = fp.train_start
        valid_out["train_end"] = fp.train_end
        valid_out["valid_start"] = fp.valid_start
        valid_out["valid_end"] = fp.valid_end
        all_fold_frames.append(valid_out)

    predictions = pd.concat(all_fold_frames, ignore_index=True)
    residual_fit = pd.DataFrame(fit_rows)
    prediction_path = REPORTS_DIR / f"predictions_{model_name}.csv"
    residual_fit_path = REPORTS_DIR / f"{model_name}_residual_fit.csv"
    predictions.to_csv(prediction_path, index=False)
    residual_fit.to_csv(residual_fit_path, index=False)

    summary_rows: list[dict[str, object]] = []
    day_rows_all: list[dict[str, object]] = []
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
            summary["curve_z_rmse"] = _daily_curve_z_rmse(
                subset.assign(pred=predictions.loc[subset.index, pred_col].to_numpy(dtype=float)),
                time_col=time_col,
                target_col=target_col,
                pred_col="pred",
            )
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

    paths = {
        "summary": REPORTS_DIR / f"backtest_{model_name}.csv",
        "daily": REPORTS_DIR / f"backtest_{model_name}_daily.csv",
        "overall": REPORTS_DIR / f"{model_name}_overall.csv",
        "predictions": prediction_path,
        "residual_fit": residual_fit_path,
        "report": REPORTS_DIR / f"{model_name}_experiment.md",
    }
    by_fold.to_csv(paths["summary"], index=False)
    daily.to_csv(paths["daily"], index=False)
    overall.to_csv(paths["overall"], index=False)
    write_report(cfg=cfg, overall=overall, by_fold=by_fold, residual_fit=residual_fit, paths=paths)
    print(overall.to_string(index=False))
    print(f"report_path={paths['report']}")


if __name__ == "__main__":
    main()
