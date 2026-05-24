from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

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
from electricity.models.sequence import make_daily_sequence_frame, torch_available

if torch_available():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from electricity.models.sequence import SmallTFTLikeResidual
else:  # pragma: no cover - checked at runtime.
    torch = None
    SmallTFTLikeResidual = None


DEFAULT_CONFIG = Path("configs/sequence_tftlike_residual_5fold.yaml")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.set_num_threads(1)


def standardize_daily_features(
    train_features: np.ndarray,
    valid_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_flat = train_features.reshape(-1, train_features.shape[-1])
    valid_flat = valid_features.reshape(-1, valid_features.shape[-1])
    scaler.fit(train_flat)
    train_scaled = scaler.transform(train_flat).reshape(train_features.shape)
    valid_scaled = scaler.transform(valid_flat).reshape(valid_features.shape)
    return train_scaled.astype(np.float32), valid_scaled.astype(np.float32)


def train_tftlike_model(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, dict[str, float]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for TFT-like residual experiment")

    hidden_size = int(cfg.get("hidden_size", 16))
    num_heads = int(cfg.get("num_heads", 1))
    dropout = float(cfg.get("dropout", 0.3))
    lr = float(cfg.get("learning_rate", 0.001))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    max_epochs = int(cfg.get("max_epochs", 120))
    patience = int(cfg.get("patience", 20))
    batch_size = int(cfg.get("batch_size", 16))
    seed = int(cfg.get("random_state", 2026))
    set_seed(seed)

    model = SmallTFTLikeResidual(
        input_size=train_x.shape[-1],
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_x_t = torch.from_numpy(valid_x)
    valid_y_t = torch.from_numpy(valid_y.astype(np.float32))

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_pred = model(valid_x_t)
            valid_loss = float(loss_fn(valid_pred, valid_y_t).item())
        if valid_loss + 1e-6 < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(valid_x_t).cpu().numpy()
        train_pred = model(torch.from_numpy(train_x)).cpu().numpy()
    fit = {
        "best_epoch": float(best_epoch),
        "best_valid_smooth_l1": float(best_loss),
        "train_residual_rmse": float(np.sqrt(mean_squared_error(train_y.reshape(-1), train_pred.reshape(-1)))),
        "valid_residual_rmse": float(np.sqrt(mean_squared_error(valid_y.reshape(-1), pred.reshape(-1)))),
        "valid_residual_mae": float(mean_absolute_error(valid_y.reshape(-1), pred.reshape(-1))),
    }
    return pred, fit


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
        "# Sequence TFT-like Residual Experiment",
        "",
        "## Scope",
        "",
        f"- Model: `{model_name}`.",
        "- Small TFT-like residual model: GRU + single-head attention + gated fusion.",
        "- It predicts residuals on top of current champion, not standalone prices.",
        "- Inputs are only competition-visible next-day features and derived safe features.",
        "- Validation uses the same 5-fold setup and same champion dispatch prior.",
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
                "- Next: evaluate gated residual and online-risk dashboard before submit.",
            ]
        elif delta > 0 and loss_delta <= 0:
            lines += [
                f"- Positive but below submit threshold: best nonzero delta is {delta:.1f}/day.",
                "- Keep as diagnostic unless gated variant improves further.",
            ]
        else:
            lines += [
                "- Does not improve dispatch profit safely.",
                "- Reject this standalone TFT-like residual.",
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
    if not torch_available():
        raise RuntimeError("PyTorch is not installed. Install torch before running this script.")

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

    train_labels = raw_df[[time_col, target_col]].copy()
    weights = [float(x) for x in cfg["sequence_residual"].get("blend_weights", [0.0])]
    model_cfg = cfg["sequence_residual"].get("tft_like", {})

    all_fold_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    for fp in fold_predictions:
        print(f"tftlike_fold={fp.fold}", flush=True)
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

        train_x, valid_x = standardize_daily_features(train_seq.features, valid_seq.features)
        valid_pred, fit = train_tftlike_model(
            train_x=train_x,
            train_y=train_seq.residual.astype(np.float32),
            valid_x=valid_x,
            valid_y=valid_seq.residual.astype(np.float32),
            cfg=model_cfg,
        )
        fit_rows.append(
            {
                "fold": fp.fold,
                "train_days": train_seq.features.shape[0],
                "valid_days": valid_seq.features.shape[0],
                **fit,
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
        valid_out["residual_pred"] = valid_pred.reshape(-1)
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
