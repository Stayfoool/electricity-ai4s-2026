from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from electricity.data import load_train_frame
from electricity.dispatch import build_dispatch_prior, optimize_day
from electricity.eval.metrics import daily_profit
from electricity.features import build_feature_frame
from electricity.models.sequence import make_daily_sequence_frame, torch_available

if torch_available():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from electricity.models.sequence import SmallTCNPriceModel
else:  # pragma: no cover - checked at runtime.
    torch = None
    SmallTCNPriceModel = None


REPORTS_DIR = Path("reports")
DEFAULT_CONFIG = Path("configs/sequence_tcn_direct_price_5fold.yaml")
CHAMPION_SUMMARY = Path("reports/backtest_ens_champion_segmented6_prior_5fold.csv")


@dataclass(frozen=True)
class FoldBounds:
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    train_start: pd.Timestamp | None


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fold_bounds(fold: dict) -> FoldBounds:
    return FoldBounds(
        train_end=pd.Timestamp(fold["train_end"]),
        valid_start=pd.Timestamp(fold["valid_start"]),
        valid_end=pd.Timestamp(fold["valid_end"]),
        train_start=pd.Timestamp(fold["train_start"]) if fold.get("train_start") else None,
    )


def fold_train_frame(
    df: pd.DataFrame,
    *,
    time_col: str,
    train_end: pd.Timestamp,
    train_window_days: int | None,
    train_start: pd.Timestamp | None = None,
    valid_start: pd.Timestamp | None = None,
    valid_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    train_df = df[df[time_col] <= train_end].copy()
    if train_start is not None:
        train_df = train_df[train_df[time_col] >= train_start].copy()
    if train_window_days is not None:
        window_start = train_end - pd.Timedelta(days=train_window_days) + pd.Timedelta(seconds=1)
        train_df = train_df[train_df[time_col] >= window_start].copy()
    if valid_start is not None and valid_end is not None:
        outside = (train_df[time_col] < valid_start) | (train_df[time_col] > valid_end)
        train_df = train_df[outside].copy()
    return train_df


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


def add_sequence_features(df: pd.DataFrame, cfg: dict, *, time_col: str) -> pd.DataFrame:
    feature_cfg = cfg.get("sequence_direct", {}).get("feature_sets", {})
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
    feature_cfg = cfg.get("sequence_direct", {}).get("feature_sets", {})
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


def daily_curve_z_rmse(
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


def maybe_build_prior(
    cfg: dict,
    labels: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    train_end: pd.Timestamp,
    train_start: pd.Timestamp | None = None,
    valid_start: pd.Timestamp | None = None,
    valid_end: pd.Timestamp | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
    prior_cfg = cfg.get("dispatch", {}).get("prior")
    if not prior_cfg or not prior_cfg.get("enabled", False):
        return None, None, 0.0, 0.0
    prior_labels = labels[[time_col, target_col]]
    if train_start is not None:
        prior_labels = prior_labels[prior_labels[time_col] >= train_start]
    if valid_start is not None and valid_end is not None:
        outside = (prior_labels[time_col] < valid_start) | (prior_labels[time_col] > valid_end)
        prior_labels = prior_labels[outside]
    log_pc, log_pd = build_dispatch_prior(
        prior_labels,
        time_col=time_col,
        target_col=target_col,
        train_end=train_end,
        alpha=float(prior_cfg.get("alpha", 0.5)),
    )
    return (
        log_pc,
        log_pd,
        float(prior_cfg.get("lambda_charge", 0.0)),
        float(prior_cfg.get("lambda_discharge", 0.0)),
    )


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
    log_pc, log_pd, lc_w, ld_w = maybe_build_prior(
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
    return {
        "fold": fold,
        "valid_rows": len(work),
        "valid_days": len(profits),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "curve_z_rmse": daily_curve_z_rmse(
            work.rename(columns={pred_col: "pred"}),
            time_col=time_col,
            target_col=target_col,
            pred_col="pred",
        ),
        "mean_abs_charge_gap": float(np.mean(abs_charge_gaps)) if abs_charge_gaps else 0.0,
        "mean_abs_discharge_gap": float(np.mean(abs_discharge_gaps)) if abs_discharge_gaps else 0.0,
        "avg_profit": avg_profit,
        "oracle_avg_profit": oracle_avg,
        "oracle_ratio": avg_profit / oracle_avg if oracle_avg > 0 else 0.0,
        "avg_regret": float(np.mean(regrets)) if regrets else 0.0,
        "loss_days": loss_days,
    }, day_rows


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


def make_price_sequence(
    df: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    feature_cols: list[str],
):
    work = df[[time_col, target_col, *feature_cols]].copy()
    work["base_pred"] = 0.0
    work["residual"] = work[target_col]
    return make_daily_sequence_frame(
        work,
        time_col=time_col,
        target_col=target_col,
        base_pred_col="base_pred",
        residual_col="residual",
        feature_cols=feature_cols,
    )


def train_gru_price_model(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, dict[str, float]]:
    if torch is None or SmallTCNPriceModel is None:
        raise RuntimeError("PyTorch is required for TCN direct price experiment")

    seed = int(cfg.get("random_state", 2026))
    set_seed(seed)
    model = SmallTCNPriceModel(
        input_size=train_x.shape[-1],
        hidden_size=int(cfg.get("hidden_size", 32)),
        levels=int(cfg.get("levels", 3)),
        kernel_size=int(cfg.get("kernel_size", 5)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 0.001)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    loss_fn = torch.nn.SmoothL1Loss(beta=float(cfg.get("huber_beta", 0.5)))
    max_epochs = int(cfg.get("max_epochs", 160))
    patience = int(cfg.get("patience", 24))
    batch_size = int(cfg.get("batch_size", 16))

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
        "train_price_rmse": float(np.sqrt(mean_squared_error(train_y.reshape(-1), train_pred.reshape(-1)))),
        "valid_price_rmse": float(np.sqrt(mean_squared_error(valid_y.reshape(-1), pred.reshape(-1)))),
        "valid_price_mae": float(mean_absolute_error(valid_y.reshape(-1), pred.reshape(-1))),
    }
    return pred, fit


def transform_daily_target(targets: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "absolute":
        return targets
    daily_mean = targets.mean(axis=1, keepdims=True)
    centered = targets - daily_mean
    if target_mode == "centered":
        return centered
    if target_mode == "zscore":
        daily_std = targets.std(axis=1, keepdims=True)
        daily_std = np.where(daily_std < 1e-6, 1.0, daily_std)
        return centered / daily_std
    raise ValueError(f"unsupported sequence_direct.target_mode={target_mode}")


def summarize_overall(by_fold: pd.DataFrame) -> pd.DataFrame:
    model_name = str(by_fold["model"].iloc[0]) if "model" in by_fold.columns else "sequence_tcn"
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "mean_profit": float(by_fold["avg_profit"].mean()),
                "min_profit": float(by_fold["avg_profit"].min()),
                "mean_rmse": float(by_fold["rmse"].mean()),
                "mean_curve_z_rmse": float(by_fold["curve_z_rmse"].mean()),
                "loss_days": int(by_fold["loss_days"].sum()),
                "mean_regret": float(by_fold["avg_regret"].mean()),
                "mean_charge_gap": float(by_fold["mean_abs_charge_gap"].mean()),
                "mean_discharge_gap": float(by_fold["mean_abs_discharge_gap"].mean()),
            }
        ]
    )


def attach_champion_deltas(by_fold: pd.DataFrame) -> pd.DataFrame:
    if not CHAMPION_SUMMARY.exists():
        by_fold["profit_delta_vs_champion"] = np.nan
        by_fold["loss_day_delta_vs_champion"] = np.nan
        by_fold["regret_delta_vs_champion"] = np.nan
        return by_fold
    champion = pd.read_csv(CHAMPION_SUMMARY)[["fold", "avg_profit", "loss_days", "avg_regret"]].rename(
        columns={
            "avg_profit": "champion_avg_profit",
            "loss_days": "champion_loss_days",
            "avg_regret": "champion_avg_regret",
        }
    )
    out = by_fold.merge(champion, on="fold", how="left")
    out["profit_delta_vs_champion"] = out["avg_profit"] - out["champion_avg_profit"]
    out["loss_day_delta_vs_champion"] = out["loss_days"] - out["champion_loss_days"]
    out["regret_delta_vs_champion"] = out["avg_regret"] - out["champion_avg_regret"]
    return out


def write_report(
    *,
    cfg: dict,
    overall: pd.DataFrame,
    by_fold: pd.DataFrame,
    fit: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    model_name = cfg["sequence_direct"]["name"]
    target_mode = cfg["sequence_direct"].get("target_mode", "absolute")
    lines = [
        "# TCN Direct Price Experiment",
        "",
        "## Scope",
        "",
        f"- Model: `{model_name}`.",
        f"- Target mode: `{target_mode}`.",
        "- Directly predicts a 96-point daily curve used by dispatch.",
        "- Does not use champion predictions or true historical prices as lag inputs.",
        "- Inputs are official next-day forecast boundary features plus bid_space/business/time features.",
        "- Dispatch still uses the same brute-force legal pair enumeration and current prior.",
        "",
        "## Overall",
        "",
        markdown_table(overall),
        "",
        "## Fold Summary",
        "",
        markdown_table(by_fold, limit=80),
        "",
        "## Fit Summary",
        "",
        markdown_table(fit),
        "",
        "## Decision",
        "",
    ]
    if "profit_delta_vs_champion" in by_fold.columns and by_fold["profit_delta_vs_champion"].notna().any():
        delta = float(by_fold["profit_delta_vs_champion"].mean())
        loss_delta = int(by_fold["loss_day_delta_vs_champion"].sum())
        if delta >= 100 and loss_delta <= 0:
            lines += [
                "- Clears the local promotion threshold versus current champion.",
                "- Next: generate submit implementation and online-risk check before submitting.",
            ]
        elif delta > 0:
            lines += [
                f"- Positive but below promotion threshold: mean lift is {delta:.1f}/day.",
                "- Keep as candidate only unless a gated/ensemble version improves stability.",
            ]
        else:
            lines += [
                f"- Does not beat current champion: mean delta is {delta:.1f}/day.",
                "- Keep as deep-learning baseline only, not a submit candidate.",
            ]
    else:
        lines += ["- Champion summary was unavailable, so only standalone metrics were produced."]
    lines += ["", "## Artifacts", "", *[f"- `{path}`" for path in paths.values()]]
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
    model_name = cfg["sequence_direct"]["name"]
    target_mode = cfg["sequence_direct"].get("target_mode", "absolute")
    REPORTS_DIR.mkdir(exist_ok=True)

    raw_df = load_train_frame(cfg)
    feature_df = add_sequence_features(raw_df, cfg, time_col=time_col)
    feature_cols = sequence_feature_columns(cfg)
    missing = sorted(set(feature_cols) - set(feature_df.columns))
    if missing:
        raise ValueError(f"missing sequence features: {missing}")

    train_labels = raw_df[[time_col, target_col]].copy()
    model_cfg = cfg["sequence_direct"].get("gru", {})
    all_fold_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []

    for fold in cfg["folds"]:
        bounds = fold_bounds(fold)
        fold_name = str(fold["name"])
        print(f"tcn_direct_fold={fold_name}", flush=True)
        train_df = fold_train_frame(
            feature_df,
            time_col=time_col,
            train_end=bounds.train_end,
            train_window_days=cfg["sequence_direct"].get("train_window_days"),
            train_start=bounds.train_start,
            valid_start=bounds.valid_start,
            valid_end=bounds.valid_end,
        )
        valid_df = feature_df[
            (feature_df[time_col] >= bounds.valid_start) & (feature_df[time_col] <= bounds.valid_end)
        ].copy()
        train_df = train_df.dropna(subset=feature_cols + [target_col]).copy()
        valid_df = valid_df.dropna(subset=feature_cols + [target_col]).copy()
        if train_df.empty or valid_df.empty:
            raise ValueError(f"empty train/valid split for fold={fold_name}")

        train_seq = make_price_sequence(train_df, time_col=time_col, target_col=target_col, feature_cols=feature_cols)
        valid_seq = make_price_sequence(valid_df, time_col=time_col, target_col=target_col, feature_cols=feature_cols)
        if train_seq.features.shape[0] == 0 or valid_seq.features.shape[0] == 0:
            raise ValueError(f"empty daily sequence train/valid for fold={fold_name}")

        train_x, valid_x = standardize_daily_features(train_seq.features, valid_seq.features)
        train_y = transform_daily_target(train_seq.targets.astype(np.float32), target_mode)
        valid_y = transform_daily_target(valid_seq.targets.astype(np.float32), target_mode)
        pred, fit = train_gru_price_model(
            train_x=train_x,
            train_y=train_y.astype(np.float32),
            valid_x=valid_x,
            valid_y=valid_y.astype(np.float32),
            cfg=model_cfg,
        )
        fit_rows.append(
            {
                "fold": fold_name,
                "target_mode": target_mode,
                "train_days": train_seq.features.shape[0],
                "valid_days": valid_seq.features.shape[0],
                **fit,
            }
        )

        complete_times = np.concatenate(valid_seq.times)
        valid_out = pd.DataFrame({time_col: complete_times}).merge(
            valid_df[[time_col, target_col]], on=time_col, how="left"
        )
        valid_out = valid_out.sort_values(time_col).reset_index(drop=True)
        valid_out["fold"] = fold_name
        valid_out["target_mode"] = target_mode
        valid_out["pred"] = pred.reshape(-1)
        valid_out["train_start"] = train_seq.dates[0] if train_seq.dates else pd.NaT
        valid_out["train_end"] = train_seq.dates[-1] if train_seq.dates else pd.NaT
        valid_out["valid_start"] = bounds.valid_start
        valid_out["valid_end"] = bounds.valid_end
        all_fold_frames.append(valid_out)

        summary, day_rows = evaluate_predictions(
            valid_out,
            time_col=time_col,
            target_col=target_col,
            pred_col="pred",
            cfg=cfg,
            train_labels=train_labels,
            fold=fold_name,
            train_start=bounds.train_start,
            train_end=bounds.train_end,
            valid_start=bounds.valid_start,
            valid_end=bounds.valid_end,
        )
        summary_rows.append(summary)
        summary_rows[-1]["model"] = model_name
        summary_rows[-1]["target_mode"] = target_mode
        daily_rows.extend(day_rows)

    predictions = pd.concat(all_fold_frames, ignore_index=True)
    by_fold = attach_champion_deltas(pd.DataFrame(summary_rows))
    daily = pd.DataFrame(daily_rows)
    fit = pd.DataFrame(fit_rows)
    overall = summarize_overall(by_fold)
    if "profit_delta_vs_champion" in by_fold.columns and by_fold["profit_delta_vs_champion"].notna().any():
        overall["profit_delta_vs_champion"] = float(by_fold["profit_delta_vs_champion"].mean())
        overall["loss_day_delta_vs_champion"] = int(by_fold["loss_day_delta_vs_champion"].sum())
        overall["regret_delta_vs_champion"] = float(by_fold["regret_delta_vs_champion"].mean())

    paths = {
        "summary": REPORTS_DIR / f"backtest_{model_name}.csv",
        "daily": REPORTS_DIR / f"backtest_{model_name}_daily.csv",
        "overall": REPORTS_DIR / f"{model_name}_overall.csv",
        "predictions": REPORTS_DIR / f"predictions_{model_name}.csv",
        "fit": REPORTS_DIR / f"{model_name}_fit.csv",
        "report": REPORTS_DIR / f"{model_name}_experiment.md",
    }
    by_fold.to_csv(paths["summary"], index=False)
    daily.to_csv(paths["daily"], index=False)
    overall.to_csv(paths["overall"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    fit.to_csv(paths["fit"], index=False)
    write_report(cfg=cfg, overall=overall, by_fold=by_fold, fit=fit, paths=paths)
    print(overall.to_string(index=False))
    print(f"report_path={paths['report']}")


if __name__ == "__main__":
    main()
