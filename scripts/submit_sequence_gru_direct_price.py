from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from backtest_sequence_gru_direct_price import (
    add_sequence_features,
    make_price_sequence,
    sequence_feature_columns,
    set_seed,
    transform_daily_target,
)
from electricity.data import load_test_frame, load_train_frame
from electricity.dispatch import build_dispatch_prior, optimize_day
from electricity.models.sequence import SmallGRUPriceModel, torch_available
from electricity.submit import SUBMIT_COLUMNS, validate_submission

if torch_available():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
else:  # pragma: no cover - checked at runtime.
    torch = None
    DataLoader = None
    TensorDataset = None


DEFAULT_CONFIG = Path("configs/sequence_gru_direct_price_5fold.yaml")


@dataclass(frozen=True)
class DailyFeatureSequence:
    dates: list[pd.Timestamp]
    times: list[np.ndarray]
    features: np.ndarray


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_test_features(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    cfg: dict,
    *,
    time_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_marked = raw_train.copy()
    test_marked = raw_test.copy()
    train_marked["__is_test"] = 0
    test_marked["__is_test"] = 1
    combined = pd.concat([train_marked, test_marked], ignore_index=True, sort=False)
    combined = combined.sort_values(time_col).reset_index(drop=True)
    featured = add_sequence_features(combined, cfg, time_col=time_col)
    train_df = featured[featured["__is_test"] == 0].drop(columns=["__is_test"]).reset_index(drop=True)
    test_df = featured[featured["__is_test"] == 1].drop(columns=["__is_test"]).reset_index(drop=True)
    return train_df, test_df


def make_test_sequence(
    df: pd.DataFrame,
    *,
    time_col: str,
    feature_cols: list[str],
) -> DailyFeatureSequence:
    work = df[[time_col, *feature_cols]].copy()
    work["date"] = work[time_col].dt.normalize()
    work["slot"] = work[time_col].dt.hour * 4 + work[time_col].dt.minute // 15

    dates: list[pd.Timestamp] = []
    times: list[np.ndarray] = []
    features: list[np.ndarray] = []
    for date, group in work.groupby("date", sort=True):
        if len(group) != 96:
            raise ValueError(f"test day {date.date()} must contain 96 rows, got {len(group)}")
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            raise ValueError(f"test day {date.date()} has missing or duplicated slots")
        feature_values = group[feature_cols].to_numpy(dtype=float)
        if not np.isfinite(feature_values).all():
            raise ValueError(f"test day {date.date()} has non-finite feature values")
        dates.append(pd.Timestamp(date))
        times.append(group[time_col].to_numpy())
        features.append(feature_values)

    if not dates:
        raise ValueError("no complete test days found")
    return DailyFeatureSequence(dates=dates, times=times, features=np.stack(features))


def scale_train_test_features(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_flat = train_features.reshape(-1, train_features.shape[-1])
    test_flat = test_features.reshape(-1, test_features.shape[-1])
    scaler.fit(train_flat)
    train_scaled = scaler.transform(train_flat).reshape(train_features.shape).astype(np.float32)
    test_scaled = scaler.transform(test_flat).reshape(test_features.shape).astype(np.float32)
    return train_scaled, test_scaled


def infer_fixed_epochs(cfg: dict, explicit_epochs: int | None) -> int:
    if explicit_epochs is not None:
        return explicit_epochs
    model_name = str(cfg["sequence_direct"]["name"])
    fit_path = Path(cfg["paths"]["reports_dir"]) / f"{model_name}_fit.csv"
    if fit_path.exists():
        fit = pd.read_csv(fit_path)
        if "best_epoch" in fit.columns and not fit["best_epoch"].dropna().empty:
            return max(1, int(round(float(fit["best_epoch"].median()))))
    return int(cfg["sequence_direct"].get("gru", {}).get("submit_epochs", 20))


def train_full_model(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    model_cfg: dict,
    epochs: int,
) -> SmallGRUPriceModel:
    if torch is None or DataLoader is None or TensorDataset is None or SmallGRUPriceModel is None:
        raise RuntimeError("PyTorch is required for sequence GRU submit")

    seed = int(model_cfg.get("random_state", 2026))
    set_seed(seed)
    model = SmallGRUPriceModel(
        input_size=train_x.shape[-1],
        hidden_size=int(model_cfg.get("hidden_size", 32)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        bidirectional=bool(model_cfg.get("bidirectional", False)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.get("learning_rate", 0.001)),
        weight_decay=float(model_cfg.get("weight_decay", 0.01)),
    )
    loss_fn = torch.nn.SmoothL1Loss(beta=float(model_cfg.get("huber_beta", 0.5)))
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32)))
    loader = DataLoader(
        train_ds,
        batch_size=int(model_cfg.get("batch_size", 16)),
        shuffle=True,
        drop_last=False,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            print(f"epoch={epoch} train_smooth_l1={mean_loss:.6f}", flush=True)

    model.eval()
    return model


def predict_daily_curves(model: SmallGRUPriceModel, test_x: np.ndarray) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for sequence GRU submit")
    with torch.no_grad():
        return model(torch.from_numpy(test_x)).cpu().numpy()


def maybe_build_prior(
    cfg: dict,
    raw_train: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
) -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
    prior_cfg = cfg.get("dispatch", {}).get("prior")
    if not prior_cfg or not bool(prior_cfg.get("enabled", False)):
        return None, None, 0.0, 0.0
    log_pc, log_pd = build_dispatch_prior(
        raw_train[[time_col, target_col]],
        time_col=time_col,
        target_col=target_col,
        train_end=None,
        alpha=float(prior_cfg.get("alpha", 0.5)),
    )
    return (
        log_pc,
        log_pd,
        float(prior_cfg.get("lambda_charge", 0.0)),
        float(prior_cfg.get("lambda_discharge", 0.0)),
    )


def build_submission(
    *,
    test_seq: DailyFeatureSequence,
    pred_curves: np.ndarray,
    cfg: dict,
    log_prior_charge: np.ndarray | None,
    log_prior_discharge: np.ndarray | None,
    lambda_charge: float,
    lambda_discharge: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    decision_rows: list[dict] = []
    for day_idx, date in enumerate(test_seq.dates):
        pred_prices = pred_curves[day_idx].astype(float)
        result = optimize_day(
            pred_prices,
            tau=float(cfg["dispatch"].get("tau", 0.0)),
            block_size=int(cfg["dispatch"]["block_size"]),
            charge_power=float(cfg["dispatch"]["charge_power"]),
            discharge_power=float(cfg["dispatch"]["discharge_power"]),
            log_prior_charge=log_prior_charge,
            log_prior_discharge=log_prior_discharge,
            lambda_charge=lambda_charge,
            lambda_discharge=lambda_discharge,
        )
        times = pd.to_datetime(test_seq.times[day_idx])
        for slot in range(96):
            rows.append(
                {
                    "times": times[slot].strftime("%Y-%m-%d %H:%M:%S"),
                    "实时价格": float(pred_prices[slot]),
                    "power": float(result.power[slot]),
                }
            )
        decision_rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "charge_start": result.charge_start,
                "discharge_start": result.discharge_start,
                "predicted_spread": result.predicted_spread,
                "top2_spread": result.top2_spread,
                "top5_spread_mean": result.top5_spread_mean,
                "top5_spread_std": result.top5_spread_std,
                "top1_top2_gap": result.top1_top2_gap,
                "traded": int(np.count_nonzero(result.power) > 0),
            }
        )

    submission = pd.DataFrame(rows, columns=SUBMIT_COLUMNS)
    decisions = pd.DataFrame(decision_rows)
    validate_submission(submission, cfg)
    return submission, decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if not torch_available():
        raise RuntimeError("PyTorch is not installed")

    cfg = load_config(Path(args.config))
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    model_name = str(cfg["sequence_direct"]["name"])
    target_mode = str(cfg["sequence_direct"].get("target_mode", "absolute"))
    output_path = (
        Path(args.output)
        if args.output
        else Path(cfg["paths"]["outputs_dir"]) / f"output_{model_name}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_train = load_train_frame(cfg)
    raw_test = load_test_frame(cfg)
    train_df, test_df = build_train_test_features(raw_train, raw_test, cfg, time_col=time_col)
    feature_cols = sequence_feature_columns(cfg)
    missing = sorted(set(feature_cols) - set(train_df.columns) - set(test_df.columns))
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    train_df = train_df.dropna(subset=feature_cols + [target_col]).copy()
    train_seq = make_price_sequence(
        train_df,
        time_col=time_col,
        target_col=target_col,
        feature_cols=feature_cols,
    )
    test_seq = make_test_sequence(test_df, time_col=time_col, feature_cols=feature_cols)
    if train_seq.features.shape[0] == 0:
        raise ValueError("no complete training days found")

    train_x, test_x = scale_train_test_features(train_seq.features, test_seq.features)
    train_y = transform_daily_target(train_seq.targets.astype(np.float32), target_mode).astype(np.float32)
    epochs = infer_fixed_epochs(cfg, args.epochs)
    print(
        f"training_full_gru model={model_name} target_mode={target_mode} "
        f"train_days={train_x.shape[0]} test_days={test_x.shape[0]} "
        f"features={train_x.shape[-1]} epochs={epochs}",
        flush=True,
    )
    model = train_full_model(
        train_x=train_x,
        train_y=train_y,
        model_cfg=cfg["sequence_direct"].get("gru", {}),
        epochs=epochs,
    )
    pred_curves = predict_daily_curves(model, test_x)
    log_pc, log_pd, lc_w, ld_w = maybe_build_prior(
        cfg,
        raw_train,
        time_col=time_col,
        target_col=target_col,
    )
    if log_pc is not None:
        print(
            f"dispatch_prior_enabled lambda_charge={lc_w} lambda_discharge={ld_w}",
            flush=True,
        )

    submission, decisions = build_submission(
        test_seq=test_seq,
        pred_curves=pred_curves,
        cfg=cfg,
        log_prior_charge=log_pc,
        log_prior_discharge=log_pd,
        lambda_charge=lc_w,
        lambda_discharge=ld_w,
    )
    submission.to_csv(output_path, index=False)
    decision_path = reports_dir / f"submit_{model_name}_decisions.csv"
    decisions.to_csv(decision_path, index=False)

    trade_days = int(decisions["traded"].sum())
    print(f"config={args.config}")
    print(f"output_path={output_path}")
    print(f"decision_path={decision_path}")
    print(f"shape={submission.shape}")
    print(f"trade_days={trade_days}")
    print(
        "predicted_spread_summary="
        f"min:{decisions['predicted_spread'].min():.4f},"
        f"mean:{decisions['predicted_spread'].mean():.4f},"
        f"max:{decisions['predicted_spread'].max():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
