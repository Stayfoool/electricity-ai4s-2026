from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from backtest_sequence_gru_direct_price import (
    REPORTS_DIR,
    attach_champion_deltas,
    fold_bounds,
    fold_train_frame,
    markdown_table,
    sequence_feature_columns,
    set_seed,
)
from electricity.data import load_train_frame
from electricity.dispatch import optimize_day
from electricity.eval.metrics import daily_profit
from electricity.features import build_feature_frame
from electricity.models.sequence import (
    SmallTCNPriceModel,
    SmallTFTLikeResidual,
    make_daily_sequence_frame,
    torch_available,
)

if torch_available():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
else:  # pragma: no cover - checked at runtime.
    torch = None
    nn = None


DEFAULT_CONFIG = Path("configs/sequence_window_gru_5fold.yaml")


class WindowGRUModel(nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_proj(x)
        temporal, _ = self.gru(x)
        point = self.head(temporal).squeeze(-1)
        return torch.nn.functional.avg_pool1d(point.unsqueeze(1), kernel_size=8, stride=1).squeeze(1)


class WindowPointWrapper(nn.Module):
    def __init__(self, point_model: nn.Module) -> None:
        super().__init__()
        self.point_model = point_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        point = self.point_model(x)
        return torch.nn.functional.avg_pool1d(point.unsqueeze(1), kernel_size=8, stride=1).squeeze(1)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_sequence_features(df: pd.DataFrame, cfg: dict, *, time_col: str) -> pd.DataFrame:
    feature_cfg = cfg.get("sequence_window", {}).get("feature_sets", {})
    spec = {
        **cfg,
        "sequence_direct": {"feature_sets": feature_cfg},
        "feature_sets": {
            "derived": bool(feature_cfg.get("business", True)),
            "bid_space": bool(feature_cfg.get("bid_space", True)),
        },
    }
    return build_feature_frame(df, spec, time_col=time_col)


def standardize_daily_features(train_features: np.ndarray, valid_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_flat = train_features.reshape(-1, train_features.shape[-1])
    valid_flat = valid_features.reshape(-1, valid_features.shape[-1])
    scaler.fit(train_flat)
    return (
        scaler.transform(train_flat).reshape(train_features.shape).astype(np.float32),
        scaler.transform(valid_flat).reshape(valid_features.shape).astype(np.float32),
    )


def make_price_sequence(df: pd.DataFrame, *, time_col: str, target_col: str, feature_cols: list[str]):
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


def window_means(price_curves: np.ndarray, block_size: int = 8) -> np.ndarray:
    if price_curves.ndim != 2 or price_curves.shape[1] != 96:
        raise ValueError(f"expected daily curves shape (n, 96), got {price_curves.shape}")
    cumsum = np.concatenate([np.zeros((price_curves.shape[0], 1)), np.cumsum(price_curves, axis=1)], axis=1)
    return (cumsum[:, block_size:] - cumsum[:, :-block_size]) / block_size


def train_window_model(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, dict[str, float]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for window deep experiment")
    seed = int(cfg.get("random_state", 2026))
    set_seed(seed)
    model_type = str(cfg.get("model_type", "gru"))
    if model_type == "gru":
        model = WindowGRUModel(
            input_size=train_x.shape[-1],
            hidden_size=int(cfg.get("hidden_size", 32)),
            dropout=float(cfg.get("dropout", 0.2)),
        )
    elif model_type == "tcn":
        model = WindowPointWrapper(
            SmallTCNPriceModel(
                input_size=train_x.shape[-1],
                hidden_size=int(cfg.get("hidden_size", 32)),
                levels=int(cfg.get("levels", 3)),
                kernel_size=int(cfg.get("kernel_size", 5)),
                dropout=float(cfg.get("dropout", 0.2)),
            )
        )
    elif model_type == "tft_like":
        model = WindowPointWrapper(
            SmallTFTLikeResidual(
                input_size=train_x.shape[-1],
                hidden_size=int(cfg.get("hidden_size", 16)),
                num_heads=int(cfg.get("num_heads", 1)),
                dropout=float(cfg.get("dropout", 0.3)),
            )
        )
    else:
        raise ValueError(f"unsupported sequence_window.model.model_type={model_type}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 0.001)), weight_decay=float(cfg.get("weight_decay", 0.01)))
    loss_fn = torch.nn.SmoothL1Loss(beta=float(cfg.get("huber_beta", 0.5)))
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32)))
    loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 16)), shuffle=True, drop_last=False)
    valid_x_t = torch.from_numpy(valid_x)
    valid_y_t = torch.from_numpy(valid_y.astype(np.float32))
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, int(cfg.get("max_epochs", 160)) + 1):
        model.train()
        for xb, yb in loader:
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
            if no_improve >= int(cfg.get("patience", 24)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(valid_x_t).cpu().numpy()
        train_pred = model(torch.from_numpy(train_x)).cpu().numpy()
    return pred, {
        "best_epoch": float(best_epoch),
        "best_valid_smooth_l1": float(best_loss),
        "train_window_rmse": float(np.sqrt(mean_squared_error(train_y.reshape(-1), train_pred.reshape(-1)))),
        "valid_window_rmse": float(np.sqrt(mean_squared_error(valid_y.reshape(-1), pred.reshape(-1)))),
        "valid_window_mae": float(mean_absolute_error(valid_y.reshape(-1), pred.reshape(-1))),
    }


def optimize_window_scores(
    window_scores: np.ndarray,
    *,
    tau: float = 0.0,
    block_size: int = 8,
    charge_power: float = -1000.0,
    discharge_power: float = 1000.0,
) -> dict:
    if len(window_scores) != 89:
        raise ValueError(f"expected 89 window scores, got {len(window_scores)}")
    candidates: list[tuple[float, int, int]] = []
    for tc in range(0, 81):
        for td in range(tc + block_size, 89):
            spread = float(window_scores[td] - window_scores[tc])
            candidates.append((spread, tc, td))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_spread, best_tc_raw, best_td_raw = candidates[0]
    power = np.zeros(96, dtype=float)
    best_tc = None
    best_td = None
    if best_spread > tau:
        best_tc = best_tc_raw
        best_td = best_td_raw
        power[best_tc : best_tc + block_size] = charge_power
        power[best_td : best_td + block_size] = discharge_power
    return {
        "power": power,
        "charge_start": best_tc,
        "discharge_start": best_td,
        "predicted_spread": best_spread,
        "top_candidate_count": len(candidates),
    }


def window_scores_to_point_proxy(window_scores: np.ndarray) -> np.ndarray:
    point = np.empty(96, dtype=float)
    point[:89] = window_scores
    point[89:] = window_scores[-1]
    return point


def evaluate_window_scores(
    *,
    valid_seq,
    pred_windows: np.ndarray,
    cfg: dict,
    fold: str,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    rows = []
    profits = []
    oracle_profits = []
    regrets = []
    loss_days = 0
    abs_charge_gaps = []
    abs_discharge_gaps = []
    day_rows = []
    for idx, date in enumerate(valid_seq.dates):
        true_prices = valid_seq.targets[idx]
        point_proxy = window_scores_to_point_proxy(pred_windows[idx])
        result = optimize_window_scores(pred_windows[idx], tau=float(cfg["dispatch"].get("tau", 0.0)))
        oracle = optimize_day(true_prices, tau=0.0)
        profit = daily_profit(true_prices, result["power"])
        oracle_profit = daily_profit(true_prices, oracle.power)
        regret = oracle_profit - profit
        if profit < 0:
            loss_days += 1
        if result["charge_start"] is not None and oracle.charge_start is not None:
            abs_charge_gaps.append(abs(result["charge_start"] - oracle.charge_start))
        if result["discharge_start"] is not None and oracle.discharge_start is not None:
            abs_discharge_gaps.append(abs(result["discharge_start"] - oracle.discharge_start))
        profits.append(profit)
        oracle_profits.append(oracle_profit)
        regrets.append(regret)
        for slot, ts in enumerate(valid_seq.times[idx]):
            rows.append({"times": ts, "fold": fold, "A": true_prices[slot], "pred": point_proxy[slot]})
        day_rows.append({
            "date": pd.Timestamp(date).date().isoformat(),
            "fold": fold,
            "profit": profit,
            "oracle_profit": oracle_profit,
            "traded": int(np.count_nonzero(result["power"]) > 0),
            "charge_start": result["charge_start"],
            "discharge_start": result["discharge_start"],
            "predicted_spread": result["predicted_spread"],
            "top_candidate_count": result["top_candidate_count"],
            "oracle_charge_start": oracle.charge_start,
            "oracle_discharge_start": oracle.discharge_start,
            "regret": regret,
            "profit_ratio_day": profit / oracle_profit if oracle_profit > 0 else 0.0,
            "charge_start_gap": None if result["charge_start"] is None or oracle.charge_start is None else result["charge_start"] - oracle.charge_start,
            "discharge_start_gap": None if result["discharge_start"] is None or oracle.discharge_start is None else result["discharge_start"] - oracle.discharge_start,
        })
    pred_df = pd.DataFrame(rows)
    true_windows = window_means(valid_seq.targets)
    summary = {
        "fold": fold,
        "model": cfg["sequence_window"]["name"],
        "valid_rows": len(pred_df),
        "valid_days": len(profits),
        "window_mae": float(mean_absolute_error(true_windows.reshape(-1), pred_windows.reshape(-1))),
        "window_rmse": float(np.sqrt(mean_squared_error(true_windows.reshape(-1), pred_windows.reshape(-1)))),
        "avg_profit": float(np.mean(profits)) if profits else 0.0,
        "oracle_avg_profit": float(np.mean(oracle_profits)) if oracle_profits else 0.0,
        "oracle_ratio": float(np.mean(profits) / np.mean(oracle_profits)) if oracle_profits and np.mean(oracle_profits) > 0 else 0.0,
        "avg_regret": float(np.mean(regrets)) if regrets else 0.0,
        "loss_days": loss_days,
        "mean_abs_charge_gap": float(np.mean(abs_charge_gaps)) if abs_charge_gaps else 0.0,
        "mean_abs_discharge_gap": float(np.mean(abs_discharge_gaps)) if abs_discharge_gaps else 0.0,
    }
    return pred_df, summary, day_rows


def write_report(cfg: dict, overall: pd.DataFrame, by_fold: pd.DataFrame, fit: pd.DataFrame, paths: dict[str, Path]) -> None:
    lines = [
        "# Window Deep Experiment",
        "",
        f"- Model: `{cfg['sequence_window']['name']}`.",
        "- Input: daily 96-point feature curve.",
        "- Target: 89 rolling 2-hour window average prices.",
        "- Dispatch: enumerate legal charge/discharge window pairs using predicted window scores.",
        "",
        "## Overall",
        markdown_table(overall),
        "",
        "## Fold Summary",
        markdown_table(by_fold, limit=80),
        "",
        "## Fit Summary",
        markdown_table(fit),
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    if not torch_available():
        raise RuntimeError("PyTorch is not installed")
    cfg = load_config(Path(args.config))
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    model_name = cfg["sequence_window"]["name"]
    REPORTS_DIR.mkdir(exist_ok=True)
    raw_df = load_train_frame(cfg)
    feature_df = add_sequence_features(raw_df, cfg, time_col=time_col)
    feature_cols = sequence_feature_columns({"data": cfg["data"], "sequence_direct": cfg["sequence_window"]})
    missing = sorted(set(feature_cols) - set(feature_df.columns))
    if missing:
        raise ValueError(f"missing features: {missing}")
    summaries = []
    daily_rows = []
    pred_frames = []
    fit_rows = []
    model_cfg = cfg["sequence_window"].get("model", {})
    for fold in cfg["folds"]:
        bounds = fold_bounds(fold)
        fold_name = str(fold["name"])
        print(f"window_deep_fold={fold_name}", flush=True)
        train_df = fold_train_frame(
            feature_df,
            time_col=time_col,
            train_end=bounds.train_end,
            train_window_days=cfg["sequence_window"].get("train_window_days"),
            train_start=bounds.train_start,
            valid_start=bounds.valid_start,
            valid_end=bounds.valid_end,
        ).dropna(subset=feature_cols + [target_col])
        valid_df = feature_df[(feature_df[time_col] >= bounds.valid_start) & (feature_df[time_col] <= bounds.valid_end)].copy()
        valid_df = valid_df.dropna(subset=feature_cols + [target_col])
        train_seq = make_price_sequence(train_df, time_col=time_col, target_col=target_col, feature_cols=feature_cols)
        valid_seq = make_price_sequence(valid_df, time_col=time_col, target_col=target_col, feature_cols=feature_cols)
        train_x, valid_x = standardize_daily_features(train_seq.features, valid_seq.features)
        train_y = window_means(train_seq.targets).astype(np.float32)
        valid_y = window_means(valid_seq.targets).astype(np.float32)
        pred_windows, fit = train_window_model(train_x=train_x, train_y=train_y, valid_x=valid_x, valid_y=valid_y, cfg=model_cfg)
        fit_rows.append({"fold": fold_name, "train_days": train_seq.features.shape[0], "valid_days": valid_seq.features.shape[0], **fit})
        pred_df, summary, day_rows = evaluate_window_scores(valid_seq=valid_seq, pred_windows=pred_windows, cfg=cfg, fold=fold_name)
        summaries.append(summary)
        daily_rows.extend(day_rows)
        pred_frames.append(pred_df)
    by_fold = attach_champion_deltas(pd.DataFrame(summaries))
    daily = pd.DataFrame(daily_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    fit = pd.DataFrame(fit_rows)
    overall = pd.DataFrame([{
        "model": model_name,
        "mean_profit": float(by_fold["avg_profit"].mean()),
        "min_profit": float(by_fold["avg_profit"].min()),
        "mean_window_rmse": float(by_fold["window_rmse"].mean()),
        "loss_days": int(by_fold["loss_days"].sum()),
        "mean_regret": float(by_fold["avg_regret"].mean()),
        "mean_charge_gap": float(by_fold["mean_abs_charge_gap"].mean()),
        "mean_discharge_gap": float(by_fold["mean_abs_discharge_gap"].mean()),
        "profit_delta_vs_champion": float(by_fold["profit_delta_vs_champion"].mean()) if "profit_delta_vs_champion" in by_fold else np.nan,
        "loss_day_delta_vs_champion": int(by_fold["loss_day_delta_vs_champion"].sum()) if "loss_day_delta_vs_champion" in by_fold else np.nan,
    }])
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
    write_report(cfg, overall, by_fold, fit, paths)
    print(overall.to_string(index=False))
    print(f"report_path={paths['report']}")


if __name__ == "__main__":
    main()
