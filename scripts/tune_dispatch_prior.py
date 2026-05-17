"""Grid-search dispatch slot priors against the cached champion predictions.

Idea:
  Modify dispatch to score a candidate (tc, td) pair as
      score = spread + lambda_c * log(p_c[tc] + alpha) + lambda_d * log(p_d[td] + alpha)
  where p_c, p_d are slot-level oracle frequencies computed only from training
  data prior to each fold (no leakage).

Inputs:
  - reports/cache_predictions_ens_champion_segmented6.csv (per-fold valid preds)
  - training labels (for oracle distribution)

Outputs:
  - reports/dispatch_prior_grid.csv  (grid search results, mean profit per lambda)
  - reports/dispatch_prior_best_daily.csv (per-day with best lambda applied)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BLOCK = 8
N_TC = 81  # 0..80
N_TD = 81  # 8..88


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labels(cfg: dict) -> pd.DataFrame:
    root = Path(cfg["paths"].get("material_dir", "eletricmaterial"))
    label_path = root / cfg["data"]["train_label"]
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    df = pd.read_csv(label_path, parse_dates=[time_col])
    return df[[time_col, target_col]].sort_values(time_col).reset_index(drop=True)


def daily_oracle(prices_96: np.ndarray) -> tuple[int, int, float]:
    """Return (best_tc, best_td, best_spread) using exact enumeration on true prices."""
    prefix = np.concatenate([[0.0], np.cumsum(prices_96)])
    block_sum = prefix[BLOCK:] - prefix[:-BLOCK]  # length 89, index k = sum prices[k:k+8]
    best = (-np.inf, 0, BLOCK)
    for tc in range(0, 81):
        for td in range(tc + BLOCK, 89):
            s = block_sum[td] - block_sum[tc]
            if s > best[0]:
                best = (s, tc, td)
    return int(best[1]), int(best[2]), float(best[0])


def compute_oracle_distribution(
    train_labels: pd.DataFrame, *, time_col: str, target_col: str, train_end: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray]:
    """Counts of oracle charge / discharge slots over training days <= train_end."""
    df = train_labels[train_labels[time_col] <= train_end].copy()
    df["date"] = df[time_col].dt.normalize()
    df["slot"] = df[time_col].dt.hour * 4 + df[time_col].dt.minute // 15

    counts_c = np.zeros(N_TC, dtype=float)
    counts_d = np.zeros(N_TD, dtype=float)  # td in 8..88, store at index td-8

    for _, day in df.groupby("date", sort=False):
        if len(day) != 96:
            continue
        day = day.sort_values("slot")
        if not np.array_equal(day["slot"].to_numpy(), np.arange(96)):
            continue
        prices = day[target_col].to_numpy(dtype=float)
        tc, td, spread = daily_oracle(prices)
        if spread <= 0:
            continue  # no-trade days don't contribute to slot prior
        counts_c[tc] += 1
        counts_d[td - BLOCK] += 1
    return counts_c, counts_d


def smoothed_log_prior(counts: np.ndarray, alpha: float) -> np.ndarray:
    p = (counts + alpha) / (counts.sum() + alpha * len(counts))
    return np.log(p)


def dispatch_with_prior(
    pred_prices_96: np.ndarray,
    *,
    log_p_c: np.ndarray,
    log_p_d: np.ndarray,
    lambda_c: float,
    lambda_d: float,
) -> tuple[int, int, float]:
    """Return (tc, td, predicted_spread_only) maximizing spread + lambda_c*log_pc + lambda_d*log_pd."""
    prefix = np.concatenate([[0.0], np.cumsum(pred_prices_96)])
    block_sum = prefix[BLOCK:] - prefix[:-BLOCK]

    best_score = -np.inf
    best = (0, BLOCK, -np.inf)
    for tc in range(0, 81):
        bp_c = lambda_c * log_p_c[tc]
        for td in range(tc + BLOCK, 89):
            spread = block_sum[td] - block_sum[tc]
            score = spread + bp_c + lambda_d * log_p_d[td - BLOCK]
            if score > best_score:
                best_score = score
                best = (tc, td, float(spread))
    return int(best[0]), int(best[1]), float(best[2])


def daily_profit(true_prices: np.ndarray, tc: int, td: int) -> float:
    if tc < 0 or td < 0:
        return 0.0
    return float(true_prices[td : td + BLOCK].sum() - true_prices[tc : tc + BLOCK].sum()) * 1000.0


def evaluate_grid(
    cache: pd.DataFrame,
    labels: pd.DataFrame,
    folds: list[dict],
    *,
    time_col: str,
    target_col: str,
    lambda_grid_c: list[float],
    lambda_grid_d: list[float],
    alpha: float,
    tau: float,
) -> tuple[pd.DataFrame, dict]:
    # Build per-fold log priors
    fold_priors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fold in folds:
        train_end = pd.Timestamp(fold["train_end"])
        cc, cd = compute_oracle_distribution(
            labels, time_col=time_col, target_col=target_col, train_end=train_end
        )
        fold_priors[fold["name"]] = (
            smoothed_log_prior(cc, alpha),
            smoothed_log_prior(cd, alpha),
        )
        print(
            f"  prior[{fold['name']}] charge_top5_slots="
            f"{np.argsort(cc)[::-1][:5].tolist()}  "
            f"discharge_top5_slots(td-8)={np.argsort(cd)[::-1][:5].tolist()}"
        )

    # Pre-organize cache by fold/date
    cache = cache.copy()
    cache["date"] = cache[time_col].dt.normalize()
    cache["slot"] = cache[time_col].dt.hour * 4 + cache[time_col].dt.minute // 15

    rows = []
    daily_records: list[dict] = []
    for lc in lambda_grid_c:
        for ld in lambda_grid_d:
            profits: list[float] = []
            regrets: list[float] = []
            chg_gaps: list[float] = []
            dis_gaps: list[float] = []
            loss = 0
            traded = 0
            for fold in folds:
                fname = fold["name"]
                log_p_c, log_p_d = fold_priors[fname]
                f_cache = cache[cache["fold"] == fname]
                for date, day in f_cache.groupby("date", sort=True):
                    if len(day) != 96:
                        continue
                    day = day.sort_values("slot")
                    if not np.array_equal(day["slot"].to_numpy(), np.arange(96)):
                        continue
                    pred = day["pred"].to_numpy(dtype=float)
                    true = day["target"].to_numpy(dtype=float)
                    tc, td, spread = dispatch_with_prior(
                        pred, log_p_c=log_p_c, log_p_d=log_p_d, lambda_c=lc, lambda_d=ld
                    )
                    if spread > tau:
                        p = daily_profit(true, tc, td)
                        traded += 1
                    else:
                        tc, td, p = -1, -1, 0.0
                    o_tc, o_td, o_sp = daily_oracle(true)
                    o_p = daily_profit(true, o_tc, o_td) if o_sp > 0 else 0.0
                    if p < 0:
                        loss += 1
                    profits.append(p)
                    regrets.append(o_p - p)
                    if tc >= 0:
                        chg_gaps.append(abs(tc - o_tc))
                        dis_gaps.append(abs(td - o_td))
                    if (lc, ld) == (0.0, 0.0):
                        daily_records.append(
                            {
                                "date": date.date().isoformat(),
                                "fold": fname,
                                "profit": p,
                                "oracle_profit": o_p,
                                "regret": o_p - p,
                                "tc": tc,
                                "td": td,
                                "o_tc": o_tc,
                                "o_td": o_td,
                            }
                        )
            rows.append(
                {
                    "lambda_c": lc,
                    "lambda_d": ld,
                    "mean_profit": float(np.mean(profits)),
                    "mean_regret": float(np.mean(regrets)),
                    "loss_days": loss,
                    "trade_days": traded,
                    "abs_charge_gap": float(np.mean(chg_gaps)) if chg_gaps else 0.0,
                    "abs_discharge_gap": float(np.mean(dis_gaps)) if dis_gaps else 0.0,
                }
            )
    return pd.DataFrame(rows), {"baseline_daily": daily_records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ensemble_champion_and_segmented_6.yaml")
    parser.add_argument(
        "--cache", default="reports/cache_predictions_ens_champion_segmented6.csv"
    )
    parser.add_argument("--out", default="reports/dispatch_prior_grid.csv")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]

    cache = pd.read_csv(args.cache, parse_dates=[time_col])
    labels = load_labels(cfg)
    print(f"loaded cache: {len(cache)} rows, labels: {len(labels)} rows")

    lambdas = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

    print("\nbuilding per-fold oracle priors:")
    grid, extra = evaluate_grid(
        cache,
        labels,
        cfg["folds"],
        time_col=time_col,
        target_col=target_col,
        lambda_grid_c=lambdas,
        lambda_grid_d=lambdas,
        alpha=args.alpha,
        tau=args.tau,
    )
    grid = grid.sort_values("mean_profit", ascending=False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.out, index=False)

    print("\n=== Top 15 (lambda_c, lambda_d) by mean_profit ===")
    print(grid.head(15).to_string(index=False))

    baseline = grid[(grid.lambda_c == 0.0) & (grid.lambda_d == 0.0)].iloc[0]
    best = grid.iloc[0]
    print(
        f"\nbaseline (lambda=0,0): profit={baseline.mean_profit:.2f}  "
        f"regret={baseline.mean_regret:.2f}  loss_days={int(baseline.loss_days)}  "
        f"chg_gap={baseline.abs_charge_gap:.2f}  dis_gap={baseline.abs_discharge_gap:.2f}"
    )
    print(
        f"best     (lambda={best.lambda_c},{best.lambda_d}): profit={best.mean_profit:.2f}  "
        f"regret={best.mean_regret:.2f}  loss_days={int(best.loss_days)}  "
        f"chg_gap={best.abs_charge_gap:.2f}  dis_gap={best.abs_discharge_gap:.2f}"
    )
    print(f"lift     = {best.mean_profit - baseline.mean_profit:+.2f}")


if __name__ == "__main__":
    main()
