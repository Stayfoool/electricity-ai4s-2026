from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity.data import load_train_frame
from electricity.dispatch import optimize_day, rank_day_pairs
from electricity.eval.metrics import daily_profit
from electricity.features.bid_space import markdown_table

CONFIG_PATH = Path("configs/ensemble_champion_segmented6_prior.yaml")
PRED_CACHE = Path("reports/cache_predictions_ens_champion_segmented6.csv")
CHAMPION_DAILY = Path("reports/backtest_ens_champion_segmented6_prior_daily.csv")
WEATHER_PREDS = Path("reports/weather_renewable_correction_predictions.csv")
REPORTS_DIR = Path("reports")

MODEL_NAME = "ens_champion_segmented6_prior"
BLOCK = 8
POWER = 1000.0
TOP_K_GRID = [3, 5, 10]
GAMMA_GRID = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
THRESHOLD_GRID = [0.0, 0.02, 0.05, 0.10, 0.20]


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pair_profit(true_prices: np.ndarray, charge_start: int, discharge_start: int) -> float:
    return float(
        (
            true_prices[discharge_start : discharge_start + BLOCK].sum()
            - true_prices[charge_start : charge_start + BLOCK].sum()
        )
        * POWER
    )


def window_means(values: np.ndarray) -> np.ndarray:
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    return (prefix[BLOCK:] - prefix[:-BLOCK]) / BLOCK


def build_weather_daily_signals() -> dict[pd.Timestamp, dict[str, np.ndarray]]:
    hourly = pd.read_csv(WEATHER_PREDS, parse_dates=["target_hour", "valid_date"])
    hourly["date"] = hourly["target_hour"].dt.normalize()
    signals: dict[pd.Timestamp, dict[str, np.ndarray]] = {}
    for date, group in hourly.groupby("date", sort=True):
        group = group.sort_values("target_hour")
        if len(group) != 24:
            continue
        bid = np.repeat(group["lgb_bid_space_fct"].to_numpy(dtype=float), 4)
        orig_bid = np.repeat(group["bid_space_fct"].to_numpy(dtype=float), 4)
        wind = np.repeat(group["wind_speed_mean"].to_numpy(dtype=float), 4)
        ghi = np.repeat(group["ghi_mean"].to_numpy(dtype=float), 4)
        signals[pd.Timestamp(date)] = {
            "weather_bid_window": window_means(bid),
            "orig_bid_window": window_means(orig_bid),
            "wind_window": window_means(wind),
            "ghi_window": window_means(ghi),
        }
    return signals


def load_prediction_days() -> pd.DataFrame:
    pred = pd.read_csv(PRED_CACHE, parse_dates=["times"])
    pred = pred.rename(columns={"target": "A"})
    pred["date"] = pred["times"].dt.normalize()
    pred["slot"] = pred["times"].dt.hour * 4 + pred["times"].dt.minute // 15
    return pred


def build_prior_by_fold(cfg: dict) -> dict[str, tuple[np.ndarray, np.ndarray, float, float]]:
    # Import locally: this is an analysis script and intentionally reuses the exact backtest logic.
    from electricity.eval.backtest import _fold_bounds, _maybe_build_prior

    labels = load_train_frame(cfg)
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    out = {}
    for fold in cfg["folds"]:
        bounds = _fold_bounds(fold)
        out[fold["name"]] = _maybe_build_prior(
            cfg,
            labels,
            time_col=time_col,
            target_col=target_col,
            train_end=bounds.train_end,
            train_start=bounds.train_start,
            valid_start=bounds.valid_start,
            valid_end=bounds.valid_end,
        )
    return out


def weather_pair_spread(
    signals: dict[str, np.ndarray],
    charge_start: int,
    discharge_start: int,
) -> float:
    return float(
        signals["weather_bid_window"][discharge_start]
        - signals["weather_bid_window"][charge_start]
    )


def orig_bid_pair_spread(
    signals: dict[str, np.ndarray],
    charge_start: int,
    discharge_start: int,
) -> float:
    return float(
        signals["orig_bid_window"][discharge_start] - signals["orig_bid_window"][charge_start]
    )


def prior_decision_score(
    *,
    predicted_spread: float,
    charge_start: int,
    discharge_start: int,
    log_prior_charge: np.ndarray | None,
    log_prior_discharge: np.ndarray | None,
    lambda_charge: float,
    lambda_discharge: float,
) -> float:
    score = float(predicted_spread)
    if log_prior_charge is not None and lambda_charge != 0.0:
        score += lambda_charge * float(log_prior_charge[charge_start])
    if log_prior_discharge is not None and lambda_discharge != 0.0:
        score += lambda_discharge * float(log_prior_discharge[discharge_start - BLOCK])
    return score


def build_candidate_rows() -> pd.DataFrame:
    cfg = load_config()
    pred = load_prediction_days()
    weather = build_weather_daily_signals()
    prior_by_fold = build_prior_by_fold(cfg)
    rows: list[dict[str, object]] = []

    for date, group in pred.groupby("date", sort=True):
        group = group.sort_values("slot")
        if len(group) != 96 or not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            continue
        if pd.Timestamp(date) not in weather:
            continue
        fold = str(group["fold"].iloc[0])
        log_pc, log_pd, lambda_c, lambda_d = prior_by_fold[fold]
        pred_prices = group["pred"].to_numpy(dtype=float)
        true_prices = group["A"].to_numpy(dtype=float)
        candidates = rank_day_pairs(
            pred_prices,
            top_k=10,
            log_prior_charge=log_pc,
            log_prior_discharge=log_pd,
            lambda_charge=lambda_c,
            lambda_discharge=lambda_d,
        )
        champion = optimize_day(
            pred_prices,
            log_prior_charge=log_pc,
            log_prior_discharge=log_pd,
            lambda_charge=lambda_c,
            lambda_discharge=lambda_d,
        )
        oracle = optimize_day(true_prices)
        oracle_profit = daily_profit(true_prices, oracle.power)
        sig = weather[pd.Timestamp(date)]
        top1_spread = float(candidates[0][0])
        top1_weather = weather_pair_spread(sig, candidates[0][1], candidates[0][2])
        top1_prior_score = prior_decision_score(
            predicted_spread=top1_spread,
            charge_start=candidates[0][1],
            discharge_start=candidates[0][2],
            log_prior_charge=log_pc,
            log_prior_discharge=log_pd,
            lambda_charge=lambda_c,
            lambda_discharge=lambda_d,
        )
        for rank, (predicted_spread, charge_start, discharge_start) in enumerate(
            candidates, start=1
        ):
            w_spread = weather_pair_spread(sig, charge_start, discharge_start)
            orig_spread = orig_bid_pair_spread(sig, charge_start, discharge_start)
            prior_score = prior_decision_score(
                predicted_spread=predicted_spread,
                charge_start=charge_start,
                discharge_start=discharge_start,
                log_prior_charge=log_pc,
                log_prior_discharge=log_pd,
                lambda_charge=lambda_c,
                lambda_discharge=lambda_d,
            )
            true_profit = pair_profit(true_prices, charge_start, discharge_start)
            rows.append(
                {
                    "model": MODEL_NAME,
                    "fold": fold,
                    "date": pd.Timestamp(date).date().isoformat(),
                    "candidate_rank": rank,
                    "charge_start": int(charge_start),
                    "discharge_start": int(discharge_start),
                    "predicted_spread": float(predicted_spread),
                    "predicted_spread_delta_from_top1": top1_spread - float(predicted_spread),
                    "prior_decision_score": prior_score,
                    "prior_score_delta_from_top1": top1_prior_score - prior_score,
                    "weather_bid_spread": w_spread,
                    "weather_bid_spread_delta_from_top1": w_spread - top1_weather,
                    "orig_bid_spread": orig_spread,
                    "wind_charge_mean": float(sig["wind_window"][charge_start]),
                    "wind_discharge_mean": float(sig["wind_window"][discharge_start]),
                    "ghi_charge_mean": float(sig["ghi_window"][charge_start]),
                    "ghi_discharge_mean": float(sig["ghi_window"][discharge_start]),
                    "true_profit": true_profit,
                    "oracle_profit": oracle_profit,
                    "regret_vs_oracle": oracle_profit - true_profit,
                    "oracle_charge_start": oracle.charge_start,
                    "oracle_discharge_start": oracle.discharge_start,
                    "champion_charge_start": champion.charge_start,
                    "champion_discharge_start": champion.discharge_start,
                }
            )
    return pd.DataFrame(rows)


def select_for_params(
    group: pd.DataFrame,
    *,
    top_k: int,
    gamma: float,
    threshold: float,
) -> pd.Series:
    cand = group[group["candidate_rank"] <= top_k].copy()
    cand["rerank_score"] = (
        cand["prior_decision_score"] + gamma * cand["weather_bid_spread_delta_from_top1"]
    )
    top1 = cand[cand["candidate_rank"] == 1].iloc[0]
    best = cand.sort_values(
        ["rerank_score", "predicted_spread", "candidate_rank"], ascending=[False, False, True]
    ).iloc[0]
    if int(best["candidate_rank"]) == 1:
        return top1
    if float(best["rerank_score"] - top1["rerank_score"]) < threshold:
        return top1
    return best


def evaluate_params(
    candidates: pd.DataFrame,
    *,
    top_k: int,
    gamma: float,
    threshold: float,
) -> pd.DataFrame:
    selected_rows = []
    for (_, _date), group in candidates.groupby(["fold", "date"], sort=True):
        selected_rows.append(
            select_for_params(group, top_k=top_k, gamma=gamma, threshold=threshold)
        )
    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    out["top_k"] = top_k
    out["gamma"] = gamma
    out["threshold"] = threshold
    top1 = candidates[candidates["candidate_rank"] == 1][["date", "true_profit"]].rename(
        columns={"true_profit": "top1_profit"}
    )
    out = out.merge(top1, on="date", how="left")
    out["lift_vs_top1"] = out["true_profit"] - out["top1_profit"]
    out["changed_from_top1"] = (out["candidate_rank"] != 1).astype(int)
    return out


def summarize_selection(selected: pd.DataFrame, *, method: str) -> dict[str, object]:
    profit = selected["true_profit"].astype(float)
    return {
        "method": method,
        "days": len(selected),
        "mean_profit": float(profit.mean()),
        "worst_profit": float(profit.min()),
        "loss_days": int((profit < 0).sum()),
        "mean_lift_vs_top1": float(selected["lift_vs_top1"].mean()),
        "positive_lift_days": int((selected["lift_vs_top1"] > 0).sum()),
        "negative_lift_days": int((selected["lift_vs_top1"] < 0).sum()),
        "changed_days": int(selected["changed_from_top1"].sum()),
        "mean_selected_rank": float(selected["candidate_rank"].mean()),
    }


def crossfold_select(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = sorted(candidates["fold"].unique())
    param_rows = []
    selected_frames = []
    for heldout in folds:
        train = candidates[candidates["fold"] != heldout]
        valid = candidates[candidates["fold"] == heldout]
        train_scores = []
        for top_k in TOP_K_GRID:
            for gamma in GAMMA_GRID:
                for threshold in THRESHOLD_GRID:
                    train_selected = evaluate_params(
                        train, top_k=top_k, gamma=gamma, threshold=threshold
                    )
                    summary = summarize_selection(train_selected, method="train")
                    train_scores.append(
                        {
                            "heldout_fold": heldout,
                            "top_k": top_k,
                            "gamma": gamma,
                            "threshold": threshold,
                            "train_mean_profit": summary["mean_profit"],
                            "train_loss_days": summary["loss_days"],
                            "train_changed_days": summary["changed_days"],
                        }
                    )
        score_df = pd.DataFrame(train_scores).sort_values(
            ["train_mean_profit", "train_loss_days", "train_changed_days", "top_k", "gamma"],
            ascending=[False, True, True, True, True],
        )
        best = score_df.iloc[0].to_dict()
        param_rows.append(best)
        valid_selected = evaluate_params(
            valid,
            top_k=int(best["top_k"]),
            gamma=float(best["gamma"]),
            threshold=float(best["threshold"]),
        )
        valid_selected["heldout_fold"] = heldout
        selected_frames.append(valid_selected)
    selected = pd.concat(selected_frames, ignore_index=True)

    top1 = candidates[candidates["candidate_rank"] == 1].copy()
    top1["top1_profit"] = top1["true_profit"]
    top1["lift_vs_top1"] = 0.0
    top1["changed_from_top1"] = 0
    top1["heldout_fold"] = top1["fold"]

    oracle_top10 = candidates.loc[
        candidates.groupby(["fold", "date"])["true_profit"].idxmax()
    ].copy()
    oracle_top10 = oracle_top10.reset_index(drop=True)
    oracle_top10 = oracle_top10.merge(
        top1[["date", "true_profit"]].rename(columns={"true_profit": "top1_profit"}),
        on="date",
        how="left",
    )
    oracle_top10["lift_vs_top1"] = oracle_top10["true_profit"] - oracle_top10["top1_profit"]
    oracle_top10["changed_from_top1"] = (oracle_top10["candidate_rank"] != 1).astype(int)
    oracle_top10["heldout_fold"] = oracle_top10["fold"]

    overall = pd.DataFrame(
        [
            summarize_selection(top1, method="champion_prior_top1"),
            summarize_selection(selected, method="weather_gated_rerank_cv"),
            summarize_selection(oracle_top10, method="oracle_top10_upper_bound"),
        ]
    )
    return pd.DataFrame(param_rows), selected, overall


def validate_top1_against_report(candidates: pd.DataFrame) -> None:
    report = pd.read_csv(CHAMPION_DAILY, parse_dates=["date"])
    top1 = candidates[candidates["candidate_rank"] == 1].copy()
    top1["date"] = pd.to_datetime(top1["date"])
    merged = report.merge(
        top1[["date", "charge_start", "discharge_start", "true_profit"]],
        on="date",
        how="inner",
        suffixes=("_report", "_script"),
    )
    if len(merged) != len(report):
        raise ValueError(f"top1 validation row mismatch: report={len(report)} script={len(merged)}")
    mismatch = merged[
        (merged["charge_start_report"] != merged["charge_start_script"])
        | (merged["discharge_start_report"] != merged["discharge_start_script"])
        | ((merged["profit"] - merged["true_profit"]).abs() > 1e-6)
    ]
    if not mismatch.empty:
        path = REPORTS_DIR / "weather_gated_rerank_top1_mismatch.csv"
        mismatch.to_csv(path, index=False)
        raise ValueError(f"top1 validation failed; wrote {path}")


def summarize_by_fold(selected: pd.DataFrame, top1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected_lookup = {"weather_gated_rerank_cv": selected, "champion_prior_top1": top1}
    for method, df in selected_lookup.items():
        for fold, group in df.groupby("fold", sort=True):
            profit = group["true_profit"].astype(float)
            changed = group.get("changed_from_top1", pd.Series(0, index=group.index))
            lift = group.get("lift_vs_top1", pd.Series(0.0, index=group.index))
            rows.append(
                {
                    "method": method,
                    "fold": fold,
                    "days": len(group),
                    "mean_profit": float(profit.mean()),
                    "loss_days": int((profit < 0).sum()),
                    "changed_days": int(changed.sum()),
                    "mean_lift_vs_top1": float(lift.mean()),
                }
            )
    return pd.DataFrame(rows)


def write_markdown(overall: pd.DataFrame, by_fold: pd.DataFrame, params: pd.DataFrame) -> None:
    rerank_profit = float(
        overall.loc[overall["method"] == "weather_gated_rerank_cv", "mean_profit"].iloc[0]
    )
    base_profit = float(
        overall.loc[overall["method"] == "champion_prior_top1", "mean_profit"].iloc[0]
    )
    lines = [
        "# Weather Gated Rerank Diagnostics",
        "",
        "## Scope",
        "",
        "Keep current champion-prior price predictions unchanged. Enumerate the "
        "predicted top-10 legal charge/discharge pairs using the same dispatch prior "
        "as the champion. Then test whether weather-corrected bid-space can "
        "conservatively rerank candidates inside the top-K set.",
        "",
        "The parameter choice is cross-fold: for each held-out validation month, "
        "`top_k`, `gamma`, and `threshold` are selected using the other validation "
        "months only.",
        "",
        "## Overall Result",
        "",
        markdown_table(overall, floatfmt=".4f"),
        "",
        "## Fold Result",
        "",
        markdown_table(by_fold, floatfmt=".4f"),
        "",
        "## Selected Parameters By Held-Out Fold",
        "",
        markdown_table(params, floatfmt=".4f"),
        "",
        "## Decision",
        "",
    ]
    if rerank_profit > base_profit + 50:
        lines += [
            "- Weather gated rerank improves held-out profit enough to justify hardening.",
            "- Next step: add a submit-time rerank path and test on 5-fold reports "
            "before promotion.",
        ]
    else:
        lines += [
            "- Weather gated rerank does not materially beat current champion-prior top1.",
            "- Keep it as diagnostic evidence; do not change champion dispatch yet.",
        ]
    lines += [
        "",
        "## Artifacts",
        "",
        "- `reports/weather_gated_rerank_candidates.csv`",
        "- `reports/weather_gated_rerank_selected_daily.csv`",
        "- `reports/weather_gated_rerank_summary.csv`",
        "- `reports/weather_gated_rerank_by_fold.csv`",
        "- `reports/weather_gated_rerank_params.csv`",
    ]
    (REPORTS_DIR / "weather_gated_rerank_diagnostics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    candidates = build_candidate_rows()
    validate_top1_against_report(candidates)
    params, selected, overall = crossfold_select(candidates)
    top1 = candidates[candidates["candidate_rank"] == 1].copy()
    top1["top1_profit"] = top1["true_profit"]
    top1["lift_vs_top1"] = 0.0
    top1["changed_from_top1"] = 0
    by_fold = summarize_by_fold(selected, top1)

    candidates.to_csv(REPORTS_DIR / "weather_gated_rerank_candidates.csv", index=False)
    selected.to_csv(REPORTS_DIR / "weather_gated_rerank_selected_daily.csv", index=False)
    overall.to_csv(REPORTS_DIR / "weather_gated_rerank_summary.csv", index=False)
    by_fold.to_csv(REPORTS_DIR / "weather_gated_rerank_by_fold.csv", index=False)
    params.to_csv(REPORTS_DIR / "weather_gated_rerank_params.csv", index=False)
    write_markdown(overall, by_fold, params)

    print(overall.to_string(index=False))
    print(f"candidate_path={REPORTS_DIR / 'weather_gated_rerank_candidates.csv'}")
    print(f"selected_path={REPORTS_DIR / 'weather_gated_rerank_selected_daily.csv'}")
    print(f"summary_path={REPORTS_DIR / 'weather_gated_rerank_summary.csv'}")
    print(f"markdown_path={REPORTS_DIR / 'weather_gated_rerank_diagnostics.md'}")


if __name__ == "__main__":
    main()
