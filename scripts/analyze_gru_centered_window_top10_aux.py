from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity.data import load_train_frame
from electricity.dispatch import build_dispatch_prior, rank_day_pairs

REPORTS_DIR = Path("reports")
ANCHOR_CONFIG = Path("configs/ensemble_champion_segmented6_prior_5fold.yaml")
ANCHOR_PRED = REPORTS_DIR / "cache_predictions_ens_champion_segmented6.csv"
GRU_CENTERED_PRED = REPORTS_DIR / "predictions_sequence_gru_centered_5fold.csv"
WINDOW_GRU_PRED = REPORTS_DIR / "predictions_sequence_window_gru_5fold.csv"

DATASET_PATH = REPORTS_DIR / "gru_centered_window_gru_top10_aux_dataset.csv"
DAILY_PATH = REPORTS_DIR / "gru_centered_window_gru_top10_aux_daily.csv"
BY_FOLD_PATH = REPORTS_DIR / "gru_centered_window_gru_top10_aux_by_fold.csv"
OVERALL_PATH = REPORTS_DIR / "gru_centered_window_gru_top10_aux_overall.csv"
RULE_PATH = REPORTS_DIR / "gru_centered_window_gru_top10_aux_rule_search.csv"
MD_PATH = REPORTS_DIR / "gru_centered_window_gru_top10_aux_diagnostics.md"

TOP_K = 10
BLOCK_SIZE = 8
POWER = 1000.0


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


def pair_profit(true_prices: np.ndarray, charge_start: int, discharge_start: int) -> float:
    charge_sum = true_prices[charge_start : charge_start + BLOCK_SIZE].sum()
    discharge_sum = true_prices[discharge_start : discharge_start + BLOCK_SIZE].sum()
    return float((discharge_sum - charge_sum) * POWER)


def block_spread(point_scores: np.ndarray, charge_start: int, discharge_start: int) -> float:
    charge_sum = point_scores[charge_start : charge_start + BLOCK_SIZE].sum()
    discharge_sum = point_scores[discharge_start : discharge_start + BLOCK_SIZE].sum()
    return float(discharge_sum - charge_sum)


def window_spread(window_scores: np.ndarray, charge_start: int, discharge_start: int) -> float:
    if len(window_scores) != 89:
        raise ValueError(f"expected 89 window scores, got {len(window_scores)}")
    return float(window_scores[discharge_start] - window_scores[charge_start])


def rank_window_pairs(window_scores: np.ndarray, top_k: int | None = None) -> list[tuple[float, int, int]]:
    rows: list[tuple[float, int, int]] = []
    for tc in range(0, 81):
        for td in range(tc + BLOCK_SIZE, 89):
            rows.append((window_spread(window_scores, tc, td), tc, td))
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows if top_k is None else rows[:top_k]


def fold_priors(cfg: dict) -> dict[str, tuple[np.ndarray | None, np.ndarray | None, float, float]]:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    labels = load_train_frame(cfg)[[time_col, target_col]].copy()
    prior_cfg = cfg.get("dispatch", {}).get("prior", {})
    if not prior_cfg.get("enabled", False):
        return {str(fold["name"]): (None, None, 0.0, 0.0) for fold in cfg["folds"]}

    out: dict[str, tuple[np.ndarray | None, np.ndarray | None, float, float]] = {}
    for fold in cfg["folds"]:
        bounds = fold_bounds(fold)
        prior_labels = labels.copy()
        if bounds.train_start is not None:
            prior_labels = prior_labels[prior_labels[time_col] >= bounds.train_start]
        outside_valid = (prior_labels[time_col] < bounds.valid_start) | (
            prior_labels[time_col] > bounds.valid_end
        )
        prior_labels = prior_labels[outside_valid]
        log_pc, log_pd = build_dispatch_prior(
            prior_labels,
            time_col=time_col,
            target_col=target_col,
            train_end=bounds.train_end,
            alpha=float(prior_cfg.get("alpha", 0.5)),
        )
        out[str(fold["name"])] = (
            log_pc,
            log_pd,
            float(prior_cfg.get("lambda_charge", 0.0)),
            float(prior_cfg.get("lambda_discharge", 0.0)),
        )
    return out


def read_prediction(path: Path, *, value_col: str, rename_value: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["times"])
    if value_col not in df.columns:
        raise ValueError(f"{path} missing value column {value_col}")
    out = df[["times", "fold", value_col]].rename(columns={value_col: rename_value}).copy()
    out["date"] = out["times"].dt.normalize()
    out["slot"] = out["times"].dt.hour * 4 + out["times"].dt.minute // 15
    return out


def load_frames() -> pd.DataFrame:
    anchor = pd.read_csv(ANCHOR_PRED, parse_dates=["times"])
    anchor = anchor.rename(columns={"target": "A", "pred": "anchor_pred"})
    anchor = anchor[["times", "fold", "A", "anchor_pred"]]
    anchor["date"] = anchor["times"].dt.normalize()
    anchor["slot"] = anchor["times"].dt.hour * 4 + anchor["times"].dt.minute // 15

    gru = read_prediction(GRU_CENTERED_PRED, value_col="pred", rename_value="gru_centered_pred")
    window = read_prediction(WINDOW_GRU_PRED, value_col="pred", rename_value="window_gru_score")
    merged = anchor.merge(gru[["times", "gru_centered_pred"]], on="times", how="inner")
    merged = merged.merge(window[["times", "window_gru_score"]], on="times", how="inner")
    if merged.empty:
        raise ValueError("merged predictions are empty")
    return merged.sort_values(["fold", "date", "slot"]).reset_index(drop=True)


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(ANCHOR_CONFIG)
    priors = fold_priors(cfg)
    frame = load_frames()

    candidate_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for (fold, date), group in frame.groupby(["fold", "date"], sort=True):
        if len(group) != 96:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            continue
        true_prices = group["A"].to_numpy(dtype=float)
        anchor_pred = group["anchor_pred"].to_numpy(dtype=float)
        gru_pred = group["gru_centered_pred"].to_numpy(dtype=float)
        window_scores = group["window_gru_score"].to_numpy(dtype=float)[:89]

        log_pc, log_pd, lc_w, ld_w = priors[str(fold)]
        anchor_top10 = rank_day_pairs(
            anchor_pred,
            top_k=TOP_K,
            log_prior_charge=log_pc,
            log_prior_discharge=log_pd,
            lambda_charge=lc_w,
            lambda_discharge=ld_w,
        )
        true_oracle = rank_day_pairs(true_prices, top_k=1)[0]
        true_oracle_profit = float(true_oracle[0] * POWER)
        gru_global = rank_day_pairs(gru_pred, top_k=1)[0]
        window_global = rank_window_pairs(window_scores, top_k=1)[0]

        top1_anchor_spread = float(anchor_top10[0][0])
        top1_gru_score = block_spread(gru_pred, anchor_top10[0][1], anchor_top10[0][2])
        top1_window_score = window_spread(window_scores, anchor_top10[0][1], anchor_top10[0][2])
        anchor_pairs = {(int(tc), int(td)) for _, tc, td in anchor_top10}

        rows_for_day: list[dict[str, object]] = []
        for rank, (anchor_spread, tc, td) in enumerate(anchor_top10, start=1):
            gru_score = block_spread(gru_pred, tc, td)
            window_score = window_spread(window_scores, tc, td)
            row = {
                "fold": str(fold),
                "date": date.date().isoformat(),
                "candidate_rank": int(rank),
                "charge_start": int(tc),
                "discharge_start": int(td),
                "anchor_predicted_spread": float(anchor_spread),
                "anchor_spread_delta_from_top1": top1_anchor_spread - float(anchor_spread),
                "gru_centered_score": gru_score,
                "gru_score_delta_from_top1": gru_score - top1_gru_score,
                "window_gru_score": window_score,
                "window_score_delta_from_top1": window_score - top1_window_score,
                "true_profit": pair_profit(true_prices, tc, td),
                "oracle_profit": true_oracle_profit,
                "is_true_oracle_pair": bool(tc == true_oracle[1] and td == true_oracle[2]),
                "is_gru_global_top1": bool(tc == gru_global[1] and td == gru_global[2]),
                "is_window_global_top1": bool(tc == window_global[1] and td == window_global[2]),
            }
            rows_for_day.append(row)
            candidate_rows.append(row)

        top1 = rows_for_day[0]
        top10_best = max(rows_for_day, key=lambda item: float(item["true_profit"]))
        gru_pick = max(rows_for_day, key=lambda item: float(item["gru_centered_score"]))
        window_pick = max(rows_for_day, key=lambda item: float(item["window_gru_score"]))
        avg_rank_df = pd.DataFrame(rows_for_day)
        avg_rank_df["gru_rank_in_top10"] = avg_rank_df["gru_centered_score"].rank(
            ascending=False, method="min"
        )
        avg_rank_df["window_rank_in_top10"] = avg_rank_df["window_gru_score"].rank(
            ascending=False, method="min"
        )
        avg_rank_pick = avg_rank_df.loc[
            (avg_rank_df["gru_rank_in_top10"] + avg_rank_df["window_rank_in_top10"]).idxmin()
        ].to_dict()

        daily_rows.append(
            {
                "fold": str(fold),
                "date": date.date().isoformat(),
                "anchor_profit": float(top1["true_profit"]),
                "anchor_charge_start": int(top1["charge_start"]),
                "anchor_discharge_start": int(top1["discharge_start"]),
                "anchor_predicted_spread": float(top1["anchor_predicted_spread"]),
                "anchor_top1_top2_gap": float(
                    rows_for_day[0]["anchor_predicted_spread"]
                    - rows_for_day[1]["anchor_predicted_spread"]
                ),
                "oracle_profit": true_oracle_profit,
                "oracle_top10_profit": float(top10_best["true_profit"]),
                "oracle_top10_lift": float(top10_best["true_profit"] - top1["true_profit"]),
                "top10_best_rank": int(top10_best["candidate_rank"]),
                "gru_global_in_anchor_top10": (int(gru_global[1]), int(gru_global[2])) in anchor_pairs,
                "window_global_in_anchor_top10": (int(window_global[1]), int(window_global[2]))
                in anchor_pairs,
                "gru_top10_profit": float(gru_pick["true_profit"]),
                "gru_top10_rank": int(gru_pick["candidate_rank"]),
                "gru_changed": int(int(gru_pick["candidate_rank"]) != 1),
                "window_top10_profit": float(window_pick["true_profit"]),
                "window_top10_rank": int(window_pick["candidate_rank"]),
                "window_changed": int(int(window_pick["candidate_rank"]) != 1),
                "avg_rank_profit": float(avg_rank_pick["true_profit"]),
                "avg_rank_top10_rank": int(avg_rank_pick["candidate_rank"]),
                "avg_rank_changed": int(int(avg_rank_pick["candidate_rank"]) != 1),
                "gru_window_same_pick": bool(
                    int(gru_pick["charge_start"]) == int(window_pick["charge_start"])
                    and int(gru_pick["discharge_start"]) == int(window_pick["discharge_start"])
                ),
                "consensus_profit": float(gru_pick["true_profit"])
                if int(gru_pick["charge_start"]) == int(window_pick["charge_start"])
                and int(gru_pick["discharge_start"]) == int(window_pick["discharge_start"])
                else float(top1["true_profit"]),
                "consensus_changed": int(
                    int(gru_pick["candidate_rank"]) != 1
                    and int(gru_pick["charge_start"]) == int(window_pick["charge_start"])
                    and int(gru_pick["discharge_start"]) == int(window_pick["discharge_start"])
                ),
            }
        )

    candidates = pd.DataFrame(candidate_rows)
    candidates["gru_rank_in_top10"] = candidates.groupby(["fold", "date"])[
        "gru_centered_score"
    ].rank(ascending=False, method="min")
    candidates["window_rank_in_top10"] = candidates.groupby(["fold", "date"])[
        "window_gru_score"
    ].rank(ascending=False, method="min")
    candidates["avg_aux_rank"] = candidates["gru_rank_in_top10"] + candidates["window_rank_in_top10"]
    return candidates, pd.DataFrame(daily_rows)


def selection_frame(daily: pd.DataFrame) -> pd.DataFrame:
    method_map = {
        "anchor_top1": "anchor_profit",
        "oracle_top10": "oracle_top10_profit",
        "gru_top10": "gru_top10_profit",
        "window_top10": "window_top10_profit",
        "avg_aux_rank": "avg_rank_profit",
        "consensus_if_same_else_anchor": "consensus_profit",
    }
    rows = []
    for method, profit_col in method_map.items():
        if profit_col == "anchor_profit":
            temp = daily[["fold", "date", "anchor_profit"]].copy()
            temp["profit"] = temp["anchor_profit"]
        else:
            temp = daily[["fold", "date", "anchor_profit", profit_col]].copy()
            temp = temp.rename(columns={profit_col: "profit"})
        temp["method"] = method
        if method == "gru_top10":
            temp["changed"] = daily["gru_changed"].to_numpy()
        elif method == "window_top10":
            temp["changed"] = daily["window_changed"].to_numpy()
        elif method == "avg_aux_rank":
            temp["changed"] = daily["avg_rank_changed"].to_numpy()
        elif method == "consensus_if_same_else_anchor":
            temp["changed"] = daily["consensus_changed"].to_numpy()
        elif method == "oracle_top10":
            temp["changed"] = (daily["top10_best_rank"] != 1).astype(int).to_numpy()
        else:
            temp["changed"] = 0
        rows.append(temp)
    out = pd.concat(rows, ignore_index=True)
    out["lift_vs_anchor"] = out["profit"] - out["anchor_profit"]
    out["loss_day"] = (out["profit"] < 0).astype(int)
    return out


def summarize_selection(selected: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    anchor = selected[selected["method"] == "anchor_top1"].groupby(group_cols)["profit"].mean()
    rows = []
    for key, group in selected.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        item = dict(zip(group_cols, key, strict=True))
        anchor_key = key[:-1] + ("anchor_top1",) if group_cols[-1] == "method" else key
        if group_cols[-1] == "method":
            anchor_profit = float(anchor.loc[anchor_key])
        else:
            anchor_profit = float(group[group["method"] == "anchor_top1"]["profit"].mean())
        item.update(
            {
                "days": int(group["date"].nunique()),
                "mean_profit": float(group["profit"].mean()),
                "min_profit": float(group["profit"].min()),
                "loss_days": int((group["profit"] < 0).sum()),
                "changed_days": int(group["changed"].sum()),
                "positive_changed_days": int(((group["changed"] == 1) & (group["lift_vs_anchor"] > 0)).sum()),
                "negative_changed_days": int(((group["changed"] == 1) & (group["lift_vs_anchor"] < 0)).sum()),
                "profit_delta_vs_anchor": float(group["profit"].mean() - anchor_profit),
                "total_lift_vs_anchor": float(group["lift_vs_anchor"].sum()),
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def _select_candidate_by_rule(
    group: pd.DataFrame,
    *,
    score_col: str,
    score_delta_col: str,
    anchor_gap_max: float,
    aux_delta_min: float,
) -> pd.Series:
    top1 = group[group["candidate_rank"] == 1].iloc[0]
    valid = group[
        (group["candidate_rank"] != 1)
        & (group["anchor_spread_delta_from_top1"] <= anchor_gap_max)
        & (group[score_delta_col] >= aux_delta_min)
    ].copy()
    if valid.empty:
        return top1
    return valid.sort_values(score_col, ascending=False).iloc[0]


def rule_search(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selected_rows = []
    gap_grid = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00]
    delta_grid = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50, 1.00]
    rule_specs = [
        ("gru_rule", "gru_centered_score", "gru_score_delta_from_top1"),
        ("window_rule", "window_gru_score", "window_score_delta_from_top1"),
        ("avg_aux_rank_rule", "avg_aux_rank", "gru_score_delta_from_top1"),
    ]
    grouped = list(candidates.groupby(["fold", "date"], sort=True))
    for rule_name, score_col, delta_col in rule_specs:
        for gap in gap_grid:
            for delta in delta_grid:
                daily = []
                for (_, _), group in grouped:
                    if rule_name == "avg_aux_rank_rule":
                        top1 = group[group["candidate_rank"] == 1].iloc[0]
                        valid = group[
                            (group["candidate_rank"] != 1)
                            & (group["anchor_spread_delta_from_top1"] <= gap)
                            & (group["gru_score_delta_from_top1"] >= delta)
                            & (group["window_score_delta_from_top1"] >= delta)
                        ].copy()
                        pick = top1 if valid.empty else valid.sort_values("avg_aux_rank").iloc[0]
                    else:
                        pick = _select_candidate_by_rule(
                            group,
                            score_col=score_col,
                            score_delta_col=delta_col,
                            anchor_gap_max=gap,
                            aux_delta_min=delta,
                        )
                    daily.append(pick)
                selected = pd.DataFrame(daily)
                anchor = candidates[candidates["candidate_rank"] == 1][
                    ["fold", "date", "true_profit"]
                ].rename(columns={"true_profit": "anchor_profit"})
                selected = selected.merge(anchor, on=["fold", "date"], how="left")
                selected["method"] = f"{rule_name}_gap{gap:g}_delta{delta:g}"
                selected["anchor_gap_max"] = gap
                selected["aux_delta_min"] = delta
                selected["changed"] = (selected["candidate_rank"] != 1).astype(int)
                selected["lift_vs_anchor"] = selected["true_profit"] - selected["anchor_profit"]
                selected_rows.append(
                    selected[
                        [
                            "fold",
                            "date",
                            "method",
                            "anchor_gap_max",
                            "aux_delta_min",
                            "candidate_rank",
                            "charge_start",
                            "discharge_start",
                            "true_profit",
                            "anchor_profit",
                            "changed",
                            "lift_vs_anchor",
                        ]
                    ].rename(columns={"true_profit": "profit"})
                )
                by_fold = selected.groupby("fold").agg(mean_profit=("true_profit", "mean"))
                rows.append(
                    {
                        "method": selected["method"].iloc[0],
                        "rule_family": rule_name,
                        "anchor_gap_max": gap,
                        "aux_delta_min": delta,
                        "mean_profit": float(selected["true_profit"].mean()),
                        "min_fold_profit": float(by_fold["mean_profit"].min()),
                        "loss_days": int((selected["true_profit"] < 0).sum()),
                        "changed_days": int(selected["changed"].sum()),
                        "positive_changed_days": int(
                            ((selected["changed"] == 1) & (selected["lift_vs_anchor"] > 0)).sum()
                        ),
                        "negative_changed_days": int(
                            ((selected["changed"] == 1) & (selected["lift_vs_anchor"] < 0)).sum()
                        ),
                        "profit_delta_vs_anchor": float(selected["lift_vs_anchor"].mean()),
                        "positive_folds": int((by_fold["mean_profit"] > by_fold.loc[:, "mean_profit"].index.map(lambda _: -np.inf)).sum()) if False else 0,
                    }
                )
    summary = pd.DataFrame(rows).sort_values(
        ["profit_delta_vs_anchor", "loss_days", "changed_days"], ascending=[False, True, True]
    )
    return summary, pd.concat(selected_rows, ignore_index=True)


def add_positive_fold_counts(rule_summary: pd.DataFrame, rule_selected: pd.DataFrame) -> pd.DataFrame:
    anchor = rule_selected.groupby(["fold", "date"], as_index=False)["anchor_profit"].first()
    anchor_fold = anchor.groupby("fold")["anchor_profit"].mean().rename("anchor_fold_profit")
    rows = []
    for method, group in rule_selected.groupby("method"):
        fold_profit = group.groupby("fold")["profit"].mean().rename("fold_profit").to_frame()
        fold_profit = fold_profit.merge(anchor_fold, left_index=True, right_index=True)
        rows.append(
            {
                "method": method,
                "positive_folds": int((fold_profit["fold_profit"] > fold_profit["anchor_fold_profit"]).sum()),
            }
        )
    counts = pd.DataFrame(rows)
    return rule_summary.drop(columns=["positive_folds"], errors="ignore").merge(counts, on="method", how="left")


def write_report(
    *,
    daily: pd.DataFrame,
    overall: pd.DataFrame,
    by_fold: pd.DataFrame,
    rules: pd.DataFrame,
) -> None:
    best_rules = rules.head(15).copy()
    signal_rows = [
        {
            "metric": "gru_global_in_anchor_top10_rate",
            "value": float(daily["gru_global_in_anchor_top10"].mean()),
        },
        {
            "metric": "window_global_in_anchor_top10_rate",
            "value": float(daily["window_global_in_anchor_top10"].mean()),
        },
        {
            "metric": "gru_window_same_pick_rate_in_anchor_top10",
            "value": float(daily["gru_window_same_pick"].mean()),
        },
        {
            "metric": "mean_oracle_top10_lift",
            "value": float(daily["oracle_top10_lift"].mean()),
        },
        {
            "metric": "positive_oracle_top10_lift_days",
            "value": int((daily["oracle_top10_lift"] > 0).sum()),
        },
    ]
    signal = pd.DataFrame(signal_rows)
    lines = [
        "# GRU Centered / Window GRU Top10 Auxiliary Diagnostics",
        "",
        "## Scope",
        "",
        "- Candidate pool: current champion top10 legal charge/discharge pairs, with dispatch prior.",
        "- Auxiliary experts: GRU centered point curve and Window GRU 89-window scores.",
        "- This is diagnostic only; no submit file is generated.",
        "",
        "## Signal Checks",
        "",
        markdown_table(signal),
        "",
        "## Overall Methods",
        "",
        markdown_table(overall.sort_values("profit_delta_vs_anchor", ascending=False), limit=20),
        "",
        "## By Fold",
        "",
        markdown_table(by_fold.sort_values(["method", "fold"]), limit=80),
        "",
        "## Best Conservative Rules",
        "",
        markdown_table(best_rules, limit=15),
        "",
        "## Decision Notes",
        "",
    ]
    best = rules.iloc[0]
    if float(best["profit_delta_vs_anchor"]) > 100 and int(best["loss_days"]) <= int(
        overall.loc[overall["method"] == "anchor_top1", "loss_days"].iloc[0]
    ):
        lines.append("- At least one conservative rule clears the +100/day threshold; next step is fold-level robustness review before any submit implementation.")
    elif float(best["profit_delta_vs_anchor"]) > 0:
        lines.append("- Some rules are positive, but current signal is below promotion threshold or risk constraints need review.")
    else:
        lines.append("- No conservative auxiliary rule beats anchor safely; keep as diagnostic.")
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    candidates, daily = build_dataset()
    selected = selection_frame(daily)
    overall = summarize_selection(selected, ["method"])
    by_fold = summarize_selection(selected, ["fold", "method"])
    rules, rule_selected = rule_search(candidates)
    rules = add_positive_fold_counts(rules, rule_selected)

    candidates.to_csv(DATASET_PATH, index=False)
    selected.to_csv(DAILY_PATH, index=False)
    by_fold.to_csv(BY_FOLD_PATH, index=False)
    overall.to_csv(OVERALL_PATH, index=False)
    rules.to_csv(RULE_PATH, index=False)
    write_report(daily=daily, overall=overall, by_fold=by_fold, rules=rules)
    print(overall.sort_values("profit_delta_vs_anchor", ascending=False).to_string(index=False))
    print("best_rules")
    print(rules.head(10).to_string(index=False))
    print(f"report_path={MD_PATH}")


if __name__ == "__main__":
    main()
