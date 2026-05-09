from __future__ import annotations

from pathlib import Path

import pandas as pd

REPORTS_DIR = Path("reports")
INPUT_PATH = REPORTS_DIR / "topk_pair_candidates.csv"
MODEL_CHAMPION = "ens_champion_segmented6"
MODEL_SEGMENTED = "lgb_segmented_6_last_180d"
MODEL_MARGIN = "lgb_segmented_6_margin_core_last_180d"
MODELS = [MODEL_CHAMPION, MODEL_SEGMENTED, MODEL_MARGIN]
TOP_K = 10

RULES = [
    "champion_top1",
    "support_then_rank",
    "rank_score",
    "support2_then_rank",
    "unanimous_then_rank",
    "champion_anchored_support",
]


def load_candidates() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df[(df["model"].isin(MODELS)) & (df["candidate_rank"] <= TOP_K)].copy()
    if df.empty:
        raise ValueError(f"no top-k candidate rows loaded from {INPUT_PATH}")

    top1 = df[df["candidate_rank"] == 1][
        ["date", "model", "predicted_spread"]
    ].rename(columns={"predicted_spread": "model_top1_spread"})
    df = df.merge(top1, on=["date", "model"], how="left")
    df["rank_score"] = TOP_K + 1 - df["candidate_rank"]
    df["spread_delta_from_model_top1"] = df["model_top1_spread"] - df["predicted_spread"]
    return df


def aggregate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["fold", "date", "charge_start", "discharge_start"]
    agg = (
        df.groupby(index_cols, as_index=False)
        .agg(
            support_count=("model", "nunique"),
            mean_rank=("candidate_rank", "mean"),
            min_rank=("candidate_rank", "min"),
            max_rank=("candidate_rank", "max"),
            rank_score=("rank_score", "sum"),
            mean_predicted_spread=("predicted_spread", "mean"),
            max_predicted_spread=("predicted_spread", "max"),
            mean_spread_delta=("spread_delta_from_model_top1", "mean"),
            true_profit=("true_profit", "first"),
            regret_vs_oracle=("regret_vs_oracle", "first"),
            is_oracle_pair=("is_oracle_pair", "max"),
        )
        .sort_values(["date", "charge_start", "discharge_start"])
        .reset_index(drop=True)
    )

    for model in MODELS:
        model_rows = df[df["model"] == model][
            ["date", "charge_start", "discharge_start", "candidate_rank", "predicted_spread"]
        ].rename(
            columns={
                "candidate_rank": f"{model}_rank",
                "predicted_spread": f"{model}_predicted_spread",
            }
        )
        agg = agg.merge(model_rows, on=["date", "charge_start", "discharge_start"], how="left")
        agg[f"{model}_present"] = agg[f"{model}_rank"].notna().astype(int)

    agg["all_present"] = agg["support_count"] == len(MODELS)
    agg["any_non_champion_support"] = (
        agg[f"{MODEL_SEGMENTED}_present"] + agg[f"{MODEL_MARGIN}_present"]
    ) > 0
    return agg


def select_first(group: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> pd.Series:
    return group.sort_values(sort_cols, ascending=ascending).iloc[0]


def select_for_rule(group: pd.DataFrame, rule: str) -> pd.Series:
    champion = group[group[f"{MODEL_CHAMPION}_rank"] == 1]
    if champion.empty:
        raise ValueError(f"missing champion rank1 for date={group['date'].iloc[0]}")
    champion_top1 = champion.iloc[0]

    if rule == "champion_top1":
        return champion_top1

    if rule == "support_then_rank":
        return select_first(
            group,
            ["support_count", "mean_rank", "mean_spread_delta", "mean_predicted_spread"],
            [False, True, True, False],
        )

    if rule == "rank_score":
        return select_first(
            group,
            ["rank_score", "support_count", "mean_spread_delta", "mean_rank"],
            [False, False, True, True],
        )

    if rule == "support2_then_rank":
        supported = group[group["support_count"] >= 2]
        if supported.empty:
            return champion_top1
        return select_first(
            supported,
            ["support_count", "mean_rank", "mean_spread_delta", "mean_predicted_spread"],
            [False, True, True, False],
        )

    if rule == "unanimous_then_rank":
        unanimous = group[group["support_count"] == len(MODELS)]
        if unanimous.empty:
            return champion_top1
        return select_first(
            unanimous,
            ["mean_rank", "mean_spread_delta", "mean_predicted_spread"],
            [True, True, False],
        )

    if rule == "champion_anchored_support":
        supported = group[
            (group[f"{MODEL_CHAMPION}_present"] == 1)
            & (group["support_count"] >= 2)
            & (group[f"{MODEL_CHAMPION}_rank"] <= 5)
        ]
        if supported.empty:
            return champion_top1
        return select_first(
            supported,
            [f"{MODEL_CHAMPION}_rank", "support_count", "mean_rank", "mean_spread_delta"],
            [True, False, True, True],
        )

    raise ValueError(f"unknown rule={rule}")


def build_daily_selection(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (fold, date), group in candidates.groupby(["fold", "date"], sort=True):
        champion_top1_profit = float(
            group.loc[group[f"{MODEL_CHAMPION}_rank"] == 1, "true_profit"].iloc[0]
        )
        oracle_topk_profit = float(group["true_profit"].max())
        for rule in RULES:
            selected = select_for_rule(group, rule)
            true_profit = float(selected["true_profit"])
            rows.append(
                {
                    "fold": fold,
                    "date": date,
                    "rule": rule,
                    "charge_start": int(selected["charge_start"]),
                    "discharge_start": int(selected["discharge_start"]),
                    "support_count": int(selected["support_count"]),
                    "mean_rank": float(selected["mean_rank"]),
                    "rank_score": float(selected["rank_score"]),
                    "champion_rank": selected[f"{MODEL_CHAMPION}_rank"],
                    "segmented_rank": selected[f"{MODEL_SEGMENTED}_rank"],
                    "margin_rank": selected[f"{MODEL_MARGIN}_rank"],
                    "true_profit": true_profit,
                    "champion_top1_profit": champion_top1_profit,
                    "oracle_topk_profit": oracle_topk_profit,
                    "lift_vs_champion_top1": true_profit - champion_top1_profit,
                    "miss_vs_oracle_topk": oracle_topk_profit - true_profit,
                    "changed_from_champion": int(rule != "champion_top1")
                    * int(
                        (int(selected["charge_start"]) != int(group.loc[
                            group[f"{MODEL_CHAMPION}_rank"] == 1, "charge_start"
                        ].iloc[0]))
                        or (int(selected["discharge_start"]) != int(group.loc[
                            group[f"{MODEL_CHAMPION}_rank"] == 1, "discharge_start"
                        ].iloc[0]))
                    ),
                    "is_oracle_pair": bool(selected["is_oracle_pair"]),
                }
            )
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rule, group in daily.groupby("rule", sort=False):
        profit = group["true_profit"].astype(float)
        lift = group["lift_vs_champion_top1"].astype(float)
        rows.append(
            {
                "rule": rule,
                "days": len(group),
                "mean_profit": float(profit.mean()),
                "worst_profit": float(profit.min()),
                "loss_days": int((profit < 0).sum()),
                "mean_lift_vs_champion": float(lift.mean()),
                "positive_lift_days": int((lift > 0).sum()),
                "negative_lift_days": int((lift < 0).sum()),
                "changed_days": int(group["changed_from_champion"].sum()),
                "mean_support_count": float(group["support_count"].mean()),
                "oracle_hit_rate": float(group["is_oracle_pair"].mean()),
                "mean_miss_vs_oracle_topk": float(group["miss_vs_oracle_topk"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_profit", ascending=False)


def summarize_by_fold(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (fold, rule), group in daily.groupby(["fold", "rule"], sort=True):
        profit = group["true_profit"].astype(float)
        lift = group["lift_vs_champion_top1"].astype(float)
        rows.append(
            {
                "fold": fold,
                "rule": rule,
                "days": len(group),
                "mean_profit": float(profit.mean()),
                "worst_profit": float(profit.min()),
                "loss_days": int((profit < 0).sum()),
                "mean_lift_vs_champion": float(lift.mean()),
                "changed_days": int(group["changed_from_champion"].sum()),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, by_fold: pd.DataFrame) -> None:
    champion_mean = float(
        summary.loc[summary["rule"] == "champion_top1", "mean_profit"].iloc[0]
    )
    best = summary.iloc[0]
    lines = [
        "# Top-K Consensus Diagnostics",
        "",
        "## Scope",
        "",
        "Use the existing predicted top-10 legal charge/discharge pairs from three experts:",
        "",
        f"- `{MODEL_CHAMPION}`",
        f"- `{MODEL_SEGMENTED}`",
        f"- `{MODEL_MARGIN}`",
        "",
        "No new price model is trained. Each rule chooses one pair per day using only "
        "candidate ranks, predicted spreads, and cross-expert agreement.",
        "",
        "## Overall Result",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Fold Result",
        "",
        by_fold.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Decision",
        "",
    ]
    if float(best["mean_profit"]) > champion_mean + 50:
        lines += [
            "- At least one consensus rule materially beats champion top1.",
            "- Next step is to validate the rule as a submit-time dispatch option.",
        ]
    else:
        lines += [
            "- Simple cross-expert consensus does not materially beat champion top1.",
            "- Keep this as diagnostic evidence; the next reranking attempt should use "
            "pairwise preference learning or richer ex-ante features.",
        ]
    (REPORTS_DIR / "topk_consensus_diagnostics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    raw = load_candidates()
    candidates = aggregate_candidates(raw)
    daily = build_daily_selection(candidates)
    summary = summarize(daily)
    by_fold = summarize_by_fold(daily)

    candidates.to_csv(REPORTS_DIR / "topk_consensus_candidates.csv", index=False)
    daily.to_csv(REPORTS_DIR / "topk_consensus_daily.csv", index=False)
    summary.to_csv(REPORTS_DIR / "topk_consensus_summary.csv", index=False)
    by_fold.to_csv(REPORTS_DIR / "topk_consensus_by_fold.csv", index=False)
    write_markdown(summary, by_fold)

    print(summary.to_string(index=False))
    print(f"candidate_path={REPORTS_DIR / 'topk_consensus_candidates.csv'}")
    print(f"daily_path={REPORTS_DIR / 'topk_consensus_daily.csv'}")
    print(f"summary_path={REPORTS_DIR / 'topk_consensus_summary.csv'}")
    print(f"by_fold_path={REPORTS_DIR / 'topk_consensus_by_fold.csv'}")
    print(f"markdown_path={REPORTS_DIR / 'topk_consensus_diagnostics.md'}")


if __name__ == "__main__":
    main()
