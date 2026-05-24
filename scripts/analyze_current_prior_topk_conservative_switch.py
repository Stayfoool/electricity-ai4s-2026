from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import analyze_current_prior_topk_pair_reranker as base
import pandas as pd

from electricity.features.bid_space import markdown_table

REPORTS_DIR = Path("reports")
DATASET_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_foldsafe_dataset.csv"
SELECTED_PATH = REPORTS_DIR / "current_prior_topk_conservative_switch_selected_daily.csv"
RULES_PATH = REPORTS_DIR / "current_prior_topk_conservative_switch_rules.csv"
ALL_RULES_PATH = REPORTS_DIR / "current_prior_topk_conservative_switch_rule_search.csv"
FOLD_SUMMARY_PATH = REPORTS_DIR / "current_prior_topk_conservative_switch_by_fold.csv"
REGIME_SUMMARY_PATH = REPORTS_DIR / "current_prior_topk_conservative_switch_by_regime.csv"
OVERALL_PATH = REPORTS_DIR / "current_prior_topk_conservative_switch_overall.csv"
OUT_MD = REPORTS_DIR / "current_prior_topk_conservative_switch_diagnostics.md"


@dataclass(frozen=True)
class Rule:
    method: str
    score_name: str
    gate_feature: str
    gate_quantile: float
    gate_threshold: float
    max_rank: int
    score_margin: float


SCORE_NAMES = [
    "support_score",
    "bidspace_rank_score",
    "margin_combo_score",
]
GATE_FEATURES = [
    "top1_top2_gap",
    "top5_spread_std",
]
GATE_QUANTILES = [0.20, 0.33]
MAX_RANKS = [3, 5]
SCORE_MARGINS = [0.0, 0.10]
MAX_CHANGED_RATE = 0.10
MIN_TRAIN_LIFT_PER_DAY = 5.0


def _ensure_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["support2_int"] = out.get("support2", False).fillna(False).astype(int)
    out["unanimous_int"] = out.get("unanimous", False).fillna(False).astype(int)
    out["support_count_safe"] = out.get("support_count", 0).fillna(0).astype(float)
    out["mean_rank_safe"] = out.get("mean_rank", 99).fillna(99).astype(float)

    out["support_score_raw"] = (
        0.35 * out["support_count_safe"]
        + 0.25 * out["support2_int"]
        + 0.25 * out["unanimous_int"]
        - 0.02 * out["mean_rank_safe"]
        - 0.01 * out["candidate_rank"].astype(float)
    )
    out["expert_rank_score_raw"] = (
        -0.08 * out["mean_rank_safe"]
        + 0.20 * out["support_count_safe"]
        - 0.01 * out["candidate_rank"].astype(float)
    )
    out["bidspace_rank_score_raw"] = (
        out.get("diff_bid_space_rank_pct", 0).fillna(0).astype(float)
        - 0.01 * out["candidate_rank"].astype(float)
    )
    out["netload_rank_score_raw"] = (
        out.get("diff_net_load_rank_pct", 0).fillna(0).astype(float)
        - 0.01 * out["candidate_rank"].astype(float)
    )
    out["margin_combo_score_raw"] = (
        0.45 * out.get("diff_bid_space_rank_pct", 0).fillna(0).astype(float)
        + 0.35 * out.get("diff_net_load_rank_pct", 0).fillna(0).astype(float)
        - 0.15 * out.get("diff_renewable_ratio_rank_pct", 0).fillna(0).astype(float)
        + 0.05 * out["support_count_safe"]
        - 0.01 * out["candidate_rank"].astype(float)
    )
    out["renewable_inverse_score_raw"] = (
        -out.get("diff_renewable_ratio_rank_pct", 0).fillna(0).astype(float)
        - 0.01 * out["candidate_rank"].astype(float)
    )

    for score in SCORE_NAMES:
        raw_col = f"{score}_raw"
        out[score] = out.groupby(["fold", "date"], sort=False)[raw_col].rank(pct=True)
    return out


def _anchor_selection(df: pd.DataFrame, *, method: str = "anchor_top1") -> pd.DataFrame:
    selected = (
        df[df["candidate_rank"] == 1]
        .sort_values(["fold", "date"])
        .copy()
        .reset_index(drop=True)
    )
    selected["method"] = method
    selected["rule_id"] = method
    selected["threshold"] = 0.0
    selected["lift_vs_anchor"] = selected["true_profit"] - selected["top1_true_profit"]
    selected["changed_from_anchor"] = 0
    oracle = (
        df.groupby(["fold", "date"], as_index=False)["true_profit"]
        .max()
        .rename(columns={"true_profit": "oracle_top10_profit"})
    )
    selected = selected.merge(oracle, on=["fold", "date"], how="left")
    selected["miss_vs_oracle_top10"] = (
        selected["oracle_top10_profit"] - selected["true_profit"]
    )
    return selected


def _oracle_selection(df: pd.DataFrame) -> pd.DataFrame:
    selected = (
        df.sort_values(
            ["fold", "date", "true_profit", "candidate_rank"],
            ascending=[True, True, False, True],
        )
        .groupby(["fold", "date"], as_index=False)
        .head(1)
        .copy()
        .reset_index(drop=True)
    )
    selected["method"] = "oracle_top10"
    selected["rule_id"] = "oracle_top10"
    selected["threshold"] = -1_000_000.0
    selected["lift_vs_anchor"] = selected["true_profit"] - selected["top1_true_profit"]
    selected["changed_from_anchor"] = (selected["candidate_rank"] != 1).astype(int)
    selected["oracle_top10_profit"] = selected["true_profit"]
    selected["miss_vs_oracle_top10"] = 0.0
    return selected.sort_values(["fold", "date"]).reset_index(drop=True)


def _top1_gate_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["candidate_rank"] == 1][
        ["fold", "date", *GATE_FEATURES]
    ].drop_duplicates(["fold", "date"])


def _make_rule(
    train: pd.DataFrame,
    *,
    score_name: str,
    gate_feature: str,
    gate_quantile: float,
    max_rank: int,
    score_margin: float,
) -> Rule:
    top1 = _top1_gate_frame(train)
    threshold = float(top1[gate_feature].quantile(gate_quantile))
    return Rule(
        method="conservative_switch",
        score_name=score_name,
        gate_feature=gate_feature,
        gate_quantile=gate_quantile,
        gate_threshold=threshold,
        max_rank=max_rank,
        score_margin=score_margin,
    )


def _rule_id(rule: Rule) -> str:
    return (
        f"switch_{rule.score_name}__gate_{rule.gate_feature}_q{rule.gate_quantile:.2f}"
        f"__rank{rule.max_rank}__margin{rule.score_margin:.2f}"
    )


def apply_rule(df: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    top1 = (
        df[df["candidate_rank"] == 1]
        .sort_values(["fold", "date"])
        .copy()
        .reset_index(drop=True)
    )
    top1_scores = top1[["fold", "date", rule.score_name, rule.gate_feature]].rename(
        columns={
            rule.score_name: "__top1_score",
            rule.gate_feature: "__top1_gate",
        }
    )
    work = df.merge(top1_scores, on=["fold", "date"], how="left")
    eligible = work[
        (work["candidate_rank"] > 1)
        & (work["candidate_rank"] <= rule.max_rank)
        & (work["__top1_gate"] <= rule.gate_threshold)
        & (work[rule.score_name] >= work["__top1_score"] + rule.score_margin)
    ].copy()

    if eligible.empty:
        selected = top1
    else:
        alternatives = (
            eligible.sort_values(
                ["fold", "date", rule.score_name, "anchor_predicted_spread", "candidate_rank"],
                ascending=[True, True, False, False, True],
            )
            .drop_duplicates(["fold", "date"], keep="first")
            .drop(columns=["__top1_score", "__top1_gate"])
        )
        top1_indexed = top1.set_index(["fold", "date"])
        alternatives_indexed = alternatives.set_index(["fold", "date"])
        top1_indexed.update(alternatives_indexed)
        selected = top1_indexed.reset_index().sort_values(["fold", "date"]).reset_index(drop=True)

    selected["method"] = rule.method
    selected["rule_id"] = _rule_id(rule)
    selected["threshold"] = rule.gate_threshold
    selected["lift_vs_anchor"] = selected["true_profit"] - selected["top1_true_profit"]
    selected["changed_from_anchor"] = (selected["candidate_rank"] != 1).astype(int)
    oracle = (
        df.groupby(["fold", "date"], as_index=False)["true_profit"]
        .max()
        .rename(columns={"true_profit": "oracle_top10_profit"})
    )
    selected = selected.merge(oracle, on=["fold", "date"], how="left")
    selected["miss_vs_oracle_top10"] = (
        selected["oracle_top10_profit"] - selected["true_profit"]
    )
    return selected.sort_values(["fold", "date"]).reset_index(drop=True)


def _selection_metrics(selected: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, float]:
    profit = selected["true_profit"].astype(float)
    lift = selected["lift_vs_anchor"].astype(float)
    anchor_profit = anchor["true_profit"].astype(float)
    return {
        "days": float(len(selected)),
        "mean_profit": float(profit.mean()),
        "anchor_mean_profit": float(anchor_profit.mean()),
        "mean_lift_vs_anchor": float(lift.mean()),
        "total_lift_vs_anchor": float(lift.sum()),
        "worst_profit": float(profit.min()),
        "p10_profit": float(profit.quantile(0.1)),
        "loss_days": float((profit < 0).sum()),
        "anchor_loss_days": float((anchor_profit < 0).sum()),
        "changed_days": float(selected["changed_from_anchor"].sum()),
        "positive_lift_days": float((lift > 0).sum()),
        "negative_lift_days": float((lift < 0).sum()),
        "changed_rate": float(selected["changed_from_anchor"].mean()),
    }


def evaluate_rule(train: pd.DataFrame, rule: Rule) -> dict[str, float | str]:
    selected = apply_rule(train, rule)
    anchor = _anchor_selection(train)
    metrics = _selection_metrics(selected, anchor)
    metrics.update(
        {
            "rule_id": selected["rule_id"].iloc[0],
            "score_name": rule.score_name,
            "gate_feature": rule.gate_feature,
            "gate_quantile": rule.gate_quantile,
            "gate_threshold": rule.gate_threshold,
            "max_rank": rule.max_rank,
            "score_margin": rule.score_margin,
        }
    )
    return metrics


def passes_constraints(row: pd.Series) -> bool:
    if row["changed_days"] < 1:
        return False
    if row["changed_rate"] > MAX_CHANGED_RATE:
        return False
    if row["mean_lift_vs_anchor"] < MIN_TRAIN_LIFT_PER_DAY:
        return False
    if row["loss_days"] > row["anchor_loss_days"]:
        return False
    if row["negative_lift_days"] > row["positive_lift_days"]:
        return False
    return True


def choose_rule(train: pd.DataFrame) -> tuple[Rule | None, pd.DataFrame]:
    rows = []
    for score_name in SCORE_NAMES:
        for gate_feature in GATE_FEATURES:
            for gate_quantile in GATE_QUANTILES:
                for max_rank in MAX_RANKS:
                    for score_margin in SCORE_MARGINS:
                        rule = _make_rule(
                            train,
                            score_name=score_name,
                            gate_feature=gate_feature,
                            gate_quantile=gate_quantile,
                            max_rank=max_rank,
                            score_margin=score_margin,
                        )
                        rows.append(evaluate_rule(train, rule))

    search = pd.DataFrame(rows)
    search["passes_constraints"] = search.apply(passes_constraints, axis=1)
    candidates = search[search["passes_constraints"]].copy()
    if candidates.empty:
        return None, search

    best = candidates.sort_values(
        [
            "mean_lift_vs_anchor",
            "negative_lift_days",
            "changed_days",
            "score_margin",
            "max_rank",
        ],
        ascending=[False, True, True, False, True],
    ).iloc[0]
    return (
        Rule(
            method="conservative_switch",
            score_name=str(best["score_name"]),
            gate_feature=str(best["gate_feature"]),
            gate_quantile=float(best["gate_quantile"]),
            gate_threshold=float(best["gate_threshold"]),
            max_rank=int(best["max_rank"]),
            score_margin=float(best["score_margin"]),
        ),
        search,
    )


def crossfold_evaluate(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_frames = []
    rule_rows = []
    search_frames = []
    for heldout_fold in sorted(dataset["fold"].unique()):
        train = dataset[dataset["fold"] != heldout_fold].copy()
        valid = dataset[dataset["fold"] == heldout_fold].copy()

        anchor = _anchor_selection(valid)
        oracle = _oracle_selection(valid)
        selected_frames.extend([anchor, oracle])

        rule, search = choose_rule(train)
        search["heldout_fold"] = heldout_fold
        search_frames.append(search)
        if rule is None:
            selected = anchor.copy()
            selected["method"] = "conservative_switch"
            selected["rule_id"] = "fallback_anchor_top1"
            selected["threshold"] = 0.0
            rule_rows.append(
                {
                    "heldout_fold": heldout_fold,
                    "rule_id": "fallback_anchor_top1",
                    "selected_from_train": False,
                    "valid_rows": len(valid),
                    "train_rows": len(train),
                }
            )
        else:
            selected = apply_rule(valid, rule)
            train_selected = apply_rule(train, rule)
            train_anchor = _anchor_selection(train)
            train_metrics = _selection_metrics(train_selected, train_anchor)
            rule_rows.append(
                {
                    "heldout_fold": heldout_fold,
                    "rule_id": selected["rule_id"].iloc[0],
                    "selected_from_train": True,
                    "score_name": rule.score_name,
                    "gate_feature": rule.gate_feature,
                    "gate_quantile": rule.gate_quantile,
                    "gate_threshold": rule.gate_threshold,
                    "max_rank": rule.max_rank,
                    "score_margin": rule.score_margin,
                    "train_mean_lift_vs_anchor": train_metrics["mean_lift_vs_anchor"],
                    "train_changed_days": train_metrics["changed_days"],
                    "train_negative_lift_days": train_metrics["negative_lift_days"],
                    "train_loss_days": train_metrics["loss_days"],
                    "train_anchor_loss_days": train_metrics["anchor_loss_days"],
                    "valid_rows": len(valid),
                    "train_rows": len(train),
                }
            )
        selected_frames.append(selected)

    selected_all = pd.concat(selected_frames, ignore_index=True)
    selected_all["date"] = pd.to_datetime(selected_all["date"])
    rules = pd.DataFrame(rule_rows)
    search_all = pd.concat(search_frames, ignore_index=True)
    return selected_all, rules, search_all


def summarize_overall(by_regime: pd.DataFrame) -> pd.DataFrame:
    all_5fold = by_regime[by_regime["regime"] == "all_5fold"].copy()
    anchor_profit = float(
        all_5fold.loc[all_5fold["method"] == "anchor_top1", "mean_profit"].iloc[0]
    )
    anchor_loss = int(
        all_5fold.loc[all_5fold["method"] == "anchor_top1", "loss_days"].iloc[0]
    )
    all_5fold["profit_delta_vs_anchor"] = all_5fold["mean_profit"] - anchor_profit
    all_5fold["loss_day_delta_vs_anchor"] = all_5fold["loss_days"] - anchor_loss
    return all_5fold.sort_values("mean_profit", ascending=False)


def _changed_day_audit(selected: pd.DataFrame) -> pd.DataFrame:
    changed = selected[
        (selected["method"] == "conservative_switch")
        & (selected["changed_from_anchor"] == 1)
    ].copy()
    if changed.empty:
        return changed
    cols = [
        "date",
        "fold",
        "candidate_rank",
        "true_profit",
        "top1_true_profit",
        "lift_vs_anchor",
        "charge_start",
        "discharge_start",
        "top1_charge_start",
        "top1_discharge_start",
        "rule_id",
    ]
    return changed[cols].sort_values("lift_vs_anchor")


def write_report(
    *,
    selected: pd.DataFrame,
    rules: pd.DataFrame,
    search: pd.DataFrame,
    by_fold: pd.DataFrame,
    by_regime: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    rank_counts = (
        selected.groupby(["method", "candidate_rank"], as_index=False)
        .size()
        .rename(columns={"size": "days"})
    )
    changed_audit = _changed_day_audit(selected)
    passing = search[search["passes_constraints"]].copy()
    top_train_rules = passing.sort_values(
        ["mean_lift_vs_anchor", "negative_lift_days", "changed_days"],
        ascending=[False, True, True],
    ).head(20)
    switch = overall[overall["method"] == "conservative_switch"].iloc[0]

    lines = [
        "# Current Prior Top-K Conservative Switch",
        "",
        "## Scope",
        "",
        "- Anchor: current online champion-prior top1 pair.",
        "- Candidate set: champion-prior top10 legal charge/discharge pairs.",
        "- Goal: default to top1 and switch only on a small number of high-risk days.",
        "- Validation: for each held-out fold, choose one simple rule using only other folds.",
        "- No price model retraining and no submit file generation in this experiment.",
        "",
        "## Rule Family",
        "",
        "- Gate: switch is allowed only when a top1 confidence feature is below a training "
        "quantile threshold.",
        "- Candidate: among top-k alternatives, choose the highest auxiliary score.",
        "- Auxiliary scores use only prediction-time features: expert support/rank, bid_space, "
        "net_load, renewable ratio, and timing features.",
        "- Constraints: training changed rate <= 10%, training lift >= 5/day, loss days not "
        "above anchor, and negative changed days not above positive changed days.",
        "",
        "## Overall",
        "",
        markdown_table(overall, floatfmt=".4f"),
        "",
        "## Regime Summary",
        "",
        markdown_table(by_regime, floatfmt=".4f"),
        "",
        "## Fold Summary",
        "",
        markdown_table(by_fold, floatfmt=".4f"),
        "",
        "## Selected Rules By Held-Out Fold",
        "",
        markdown_table(rules, floatfmt=".4f"),
        "",
        "## Changed-Day Audit",
        "",
        markdown_table(changed_audit, floatfmt=".4f"),
        "",
        "## Selected Rank Counts",
        "",
        markdown_table(rank_counts, floatfmt=".4f"),
        "",
        "## Top Passing Train Rules",
        "",
        markdown_table(top_train_rules, floatfmt=".4f"),
        "",
        "## Decision",
        "",
    ]
    if (
        float(switch["profit_delta_vs_anchor"]) > 100
        and int(switch["loss_day_delta_vs_anchor"]) <= 0
    ):
        lines += [
            "- Conservative switching beats anchor by more than 100/day without "
            "increasing loss days.",
            "- Next step: inspect changed days manually, then consider submit-time implementation.",
        ]
    elif float(switch["profit_delta_vs_anchor"]) > 0:
        lines += [
            "- Conservative switching is positive but below the promotion threshold.",
            "- Keep it as a diagnostic; do not submit until the signal is stronger.",
        ]
    else:
        lines += [
            "- Conservative switching does not beat anchor top1.",
            "- Do not promote this rule family as currently defined.",
        ]
    lines += [
        "",
        "## Artifacts",
        "",
        f"- `{SELECTED_PATH}` rows: `{len(selected)}`",
        f"- `{RULES_PATH}`",
        f"- `{ALL_RULES_PATH}`",
        f"- `{FOLD_SUMMARY_PATH}`",
        f"- `{REGIME_SUMMARY_PATH}`",
        f"- `{OVERALL_PATH}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    dataset = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    dataset = _ensure_scores(dataset)
    selected, rules, search = crossfold_evaluate(dataset)
    by_fold = base.summarize_selection(selected, group_cols=["fold", "method"])
    by_regime = base.summarize_regimes(selected)
    overall = summarize_overall(by_regime)

    selected.to_csv(SELECTED_PATH, index=False)
    rules.to_csv(RULES_PATH, index=False)
    search.to_csv(ALL_RULES_PATH, index=False)
    by_fold.to_csv(FOLD_SUMMARY_PATH, index=False)
    by_regime.to_csv(REGIME_SUMMARY_PATH, index=False)
    overall.to_csv(OVERALL_PATH, index=False)
    write_report(
        selected=selected,
        rules=rules,
        search=search,
        by_fold=by_fold,
        by_regime=by_regime,
        overall=overall,
    )
    print(overall.to_string(index=False))
    print(f"markdown_path={OUT_MD}")


if __name__ == "__main__":
    main()
