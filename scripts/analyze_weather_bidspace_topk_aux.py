from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import analyze_current_prior_topk_pair_reranker as base
import pandas as pd

from electricity.features.bid_space import markdown_table

REPORTS_DIR = Path("reports")
DATASET_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_foldsafe_dataset.csv"
SELECTED_PATH = REPORTS_DIR / "weather_bidspace_topk_aux_selected_daily.csv"
RULES_PATH = REPORTS_DIR / "weather_bidspace_topk_aux_rules.csv"
ALL_RULES_PATH = REPORTS_DIR / "weather_bidspace_topk_aux_rule_search.csv"
FOLD_SUMMARY_PATH = REPORTS_DIR / "weather_bidspace_topk_aux_by_fold.csv"
REGIME_SUMMARY_PATH = REPORTS_DIR / "weather_bidspace_topk_aux_by_regime.csv"
OVERALL_PATH = REPORTS_DIR / "weather_bidspace_topk_aux_overall.csv"
OUT_MD = REPORTS_DIR / "weather_bidspace_topk_aux_diagnostics.md"

GATE_FEATURES = ["top1_top2_gap", "top5_spread_std"]
GATE_QUANTILES = [0.10, 0.20, 0.33]
MAX_RANKS = [2, 3, 5]
SCORE_MARGINS = [0.0, 0.05, 0.10, 0.20]
MAX_CHANGED_RATE = 0.10
MIN_TRAIN_LIFT_PER_DAY = 5.0
MAX_NEG_POS_RATIO = 1.0

CORE_SCORE_NAMES = [
    "bidspace_score",
    "netload_score",
    "weather_cloud_score",
    "weather_wind_score",
    "bid_weather_score",
    "bid_weather_support_score",
    "risk_consensus_score",
]


@dataclass(frozen=True)
class Rule:
    method: str
    score_name: str
    gate_feature: str
    gate_quantile: float
    gate_threshold: float
    max_rank: int
    score_margin: float


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _rank_pct_by_day(df: pd.DataFrame, raw_col: str, out_col: str) -> None:
    df[out_col] = df.groupby(["fold", "date"], sort=False)[raw_col].rank(pct=True)


def _ensure_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    rank_penalty = 0.01 * _num(out, "candidate_rank")
    support = _num(out, "support_count")
    mean_rank = _num(out, "mean_rank", 99.0)
    support2 = _num(out, "support2")
    unanimous = _num(out, "unanimous")

    # Core physical signals: discharge window should be high, charge window should be low.
    bid_rank = _num(out, "diff_bid_space_rank_pct")
    net_rank = _num(out, "diff_net_load_rank_pct")
    renewable_rank = _num(out, "diff_renewable_ratio_rank_pct")
    wind_ratio_rank = _num(out, "diff_wind_ratio_rank_pct")

    # Fold-safe weather month-hour residual signals built by the prior fold-safe dataset.
    tcc_rank = _num(out, "diff_nwp_tcc_mean_mh_resid_rank_pct")
    wind_mean_rank = _num(out, "diff_nwp_wind_speed100_mean_mh_resid_rank_pct")
    wind_max_rank = _num(out, "diff_nwp_wind_speed100_max_mh_resid_rank_pct")
    ghi_rank = _num(out, "diff_nwp_ghi_mean_mh_resid_rank_pct")
    u100_rank = _num(out, "diff_nwp_u100_mean_mh_resid_rank_pct")

    support_raw = 0.18 * support + 0.08 * support2 + 0.10 * unanimous - 0.01 * mean_rank
    out["bidspace_score_raw"] = bid_rank - rank_penalty
    out["netload_score_raw"] = net_rank - rank_penalty
    out["weather_cloud_score_raw"] = 0.65 * tcc_rank + 0.35 * ghi_rank - rank_penalty
    out["weather_wind_score_raw"] = (
        0.50 * wind_max_rank + 0.30 * wind_mean_rank + 0.20 * u100_rank - rank_penalty
    )
    out["bid_weather_score_raw"] = (
        0.35 * bid_rank
        + 0.30 * net_rank
        + 0.15 * tcc_rank
        + 0.15 * wind_max_rank
        - 0.10 * renewable_rank
        - rank_penalty
    )
    out["bid_weather_support_score_raw"] = out["bid_weather_score_raw"] + support_raw
    out["risk_consensus_score_raw"] = (
        0.30 * bid_rank
        + 0.25 * net_rank
        + 0.15 * tcc_rank
        + 0.15 * wind_max_rank
        - 0.10 * renewable_rank
        - 0.05 * wind_ratio_rank
        + support_raw
        - rank_penalty
    )

    for score_name in CORE_SCORE_NAMES:
        _rank_pct_by_day(out, f"{score_name}_raw", score_name)
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
    selected["miss_vs_oracle_top10"] = selected["oracle_top10_profit"] - selected["true_profit"]
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
        method="weather_bidspace_topk_aux",
        score_name=score_name,
        gate_feature=gate_feature,
        gate_quantile=gate_quantile,
        gate_threshold=threshold,
        max_rank=max_rank,
        score_margin=score_margin,
    )


def _rule_id(rule: Rule) -> str:
    return (
        f"aux_{rule.score_name}__gate_{rule.gate_feature}_q{rule.gate_quantile:.2f}"
        f"__rank{rule.max_rank}__margin{rule.score_margin:.2f}"
    )


def apply_rule(df: pd.DataFrame, rule: Rule, *, include_oracle: bool = True) -> pd.DataFrame:
    top1 = (
        df[df["candidate_rank"] == 1]
        .sort_values(["fold", "date"])
        .copy()
        .reset_index(drop=True)
    )
    top1_scores = top1[["fold", "date", rule.score_name, rule.gate_feature]].rename(
        columns={rule.score_name: "__top1_score", rule.gate_feature: "__top1_gate"}
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
        alt_keys = pd.MultiIndex.from_frame(alternatives[["fold", "date"]])
        top1_keys = pd.MultiIndex.from_frame(top1[["fold", "date"]])
        selected = pd.concat(
            [alternatives, top1.loc[~top1_keys.isin(alt_keys)]],
            ignore_index=True,
        ).sort_values(["fold", "date"]).reset_index(drop=True)

    selected["method"] = rule.method
    selected["rule_id"] = _rule_id(rule)
    selected["threshold"] = rule.gate_threshold
    selected["lift_vs_anchor"] = selected["true_profit"] - selected["top1_true_profit"]
    selected["changed_from_anchor"] = (selected["candidate_rank"] != 1).astype(int)
    if include_oracle:
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


def _selection_metrics(
    selected: pd.DataFrame,
    anchor: pd.DataFrame | None = None,
) -> dict[str, float]:
    profit = selected["true_profit"].astype(float)
    lift = selected["lift_vs_anchor"].astype(float)
    if anchor is None:
        anchor_profit = selected["top1_true_profit"].astype(float)
    else:
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


def evaluate_rule(train: pd.DataFrame, rule: Rule) -> dict[str, float | str | bool]:
    selected = apply_rule(train, rule, include_oracle=False)
    metrics = _selection_metrics(selected)
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
    if row["negative_lift_days"] > row["positive_lift_days"] * MAX_NEG_POS_RATIO:
        return False
    if row["worst_profit"] < -20_000:
        return False
    return True


def choose_rule(train: pd.DataFrame) -> tuple[Rule | None, pd.DataFrame]:
    rows = []
    for score_name in CORE_SCORE_NAMES:
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
            method="weather_bidspace_topk_aux",
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
            selected["method"] = "weather_bidspace_topk_aux"
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
            train_selected = apply_rule(train, rule, include_oracle=False)
            train_metrics = _selection_metrics(train_selected)
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
    return selected_all, pd.DataFrame(rule_rows), pd.concat(search_frames, ignore_index=True)


def summarize_overall(by_regime: pd.DataFrame) -> pd.DataFrame:
    all_5fold = by_regime[by_regime["regime"] == "all_5fold"].copy()
    anchor_profit = float(
        all_5fold.loc[all_5fold["method"] == "anchor_top1", "mean_profit"].iloc[0]
    )
    anchor_loss = int(all_5fold.loc[all_5fold["method"] == "anchor_top1", "loss_days"].iloc[0])
    all_5fold["profit_delta_vs_anchor"] = all_5fold["mean_profit"] - anchor_profit
    all_5fold["loss_day_delta_vs_anchor"] = all_5fold["loss_days"] - anchor_loss
    return all_5fold.sort_values("mean_profit", ascending=False)


def _changed_day_audit(selected: pd.DataFrame) -> pd.DataFrame:
    changed = selected[
        (selected["method"] == "weather_bidspace_topk_aux")
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
    aux = overall[overall["method"] == "weather_bidspace_topk_aux"].iloc[0]

    lines = [
        "# Weather + Bid-Space Top-K Auxiliary Switch",
        "",
        "## Scope",
        "",
        "- Anchor: current online champion-prior top1 pair.",
        "- Candidate set: champion-prior top10 legal charge/discharge pairs.",
        (
            "- Goal: keep champion price prediction unchanged and only use weather/bid_space "
            "as a small auxiliary signal inside top-k."
        ),
        "- Validation: for each held-out fold, choose one simple rule using only other folds.",
        "- No price model retraining and no submit file generation in this experiment.",
        "",
        "## Rule Family",
        "",
        "- Gate: switch is allowed only when top1 confidence is low.",
        (
            "- Scores: bid_space, net_load, cloud/GHI residual, wind residual, and "
            "conservative combinations with expert support."
        ),
        (
            "- Constraints: training changed rate <= 10%, training lift >= 5/day, "
            "loss days not above anchor, and negative changed days not above positive "
            "changed days."
        ),
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
    if float(aux["profit_delta_vs_anchor"]) > 100 and int(aux["loss_day_delta_vs_anchor"]) <= 0:
        lines += [
            (
                "- The auxiliary switch beats anchor by more than 100/day without "
                "increasing loss days."
            ),
            "- Next step: inspect changed days manually, then consider submit-time implementation.",
        ]
    elif float(aux["profit_delta_vs_anchor"]) > 0:
        lines += [
            "- The auxiliary switch is positive but below the promotion threshold.",
            "- Keep it as diagnostic evidence; do not submit until the signal is stronger.",
        ]
    else:
        lines += [
            "- The auxiliary switch does not beat anchor top1.",
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
