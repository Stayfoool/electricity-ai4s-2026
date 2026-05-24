from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from electricity.features.bid_space import markdown_table

REPORTS_DIR = Path("reports")

CANDIDATES_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_foldsafe_dataset.csv"
CONSERVATIVE_SWITCH_PATH = (
    REPORTS_DIR / "current_prior_topk_conservative_switch_selected_daily.csv"
)
RERANKER_SELECTED_PATH = (
    REPORTS_DIR / "current_prior_topk_pair_reranker_foldsafe_selected_daily.csv"
)
CONSENSUS_PATH = REPORTS_DIR / "current_prior_topk_consensus_daily.csv"
PRIOR_DISABLED_PATH = REPORTS_DIR / "backtest_ens_champion_segmented6_daily.csv"

SELECTED_PATH = REPORTS_DIR / "bad_day_action_not_no_trade_selected_daily.csv"
RULES_PATH = REPORTS_DIR / "bad_day_action_not_no_trade_rules.csv"
RULE_SEARCH_PATH = REPORTS_DIR / "bad_day_action_not_no_trade_rule_search.csv"
OVERALL_PATH = REPORTS_DIR / "bad_day_action_not_no_trade_overall.csv"
BY_REGIME_PATH = REPORTS_DIR / "bad_day_action_not_no_trade_by_regime.csv"
BY_FOLD_PATH = REPORTS_DIR / "bad_day_action_not_no_trade_by_fold.csv"
ACTION_POOL_PATH = REPORTS_DIR / "bad_day_action_not_no_trade_action_pool.csv"
OUT_MD = REPORTS_DIR / "bad_day_action_not_no_trade_diagnostics.md"

MIN_CHANGED_DAYS = 2
MAX_CHANGED_RATE = 0.15
MIN_TRAIN_LIFT_PER_DAY = 10.0
QUANTILES = [0.05, 0.10, 0.20, 0.33, 0.50, 0.67, 0.80, 0.90, 0.95]

REGIMES = {
    "standard_09_12": {"valid_2025_09", "valid_2025_10", "valid_2025_11", "valid_2025_12"},
    "jan_feb_like": {"valid_2025_jan_feb"},
}

BASE_RISK_FEATURES = [
    "anchor_predicted_spread",
    "top1_top2_gap",
    "top1_top5_mean_gap",
    "top5_spread_std",
    "support_count",
    "rank_score",
    "mean_rank",
    "support_present_cols_sum",
    "lgb_segmented_6_last_180d_rank",
    "lgb_segmented_6_last_180d_predicted_spread",
    "lgb_segmented_6_margin_core_last_180d_rank",
    "lgb_segmented_6_margin_core_last_180d_predicted_spread",
    "charge_start",
    "discharge_start",
    "gap_slots",
    "charge_hour",
    "discharge_hour",
    "month",
    "dayofweek",
    "is_weekend",
    "diff_bid_space_mean",
    "diff_bid_space_rank_pct",
    "day_bid_space_std",
    "day_bid_space_range",
    "diff_net_load_mean",
    "diff_net_load_rank_pct",
    "day_net_load_std",
    "day_net_load_range",
    "diff_renewable_ratio_mean",
    "diff_renewable_ratio_rank_pct",
    "day_renewable_ratio_std",
    "day_renewable_ratio_range",
    "diff_wind_ratio_rank_pct",
    "diff_solar_ratio_rank_pct",
    "diff_tie_line_ratio_rank_pct",
    "diff_non_market_ratio_rank_pct",
    "diff_nwp_wind_speed100_mean_mh_resid_mean",
    "diff_nwp_wind_speed100_mean_mh_resid_rank_pct",
    "day_nwp_wind_speed100_mean_mh_resid_std",
    "diff_nwp_ghi_mean_mh_resid_mean",
    "diff_nwp_ghi_mean_mh_resid_rank_pct",
    "day_nwp_ghi_mean_mh_resid_std",
    "diff_nwp_tcc_mean_mh_resid_mean",
    "diff_nwp_tcc_mean_mh_resid_rank_pct",
    "day_nwp_tcc_mean_mh_resid_std",
    "diff_nwp_sp_mean_mh_resid_mean",
    "diff_nwp_sp_mean_mh_resid_rank_pct",
    "day_nwp_sp_mean_mh_resid_std",
]

AGG_RISK_FEATURES = [
    "top10_predicted_spread_std",
    "top10_predicted_spread_range",
    "top10_charge_start_std",
    "top10_discharge_start_std",
    "top10_charge_start_range",
    "top10_discharge_start_range",
    "top10_support_count_max",
    "top10_support_count_mean",
    "top10_mean_rank_min",
    "top10_bidspace_score_max_delta",
    "top10_margin_score_max_delta",
    "top10_support_score_max_delta",
    "rank2_spread_delta_from_top1",
    "rank2_bidspace_rank_score_delta",
    "rank2_margin_score_delta",
    "rank2_support_score_delta",
]


@dataclass(frozen=True)
class Rule:
    action: str
    feature: str
    op: str
    quantile: float
    threshold: float


def _read_csv(path: Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


def _ensure_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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
    out["bidspace_rank_score_raw"] = (
        out.get("diff_bid_space_rank_pct", 0).fillna(0).astype(float)
        - 0.01 * out["candidate_rank"].astype(float)
    )
    out["margin_combo_score_raw"] = (
        0.45 * out.get("diff_bid_space_rank_pct", 0).fillna(0).astype(float)
        + 0.35 * out.get("diff_net_load_rank_pct", 0).fillna(0).astype(float)
        - 0.15 * out.get("diff_renewable_ratio_rank_pct", 0).fillna(0).astype(float)
        + 0.05 * out["support_count_safe"]
        - 0.01 * out["candidate_rank"].astype(float)
    )
    for score in ["support_score", "bidspace_rank_score", "margin_combo_score"]:
        out[score] = out.groupby(["fold", "date"], sort=False)[f"{score}_raw"].rank(pct=True)
    return out


def _changed_from_anchor(df: pd.DataFrame) -> pd.Series:
    return (
        (df["charge_start"].astype(float) != df["top1_charge_start"].astype(float))
        | (df["discharge_start"].astype(float) != df["top1_discharge_start"].astype(float))
        | (df["candidate_rank"].fillna(1).astype(float) != 1.0)
    )


def _normalize_candidate_action(df: pd.DataFrame, action: str) -> pd.DataFrame:
    keep_cols = [
        "fold",
        "date",
        "candidate_rank",
        "charge_start",
        "discharge_start",
        "true_profit",
        "top1_true_profit",
        "top1_charge_start",
        "top1_discharge_start",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out["action"] = action
    out["profit"] = out["true_profit"].astype(float)
    out["anchor_profit"] = out["top1_true_profit"].astype(float)
    out["lift_vs_anchor"] = out["profit"] - out["anchor_profit"]
    out["changed_from_anchor"] = _changed_from_anchor(out).astype(int)
    return out


def _normalize_selected_action(
    df: pd.DataFrame,
    *,
    action: str,
    method_col: str,
    method_value: str,
) -> pd.DataFrame:
    sub = df[df[method_col] == method_value].copy()
    if sub.empty:
        return sub

    if "top1_true_profit" not in sub.columns and "anchor_top1_profit" in sub.columns:
        sub["top1_true_profit"] = sub["anchor_top1_profit"]
    if "candidate_rank" not in sub.columns and "anchor_rank" in sub.columns:
        sub["candidate_rank"] = sub["anchor_rank"]
    if "top1_charge_start" not in sub.columns:
        sub["top1_charge_start"] = np.nan
    if "top1_discharge_start" not in sub.columns:
        sub["top1_discharge_start"] = np.nan

    out = _normalize_candidate_action(sub, action)
    return out


def build_action_pool(candidates: pd.DataFrame) -> pd.DataFrame:
    scored = _ensure_scores(candidates)
    frames: list[pd.DataFrame] = []

    for rank in [2, 3]:
        rank_df = scored[scored["candidate_rank"] == rank].copy()
        frames.append(_normalize_candidate_action(rank_df, f"anchor_rank{rank}"))

    for action, score, max_rank in [
        ("aux_bidspace_rank_top3", "bidspace_rank_score", 3),
        ("aux_margin_combo_top3", "margin_combo_score", 3),
        ("aux_support_score_top3", "support_score", 3),
    ]:
        sub = scored[scored["candidate_rank"] <= max_rank].copy()
        selected = (
            sub.sort_values(
                ["fold", "date", score, "anchor_predicted_spread", "candidate_rank"],
                ascending=[True, True, False, False, True],
            )
            .drop_duplicates(["fold", "date"], keep="first")
            .copy()
        )
        frames.append(_normalize_candidate_action(selected, action))

    conservative = _read_csv(CONSERVATIVE_SWITCH_PATH, parse_dates=["date"])
    if not conservative.empty:
        frames.append(
            _normalize_selected_action(
                conservative,
                action="conservative_switch",
                method_col="method",
                method_value="conservative_switch",
            )
        )

    reranker = _read_csv(RERANKER_SELECTED_PATH, parse_dates=["date"])
    if not reranker.empty:
        for method in ["gbr_lift", "gbr_true_profit"]:
            frames.append(
                _normalize_selected_action(
                    reranker,
                    action=f"reranker_{method}",
                    method_col="method",
                    method_value=method,
                )
            )

    consensus = _read_csv(CONSENSUS_PATH, parse_dates=["date"])
    if not consensus.empty:
        for rule in [
            "support2_then_anchor_rank",
            "unanimous_then_anchor_rank",
            "support_score",
            "anchor_rank_le5_support2",
        ]:
            frames.append(
                _normalize_selected_action(
                    consensus,
                    action=f"consensus_{rule}",
                    method_col="rule",
                    method_value=rule,
                )
            )

    prior_disabled = _read_csv(PRIOR_DISABLED_PATH, parse_dates=["date"])
    if not prior_disabled.empty:
        anchor = candidates[candidates["candidate_rank"] == 1][
            [
                "fold",
                "date",
                "top1_true_profit",
                "top1_charge_start",
                "top1_discharge_start",
            ]
        ].drop_duplicates(["fold", "date"])
        prior = prior_disabled.merge(anchor, on=["fold", "date"], how="inner")
        prior["candidate_rank"] = np.nan
        prior["true_profit"] = prior["profit"]
        frames.append(_normalize_candidate_action(prior, "prior_disabled_champion"))

    pool = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    pool = pool.drop_duplicates(["fold", "date", "action"], keep="first")
    return pool.sort_values(["action", "fold", "date"]).reset_index(drop=True)


def build_daily_features(candidates: pd.DataFrame) -> pd.DataFrame:
    scored = _ensure_scores(candidates)
    top1 = scored[scored["candidate_rank"] == 1].copy()
    top1["anchor_profit"] = top1["top1_true_profit"].astype(float)
    oracle = (
        scored.groupby(["fold", "date"], as_index=False)["true_profit"]
        .max()
        .rename(columns={"true_profit": "oracle_top10_profit"})
    )
    top1 = top1.merge(oracle, on=["fold", "date"], how="left")
    top1["oracle_top10_lift"] = top1["oracle_top10_profit"] - top1["anchor_profit"]

    agg = (
        scored.groupby(["fold", "date"], as_index=False)
        .agg(
            top10_predicted_spread_std=("anchor_predicted_spread", "std"),
            top10_predicted_spread_min=("anchor_predicted_spread", "min"),
            top10_predicted_spread_max=("anchor_predicted_spread", "max"),
            top10_charge_start_std=("charge_start", "std"),
            top10_discharge_start_std=("discharge_start", "std"),
            top10_charge_start_min=("charge_start", "min"),
            top10_charge_start_max=("charge_start", "max"),
            top10_discharge_start_min=("discharge_start", "min"),
            top10_discharge_start_max=("discharge_start", "max"),
            top10_support_count_max=("support_count", "max"),
            top10_support_count_mean=("support_count", "mean"),
            top10_mean_rank_min=("mean_rank", "min"),
            top10_bidspace_score_max=("bidspace_rank_score", "max"),
            top10_margin_score_max=("margin_combo_score", "max"),
            top10_support_score_max=("support_score", "max"),
        )
        .copy()
    )
    agg["top10_predicted_spread_range"] = (
        agg["top10_predicted_spread_max"] - agg["top10_predicted_spread_min"]
    )
    agg["top10_charge_start_range"] = agg["top10_charge_start_max"] - agg["top10_charge_start_min"]
    agg["top10_discharge_start_range"] = (
        agg["top10_discharge_start_max"] - agg["top10_discharge_start_min"]
    )
    top1_scores = top1[
        ["fold", "date", "bidspace_rank_score", "margin_combo_score", "support_score"]
    ].rename(
        columns={
            "bidspace_rank_score": "__top1_bidspace_rank_score",
            "margin_combo_score": "__top1_margin_score",
            "support_score": "__top1_support_score",
        }
    )
    agg = agg.merge(top1_scores, on=["fold", "date"], how="left")
    agg["top10_bidspace_score_max_delta"] = (
        agg["top10_bidspace_score_max"] - agg["__top1_bidspace_rank_score"]
    )
    agg["top10_margin_score_max_delta"] = (
        agg["top10_margin_score_max"] - agg["__top1_margin_score"]
    )
    agg["top10_support_score_max_delta"] = (
        agg["top10_support_score_max"] - agg["__top1_support_score"]
    )

    rank2 = scored[scored["candidate_rank"] == 2][
        [
            "fold",
            "date",
            "anchor_spread_delta_from_top1",
            "bidspace_rank_score",
            "margin_combo_score",
            "support_score",
        ]
    ].rename(
        columns={
            "anchor_spread_delta_from_top1": "rank2_spread_delta_from_top1",
            "bidspace_rank_score": "rank2_bidspace_rank_score",
            "margin_combo_score": "rank2_margin_score",
            "support_score": "rank2_support_score",
        }
    )
    top1 = top1.merge(agg, on=["fold", "date"], how="left").merge(
        rank2,
        on=["fold", "date"],
        how="left",
    )
    top1["rank2_bidspace_rank_score_delta"] = (
        top1["rank2_bidspace_rank_score"] - top1["bidspace_rank_score"]
    )
    top1["rank2_margin_score_delta"] = top1["rank2_margin_score"] - top1["margin_combo_score"]
    top1["rank2_support_score_delta"] = top1["rank2_support_score"] - top1["support_score"]

    return top1.sort_values(["fold", "date"]).reset_index(drop=True)


def risk_features(df: pd.DataFrame) -> list[str]:
    blocked = {
        "true_profit",
        "oracle_profit",
        "regret_vs_oracle",
        "is_oracle_pair",
        "top1_true_profit",
        "target_lift",
        "anchor_profit",
        "oracle_top10_profit",
        "oracle_top10_lift",
    }
    candidates = BASE_RISK_FEATURES + AGG_RISK_FEATURES
    return [
        col
        for col in candidates
        if col in df.columns and col not in blocked and pd.api.types.is_numeric_dtype(df[col])
    ]


def _thresholds(series: pd.Series) -> list[tuple[float, float]]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.nunique() < 4:
        return []
    values = clean.quantile(QUANTILES)
    return sorted({(float(q), float(v)) for q, v in values.items()}, key=lambda x: x[1])


def _mask(df: pd.DataFrame, feature: str, op: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(df[feature], errors="coerce")
    if op == "<=":
        return values <= threshold
    if op == ">":
        return values > threshold
    raise ValueError(f"unsupported op={op}")


def _merge_action(daily: pd.DataFrame, action_pool: pd.DataFrame, action: str) -> pd.DataFrame:
    action_df = action_pool[action_pool["action"] == action][
        [
            "fold",
            "date",
            "profit",
            "lift_vs_anchor",
            "changed_from_anchor",
            "candidate_rank",
            "charge_start",
            "discharge_start",
        ]
    ].rename(
        columns={
            "profit": "action_profit",
            "lift_vs_anchor": "action_lift_vs_anchor",
            "changed_from_anchor": "action_changed_from_anchor",
            "candidate_rank": "action_candidate_rank",
            "charge_start": "action_charge_start",
            "discharge_start": "action_discharge_start",
        }
    )
    return daily.merge(action_df, on=["fold", "date"], how="left")


def evaluate_action_rule(
    daily: pd.DataFrame,
    action_pool: pd.DataFrame,
    rule: Rule,
) -> pd.DataFrame:
    work = _merge_action(daily, action_pool, rule.action)
    raw_mask = _mask(work, rule.feature, rule.op, rule.threshold)
    action_available = work["action_profit"].notna()
    trigger = raw_mask & action_available
    action_changed = work["action_changed_from_anchor"].fillna(0).astype(int) == 1
    effective_change = trigger & action_changed
    selected_profit = np.where(trigger, work["action_profit"], work["anchor_profit"])

    out = work.copy()
    out["selected_action"] = np.where(trigger, rule.action, "anchor_top1")
    out["selected_profit"] = selected_profit
    out["selected_lift_vs_anchor"] = out["selected_profit"] - out["anchor_profit"]
    out["triggered"] = trigger.astype(int)
    out["changed_from_anchor"] = effective_change.astype(int)
    out["rule_action"] = rule.action
    out["rule_feature"] = rule.feature
    out["rule_op"] = rule.op
    out["rule_quantile"] = rule.quantile
    out["rule_threshold"] = rule.threshold
    out["feature_value"] = pd.to_numeric(out[rule.feature], errors="coerce")
    return out


def _selection_metrics(selected: pd.DataFrame) -> dict[str, float]:
    profit = selected["selected_profit"].astype(float)
    anchor = selected["anchor_profit"].astype(float)
    lift = selected["selected_lift_vs_anchor"].astype(float)
    changed = selected["changed_from_anchor"].astype(int)
    return {
        "days": float(len(selected)),
        "mean_profit": float(profit.mean()),
        "anchor_mean_profit": float(anchor.mean()),
        "mean_lift_vs_anchor": float(lift.mean()),
        "total_lift_vs_anchor": float(lift.sum()),
        "worst_profit": float(profit.min()),
        "anchor_worst_profit": float(anchor.min()),
        "p10_profit": float(profit.quantile(0.1)),
        "anchor_p10_profit": float(anchor.quantile(0.1)),
        "loss_days": float((profit < 0).sum()),
        "anchor_loss_days": float((anchor < 0).sum()),
        "triggered_days": float(selected["triggered"].sum()),
        "changed_days": float(changed.sum()),
        "positive_lift_days": float((lift > 0).sum()),
        "negative_lift_days": float((lift < 0).sum()),
        "changed_rate": float(changed.mean()),
    }


def _passes_train_constraints(row: dict[str, float]) -> bool:
    if row["changed_days"] < MIN_CHANGED_DAYS:
        return False
    if row["changed_rate"] > MAX_CHANGED_RATE:
        return False
    if row["mean_lift_vs_anchor"] < MIN_TRAIN_LIFT_PER_DAY:
        return False
    if row["loss_days"] > row["anchor_loss_days"]:
        return False
    if row["worst_profit"] < row["anchor_worst_profit"] - 1e-9:
        return False
    if row["negative_lift_days"] > row["positive_lift_days"]:
        return False
    return True


def search_rules(
    train: pd.DataFrame,
    action_pool: pd.DataFrame,
    features: list[str],
    actions: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for action in actions:
        action_train = _merge_action(train, action_pool, action)
        if action_train["action_profit"].notna().sum() == 0:
            continue
        for feature in features:
            for quantile, threshold in _thresholds(train[feature]):
                for op in ["<=", ">"]:
                    rule = Rule(
                        action=action,
                        feature=feature,
                        op=op,
                        quantile=quantile,
                        threshold=threshold,
                    )
                    selected = evaluate_action_rule(train, action_pool, rule)
                    metrics = _selection_metrics(selected)
                    row: dict[str, object] = {
                        "action": action,
                        "feature": feature,
                        "op": op,
                        "quantile": quantile,
                        "threshold": threshold,
                    }
                    row.update(metrics)
                    row["passes_constraints"] = _passes_train_constraints(metrics)
                    rows.append(row)
    return pd.DataFrame(rows)


def choose_rule(search: pd.DataFrame) -> Rule | None:
    if search.empty:
        return None
    candidates = search[search["passes_constraints"]].copy()
    if candidates.empty:
        return None
    best = candidates.sort_values(
        [
            "mean_lift_vs_anchor",
            "negative_lift_days",
            "changed_days",
            "triggered_days",
            "loss_days",
        ],
        ascending=[False, True, True, True, True],
    ).iloc[0]
    return Rule(
        action=str(best["action"]),
        feature=str(best["feature"]),
        op=str(best["op"]),
        quantile=float(best["quantile"]),
        threshold=float(best["threshold"]),
    )


def fallback_rule(features: list[str]) -> Rule:
    return Rule(
        action="anchor_top1",
        feature=features[0],
        op=">",
        quantile=1.0,
        threshold=float("inf"),
    )


def crossfold_evaluate(
    daily: pd.DataFrame,
    action_pool: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_frames: list[pd.DataFrame] = []
    rule_rows: list[dict[str, object]] = []
    search_frames: list[pd.DataFrame] = []
    all_actions = sorted(action_pool["action"].unique())

    for heldout_fold in sorted(daily["fold"].unique()):
        train = daily[daily["fold"] != heldout_fold].copy()
        valid = daily[daily["fold"] == heldout_fold].copy()
        valid_available = set(
            action_pool.loc[action_pool["fold"] == heldout_fold, "action"].unique()
        )
        actions = [action for action in all_actions if action in valid_available]
        search = search_rules(train, action_pool, features, actions)
        search["heldout_fold"] = heldout_fold
        search_frames.append(search)

        rule = choose_rule(search)
        selected_from_train = rule is not None
        if rule is None:
            rule = fallback_rule(features)
            selected = valid.copy()
            selected["selected_action"] = "anchor_top1"
            selected["selected_profit"] = selected["anchor_profit"]
            selected["selected_lift_vs_anchor"] = 0.0
            selected["triggered"] = 0
            selected["changed_from_anchor"] = 0
            selected["rule_action"] = "anchor_top1"
            selected["rule_feature"] = rule.feature
            selected["rule_op"] = rule.op
            selected["rule_quantile"] = rule.quantile
            selected["rule_threshold"] = rule.threshold
            selected["feature_value"] = np.nan
            train_metrics = _selection_metrics(
                selected.assign(selected_profit=selected["anchor_profit"])
            )
        else:
            selected = evaluate_action_rule(valid, action_pool, rule)
            train_selected = evaluate_action_rule(train, action_pool, rule)
            train_metrics = _selection_metrics(train_selected)

        valid_metrics = _selection_metrics(selected)
        rule_rows.append(
            {
                "heldout_fold": heldout_fold,
                "selected_from_train": selected_from_train,
                "action": rule.action,
                "feature": rule.feature,
                "op": rule.op,
                "quantile": rule.quantile,
                "threshold": rule.threshold,
                "train_mean_lift_vs_anchor": train_metrics["mean_lift_vs_anchor"],
                "train_loss_days": train_metrics["loss_days"],
                "train_anchor_loss_days": train_metrics["anchor_loss_days"],
                "train_changed_days": train_metrics["changed_days"],
                "train_negative_lift_days": train_metrics["negative_lift_days"],
                "valid_mean_lift_vs_anchor": valid_metrics["mean_lift_vs_anchor"],
                "valid_loss_days": valid_metrics["loss_days"],
                "valid_anchor_loss_days": valid_metrics["anchor_loss_days"],
                "valid_changed_days": valid_metrics["changed_days"],
                "valid_negative_lift_days": valid_metrics["negative_lift_days"],
                "train_rows": len(train),
                "valid_rows": len(valid),
            }
        )
        selected_frames.append(selected)

    selected_all = pd.concat(selected_frames, ignore_index=True)
    search_all = pd.concat(search_frames, ignore_index=True)
    return selected_all, pd.DataFrame(rule_rows), search_all


def _selection_frame(selected: pd.DataFrame) -> pd.DataFrame:
    base = selected[
        ["fold", "date", "anchor_profit", "oracle_top10_profit", "selected_profit"]
    ].copy()
    anchor = base.copy()
    anchor["profit"] = anchor["anchor_profit"]
    anchor["method"] = "anchor_top1"
    anchor["lift_vs_anchor"] = 0.0
    anchor["changed_from_anchor"] = 0

    oracle = base.rename(columns={"oracle_top10_profit": "profit"})
    oracle["method"] = "oracle_top10"
    oracle["lift_vs_anchor"] = oracle["profit"] - oracle["anchor_profit"]
    oracle["changed_from_anchor"] = (oracle["lift_vs_anchor"].abs() > 1e-9).astype(int)

    bad_day = selected.rename(columns={"selected_profit": "profit"}).copy()
    bad_day["method"] = "bad_day_action_not_no_trade"
    bad_day["lift_vs_anchor"] = bad_day["profit"] - bad_day["anchor_profit"]

    cols = [
        "fold",
        "date",
        "method",
        "profit",
        "anchor_profit",
        "lift_vs_anchor",
        "changed_from_anchor",
    ]
    return pd.concat([anchor[cols], oracle[cols], bad_day[cols]], ignore_index=True)


def _regime_mask(df: pd.DataFrame, regime: str) -> pd.Series:
    folds = df["fold"].astype(str)
    if regime in REGIMES:
        return folds.isin(REGIMES[regime])
    if regime == "winter_11_12_jan_feb":
        dates = pd.to_datetime(df["date"])
        return (folds == "valid_2025_jan_feb") | dates.dt.month.isin([11, 12])
    if regime == "all_5fold":
        return pd.Series(True, index=df.index)
    raise ValueError(f"unknown regime={regime}")


def summarize_selection(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        profit = group["profit"].astype(float)
        lift = group["lift_vs_anchor"].astype(float)
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "days": len(group),
                "mean_profit": float(profit.mean()),
                "worst_profit": float(profit.min()),
                "p10_profit": float(profit.quantile(0.1)),
                "loss_days": int((profit < 0).sum()),
                "mean_lift_vs_anchor": float(lift.mean()),
                "positive_lift_days": int((lift > 0).sum()),
                "negative_lift_days": int((lift < 0).sum()),
                "changed_days": int(group["changed_from_anchor"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_regimes(selection: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for regime in ["standard_09_12", "jan_feb_like", "winter_11_12_jan_feb", "all_5fold"]:
        part = selection[_regime_mask(selection, regime)].copy()
        summary = summarize_selection(part, ["method"])
        summary.insert(0, "regime", regime)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def summarize_overall(by_regime: pd.DataFrame) -> pd.DataFrame:
    all_5fold = by_regime[by_regime["regime"] == "all_5fold"].copy()
    anchor = all_5fold[all_5fold["method"] == "anchor_top1"].iloc[0]
    all_5fold["profit_delta_vs_anchor"] = all_5fold["mean_profit"] - float(
        anchor["mean_profit"]
    )
    all_5fold["loss_day_delta_vs_anchor"] = all_5fold["loss_days"] - int(anchor["loss_days"])
    return all_5fold.sort_values("mean_profit", ascending=False)


def summarize_action_pool(action_pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action, group in action_pool.groupby("action", sort=True):
        profit = group["profit"].astype(float)
        lift = group["lift_vs_anchor"].astype(float)
        rows.append(
            {
                "action": action,
                "days_available": len(group),
                "mean_profit_if_always_used": float(profit.mean()),
                "mean_lift_vs_anchor": float(lift.mean()),
                "loss_days": int((profit < 0).sum()),
                "changed_days": int(group["changed_from_anchor"].sum()),
                "positive_lift_days": int((lift > 0).sum()),
                "negative_lift_days": int((lift < 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_lift_vs_anchor", ascending=False)


def write_report(
    *,
    selected: pd.DataFrame,
    rules: pd.DataFrame,
    search: pd.DataFrame,
    overall: pd.DataFrame,
    by_regime: pd.DataFrame,
    by_fold: pd.DataFrame,
    action_summary: pd.DataFrame,
    features: list[str],
) -> None:
    changed = selected[selected["changed_from_anchor"] == 1].copy()
    changed = changed.sort_values("selected_lift_vs_anchor")
    passing = search[search["passes_constraints"]].copy()
    if not passing.empty:
        passing = passing.sort_values(
            ["mean_lift_vs_anchor", "negative_lift_days", "changed_days"],
            ascending=[False, True, True],
        ).head(30)

    candidate = overall[overall["method"] == "bad_day_action_not_no_trade"].iloc[0]
    lines = [
        "# Bad-Day Action Not No-Trade",
        "",
        "## Scope",
        "",
        "- Anchor: current online champion-prior top1 pair.",
        "- Candidate actions: rank2/rank3, conservative switch, consensus rules, "
        "fold-safe reranker selections, auxiliary bid_space/support selections, and "
        "prior-disabled champion where available.",
        "- Objective: identify high-risk days but switch to another pair instead of no-trade.",
        "- Validation: each held-out fold learns its rule only from the other folds.",
        "- Rule inputs use prediction-stage features only; true prices/profits are used only "
        "for scoring the held-out fold.",
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
        "## Selected Rules",
        "",
        markdown_table(rules, floatfmt=".4f"),
        "",
        "## Changed-Day Audit",
        "",
        markdown_table(
            changed[
                [
                    "date",
                    "fold",
                    "selected_action",
                    "rule_feature",
                    "rule_op",
                    "rule_threshold",
                    "feature_value",
                    "anchor_profit",
                    "selected_profit",
                    "selected_lift_vs_anchor",
                    "oracle_top10_lift",
                ]
            ],
            floatfmt=".4f",
        ),
        "",
        "## Action Pool If Always Used",
        "",
        markdown_table(action_summary, floatfmt=".4f"),
        "",
        "## Top Passing Train Rules",
        "",
        markdown_table(passing, floatfmt=".4f"),
        "",
        "## Risk Features Used",
        "",
        ", ".join(f"`{feature}`" for feature in features),
        "",
        "## Decision",
        "",
    ]
    if (
        float(candidate["profit_delta_vs_anchor"]) >= 100
        and int(candidate["loss_day_delta_vs_anchor"]) <= 0
    ):
        lines += [
            "- This rule family passes the local promotion threshold.",
            "- Next step: inspect changed days manually, then implement submit-time selector.",
        ]
    elif float(candidate["profit_delta_vs_anchor"]) > 0:
        lines += [
            "- The result is directionally positive but below the +100/day promotion threshold.",
            "- Keep it as diagnostic evidence; do not generate a submit from this rule yet.",
        ]
    else:
        lines += [
            "- This bad-day action selector does not improve over champion top1.",
            "- Do not promote it; use the changed-day audit to refine the next selector.",
        ]
    lines += [
        "",
        "## Artifacts",
        "",
        f"- `{SELECTED_PATH}`",
        f"- `{RULES_PATH}`",
        f"- `{RULE_SEARCH_PATH}`",
        f"- `{OVERALL_PATH}`",
        f"- `{BY_REGIME_PATH}`",
        f"- `{BY_FOLD_PATH}`",
        f"- `{ACTION_POOL_PATH}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    candidates = _read_csv(CANDIDATES_PATH, parse_dates=["date"])
    if candidates.empty:
        raise FileNotFoundError(CANDIDATES_PATH)

    daily = build_daily_features(candidates)
    features = risk_features(daily)
    action_pool = build_action_pool(candidates)
    selected, rules, search = crossfold_evaluate(daily, action_pool, features)
    selection = _selection_frame(selected)
    by_regime = summarize_regimes(selection)
    overall = summarize_overall(by_regime)
    by_fold = summarize_selection(selection, ["fold", "method"])
    action_summary = summarize_action_pool(action_pool)

    selected.to_csv(SELECTED_PATH, index=False)
    rules.to_csv(RULES_PATH, index=False)
    search.to_csv(RULE_SEARCH_PATH, index=False)
    overall.to_csv(OVERALL_PATH, index=False)
    by_regime.to_csv(BY_REGIME_PATH, index=False)
    by_fold.to_csv(BY_FOLD_PATH, index=False)
    action_summary.to_csv(ACTION_POOL_PATH, index=False)
    write_report(
        selected=selected,
        rules=rules,
        search=search,
        overall=overall,
        by_regime=by_regime,
        by_fold=by_fold,
        action_summary=action_summary,
        features=features,
    )

    print(overall.to_string(index=False))
    print(f"markdown_path={OUT_MD}")


if __name__ == "__main__":
    main()
