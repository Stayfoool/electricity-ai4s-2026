from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS_DIR = Path("reports")

ANCHOR = "champion_prior"
ANCHOR_PATH = REPORTS_DIR / "backtest_ens_champion_segmented6_prior_5fold_daily.csv"
EXPERT_PATHS = {
    "cat180": REPORTS_DIR / "backtest_cat_baseline_last_180d_daily.csv",
    "xgb180": REPORTS_DIR / "backtest_xgb_baseline_last_180d_daily.csv",
    "seg6": REPORTS_DIR / "backtest_lgb_segmented_6_last_180d_daily.csv",
    "margin6": REPORTS_DIR / "backtest_lgb_segmented_6_margin_core_last_180d_daily.csv",
}

STANDARD_FOLDS = ["valid_2025_09", "valid_2025_10", "valid_2025_11", "valid_2025_12"]

SELECTED_PATH = REPORTS_DIR / "model_family_consensus_fallback_selected_daily.csv"
RULES_PATH = REPORTS_DIR / "model_family_consensus_fallback_rules.csv"
SEARCH_PATH = REPORTS_DIR / "model_family_consensus_fallback_rule_search.csv"
OVERALL_PATH = REPORTS_DIR / "model_family_consensus_fallback_overall.csv"
BY_FOLD_PATH = REPORTS_DIR / "model_family_consensus_fallback_by_fold.csv"
BY_MONTH_PATH = REPORTS_DIR / "model_family_consensus_fallback_by_month.csv"
EXPERT_SUMMARY_PATH = REPORTS_DIR / "model_family_consensus_fallback_expert_summary.csv"
OUT_MD = REPORTS_DIR / "model_family_consensus_fallback_diagnostics.md"

GATE_FEATURES = [
    "anchor_top1_top2_gap",
    "anchor_top5_spread_std",
    "anchor_predicted_spread",
]
GATE_QUANTILES = [0.10, 0.20, 0.33, 0.50]
SCOPES = {
    "segmented_family": ["seg6", "margin6"],
    "margin6": ["margin6"],
    "seg6": ["seg6"],
    "cat_xgb": ["cat180", "xgb180"],
    "all_experts": ["seg6", "margin6", "cat180", "xgb180"],
}
SUPPORT_RULES = [
    ("support_near2_all", 3),
    ("support_near2_all", 4),
    ("support_near4_all", 3),
    ("support_near4_all", 4),
    ("support_near2_segmented", 2),
    ("support_near2_segmented", 3),
]
MODEL_PRIORITY = {"margin6": 0, "seg6": 1, "cat180": 2, "xgb180": 3}

MAX_CHANGED_RATE = 0.10
MIN_TRAIN_LIFT_PER_DAY = 5.0
MAX_NEG_POS_RATIO = 1.0
MAX_WORST_PROFIT_DROP = 1000.0


@dataclass(frozen=True)
class Rule:
    scope: str
    gate_feature: str
    gate_quantile: float
    gate_threshold: float
    support_col: str
    min_support: int


def markdown_table(df: pd.DataFrame, *, limit: int = 30, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_无数据_"
    block = df.head(limit).copy()
    cols = list(block.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in block.iterrows():
        cells: list[str] = []
        for col in cols:
            value = row[col]
            if isinstance(value, float | np.floating):
                text = "" if pd.isna(value) else format(float(value), floatfmt)
            else:
                text = "" if pd.isna(value) else str(value)
            cells.append(text.replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read_daily(path: Path, model: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["fold"].isin(STANDARD_FOLDS)].copy()
    df["model"] = model
    return df


def common_keys(frames: list[pd.DataFrame]) -> pd.DataFrame:
    keys = frames[0][["fold", "date"]].drop_duplicates()
    for frame in frames[1:]:
        keys = keys.merge(frame[["fold", "date"]].drop_duplicates(), on=["fold", "date"])
    return keys.sort_values(["fold", "date"]).reset_index(drop=True)


def _near_support(group: pd.DataFrame, row: pd.Series, models: set[str], distance: int) -> int:
    subset = group[group["model"].isin(models)]
    charge = int(row["charge_start"])
    discharge = int(row["discharge_start"])
    return int(
        (
            (subset["charge_start"].astype(int).sub(charge).abs() <= distance)
            & (subset["discharge_start"].astype(int).sub(discharge).abs() <= distance)
        ).sum()
    )


def _exact_support(group: pd.DataFrame, row: pd.Series, models: set[str]) -> int:
    subset = group[group["model"].isin(models)]
    return int(
        (
            (subset["charge_start"].astype(int) == int(row["charge_start"]))
            & (subset["discharge_start"].astype(int) == int(row["discharge_start"]))
        ).sum()
    )


def build_dataset() -> pd.DataFrame:
    anchor = read_daily(ANCHOR_PATH, ANCHOR)
    experts = {model: read_daily(path, model) for model, path in EXPERT_PATHS.items()}
    keys = common_keys([anchor, *experts.values()])

    anchor = keys.merge(anchor, on=["fold", "date"], how="left")
    expert_frames = {model: keys.merge(df, on=["fold", "date"], how="left") for model, df in experts.items()}
    all_models = pd.concat([anchor, *expert_frames.values()], ignore_index=True)

    anchor_cols = [
        "fold",
        "date",
        "profit",
        "oracle_profit",
        "charge_start",
        "discharge_start",
        "predicted_spread",
        "top1_top2_gap",
        "top5_spread_std",
        "top1_top5_mean_gap",
    ]
    anchor_ref = anchor[anchor_cols].rename(
        columns={
            "profit": "anchor_profit",
            "oracle_profit": "oracle_profit",
            "charge_start": "anchor_charge_start",
            "discharge_start": "anchor_discharge_start",
            "predicted_spread": "anchor_predicted_spread",
            "top1_top2_gap": "anchor_top1_top2_gap",
            "top5_spread_std": "anchor_top5_spread_std",
            "top1_top5_mean_gap": "anchor_top1_top5_mean_gap",
        }
    )

    rows: list[dict[str, object]] = []
    all_model_names = {ANCHOR, *EXPERT_PATHS}
    segmented_names = {ANCHOR, "seg6", "margin6"}
    non_lgb_names = {"cat180", "xgb180"}
    for (fold, date), group in all_models.groupby(["fold", "date"], sort=True):
        anchor_day = group[group["model"] == ANCHOR].iloc[0]
        rows.append(
            {
                "fold": fold,
                "date": date,
                "candidate_model": ANCHOR,
                "charge_start": int(anchor_day["charge_start"]),
                "discharge_start": int(anchor_day["discharge_start"]),
                "candidate_predicted_spread": float(anchor_day["predicted_spread"]),
                "true_profit": float(anchor_day["profit"]),
                "candidate_is_anchor": 1,
            }
        )
        for model in EXPERT_PATHS:
            candidate = group[group["model"] == model].iloc[0]
            support_exact_all = _exact_support(group, candidate, all_model_names)
            support_near2_all = _near_support(group, candidate, all_model_names, 2)
            support_near4_all = _near_support(group, candidate, all_model_names, 4)
            support_near2_segmented = _near_support(group, candidate, segmented_names, 2)
            support_near2_non_lgb = _near_support(group, candidate, non_lgb_names, 2)
            rows.append(
                {
                    "fold": fold,
                    "date": date,
                    "candidate_model": model,
                    "charge_start": int(candidate["charge_start"]),
                    "discharge_start": int(candidate["discharge_start"]),
                    "candidate_predicted_spread": float(candidate["predicted_spread"]),
                    "true_profit": float(candidate["profit"]),
                    "candidate_is_anchor": 0,
                    "support_exact_all": support_exact_all,
                    "support_near2_all": support_near2_all,
                    "support_near4_all": support_near4_all,
                    "support_near2_segmented": support_near2_segmented,
                    "support_near2_non_lgb": support_near2_non_lgb,
                }
            )

    dataset = pd.DataFrame(rows)
    dataset = dataset.merge(anchor_ref, on=["fold", "date"], how="left")
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["month"] = dataset["date"].dt.month
    dataset["model_priority"] = dataset["candidate_model"].map(MODEL_PRIORITY).fillna(-1).astype(int)
    dataset["changed_from_anchor"] = (
        (dataset["charge_start"].astype(int) != dataset["anchor_charge_start"].astype(int))
        | (
            dataset["discharge_start"].astype(int)
            != dataset["anchor_discharge_start"].astype(int)
        )
    ).astype(int)
    dataset["pair_distance_from_anchor"] = (
        dataset["charge_start"].astype(int).sub(dataset["anchor_charge_start"].astype(int)).abs()
        + dataset["discharge_start"]
        .astype(int)
        .sub(dataset["anchor_discharge_start"].astype(int))
        .abs()
    )
    dataset["lift_vs_anchor"] = dataset["true_profit"] - dataset["anchor_profit"]
    for col in [
        "support_exact_all",
        "support_near2_all",
        "support_near4_all",
        "support_near2_segmented",
        "support_near2_non_lgb",
    ]:
        dataset[col] = dataset[col].fillna(1).astype(int)
    return dataset.sort_values(["fold", "date", "candidate_is_anchor", "candidate_model"]).reset_index(
        drop=True
    )


def rule_id(rule: Rule) -> str:
    return (
        f"scope_{rule.scope}__gate_{rule.gate_feature}_q{rule.gate_quantile:.2f}"
        f"__{rule.support_col}_ge{rule.min_support}"
    )


def make_rule(
    train: pd.DataFrame,
    *,
    scope: str,
    gate_feature: str,
    gate_quantile: float,
    support_col: str,
    min_support: int,
) -> Rule:
    anchor_days = train[train["candidate_model"] == ANCHOR].drop_duplicates(["fold", "date"])
    threshold = float(pd.to_numeric(anchor_days[gate_feature], errors="coerce").quantile(gate_quantile))
    return Rule(
        scope=scope,
        gate_feature=gate_feature,
        gate_quantile=gate_quantile,
        gate_threshold=threshold,
        support_col=support_col,
        min_support=min_support,
    )


def anchor_selection(df: pd.DataFrame, method: str = "anchor_top1") -> pd.DataFrame:
    selected = df[df["candidate_model"] == ANCHOR].copy()
    selected["method"] = method
    selected["rule_id"] = method
    selected["changed_from_anchor"] = 0
    selected["lift_vs_anchor"] = 0.0
    return selected.sort_values(["fold", "date"]).reset_index(drop=True)


def oracle_selection(df: pd.DataFrame) -> pd.DataFrame:
    selected = (
        df.sort_values(
            ["fold", "date", "true_profit", "candidate_is_anchor", "model_priority"],
            ascending=[True, True, False, False, True],
        )
        .groupby(["fold", "date"], as_index=False)
        .head(1)
        .copy()
    )
    selected["method"] = "oracle_family_top1"
    selected["rule_id"] = "oracle_family_top1"
    selected["changed_from_anchor"] = (selected["candidate_model"] != ANCHOR).astype(int)
    selected["lift_vs_anchor"] = selected["true_profit"] - selected["anchor_profit"]
    return selected.sort_values(["fold", "date"]).reset_index(drop=True)


def apply_rule(df: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    allowed_models = set(SCOPES[rule.scope])
    selected_rows: list[pd.Series] = []
    for _, group in df.groupby(["fold", "date"], sort=True):
        anchor_row = group[group["candidate_model"] == ANCHOR].iloc[0]
        if float(anchor_row[rule.gate_feature]) > rule.gate_threshold:
            selected_rows.append(anchor_row)
            continue
        eligible = group[
            group["candidate_model"].isin(allowed_models)
            & (group["changed_from_anchor"] == 1)
            & (group[rule.support_col] >= rule.min_support)
        ].copy()
        if eligible.empty:
            selected_rows.append(anchor_row)
            continue
        selected = eligible.sort_values(
            [
                rule.support_col,
                "support_near4_all",
                "support_exact_all",
                "pair_distance_from_anchor",
                "model_priority",
            ],
            ascending=[False, False, False, True, True],
        ).iloc[0]
        selected_rows.append(selected)

    selected = pd.DataFrame(selected_rows).sort_values(["fold", "date"]).reset_index(drop=True)
    selected["method"] = "model_family_consensus_fallback"
    selected["rule_id"] = rule_id(rule)
    selected["lift_vs_anchor"] = selected["true_profit"] - selected["anchor_profit"]
    selected["changed_from_anchor"] = (selected["candidate_model"] != ANCHOR).astype(int)
    return selected


def selection_metrics(selected: pd.DataFrame) -> dict[str, float]:
    profit = selected["true_profit"].astype(float)
    anchor_profit = selected["anchor_profit"].astype(float)
    lift = selected["lift_vs_anchor"].astype(float)
    return {
        "days": float(len(selected)),
        "mean_profit": float(profit.mean()),
        "anchor_mean_profit": float(anchor_profit.mean()),
        "mean_lift_vs_anchor": float(lift.mean()),
        "total_lift_vs_anchor": float(lift.sum()),
        "worst_profit": float(profit.min()),
        "anchor_worst_profit": float(anchor_profit.min()),
        "p10_profit": float(profit.quantile(0.1)),
        "loss_days": float((profit < -1e-9).sum()),
        "anchor_loss_days": float((anchor_profit < -1e-9).sum()),
        "changed_days": float(selected["changed_from_anchor"].sum()),
        "changed_rate": float(selected["changed_from_anchor"].mean()),
        "positive_lift_days": float((lift > 1e-9).sum()),
        "negative_lift_days": float((lift < -1e-9).sum()),
    }


def evaluate_rule(train: pd.DataFrame, rule: Rule) -> dict[str, object]:
    selected = apply_rule(train, rule)
    metrics = selection_metrics(selected)
    metrics.update(
        {
            "rule_id": rule_id(rule),
            "scope": rule.scope,
            "gate_feature": rule.gate_feature,
            "gate_quantile": rule.gate_quantile,
            "gate_threshold": rule.gate_threshold,
            "support_col": rule.support_col,
            "min_support": rule.min_support,
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
    if row["worst_profit"] < row["anchor_worst_profit"] - MAX_WORST_PROFIT_DROP:
        return False
    return True


def choose_rule(train: pd.DataFrame) -> tuple[Rule | None, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for scope in SCOPES:
        for gate_feature in GATE_FEATURES:
            for gate_quantile in GATE_QUANTILES:
                for support_col, min_support in SUPPORT_RULES:
                    rule = make_rule(
                        train,
                        scope=scope,
                        gate_feature=gate_feature,
                        gate_quantile=gate_quantile,
                        support_col=support_col,
                        min_support=min_support,
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
            "changed_rate",
            "min_support",
        ],
        ascending=[False, True, True, True, False],
    ).iloc[0]
    return (
        Rule(
            scope=str(best["scope"]),
            gate_feature=str(best["gate_feature"]),
            gate_quantile=float(best["gate_quantile"]),
            gate_threshold=float(best["gate_threshold"]),
            support_col=str(best["support_col"]),
            min_support=int(best["min_support"]),
        ),
        search,
    )


def crossfold_evaluate(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_frames: list[pd.DataFrame] = []
    rule_rows: list[dict[str, object]] = []
    search_frames: list[pd.DataFrame] = []

    for heldout_fold in STANDARD_FOLDS:
        train = dataset[dataset["fold"] != heldout_fold].copy()
        valid = dataset[dataset["fold"] == heldout_fold].copy()
        selected_frames.append(anchor_selection(valid))
        selected_frames.append(oracle_selection(valid))

        rule, search = choose_rule(train)
        search["heldout_fold"] = heldout_fold
        search_frames.append(search)
        if rule is None:
            selected = anchor_selection(valid, method="model_family_consensus_fallback")
            selected["rule_id"] = "fallback_anchor_top1"
            rule_rows.append(
                {
                    "heldout_fold": heldout_fold,
                    "rule_id": "fallback_anchor_top1",
                    "selected_from_train": False,
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                }
            )
        else:
            selected = apply_rule(valid, rule)
            train_selected = apply_rule(train, rule)
            train_metrics = selection_metrics(train_selected)
            rule_rows.append(
                {
                    "heldout_fold": heldout_fold,
                    "rule_id": rule_id(rule),
                    "selected_from_train": True,
                    "scope": rule.scope,
                    "gate_feature": rule.gate_feature,
                    "gate_quantile": rule.gate_quantile,
                    "gate_threshold": rule.gate_threshold,
                    "support_col": rule.support_col,
                    "min_support": rule.min_support,
                    "train_mean_lift_vs_anchor": train_metrics["mean_lift_vs_anchor"],
                    "train_changed_days": train_metrics["changed_days"],
                    "train_changed_rate": train_metrics["changed_rate"],
                    "train_positive_lift_days": train_metrics["positive_lift_days"],
                    "train_negative_lift_days": train_metrics["negative_lift_days"],
                    "train_loss_days": train_metrics["loss_days"],
                    "train_anchor_loss_days": train_metrics["anchor_loss_days"],
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                }
            )
        selected_frames.append(selected)

    return (
        pd.concat(selected_frames, ignore_index=True),
        pd.DataFrame(rule_rows),
        pd.concat(search_frames, ignore_index=True),
    )


def summarize(selected: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in selected.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = selection_metrics(group)
        row = {col: key for col, key in zip(group_cols, keys, strict=True)}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def add_anchor_deltas(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        anchor = summary[summary["method"] == "anchor_top1"].iloc[0]
        out = summary.copy()
        out["anchor_group_mean_profit"] = float(anchor["mean_profit"])
        out["anchor_group_loss_days"] = float(anchor["loss_days"])
        out["profit_delta_vs_anchor"] = out["mean_profit"] - out["anchor_group_mean_profit"]
        out["loss_day_delta_vs_anchor"] = out["loss_days"] - out["anchor_group_loss_days"]
        return out.sort_values("mean_profit", ascending=False)

    anchors = summary[summary["method"] == "anchor_top1"][
        [*group_cols, "mean_profit", "loss_days"]
    ].rename(
        columns={
            "mean_profit": "anchor_group_mean_profit",
            "loss_days": "anchor_group_loss_days",
        }
    )
    out = summary.merge(anchors, on=group_cols, how="left")
    out["profit_delta_vs_anchor"] = out["mean_profit"] - out["anchor_group_mean_profit"]
    out["loss_day_delta_vs_anchor"] = out["loss_days"] - out["anchor_group_loss_days"]
    return out.sort_values([*group_cols, "mean_profit"], ascending=[True] * len(group_cols) + [False])


def expert_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in dataset[dataset["candidate_model"] != ANCHOR].groupby("candidate_model"):
        lift = group["lift_vs_anchor"].astype(float)
        rows.append(
            {
                "candidate_model": model,
                "days": len(group),
                "mean_profit": float(group["true_profit"].mean()),
                "anchor_mean_profit": float(group["anchor_profit"].mean()),
                "mean_lift_vs_anchor": float(lift.mean()),
                "positive_lift_days": int((lift > 1e-9).sum()),
                "negative_lift_days": int((lift < -1e-9).sum()),
                "same_pair_rate": float((group["changed_from_anchor"] == 0).mean()),
                "support_near2_ge3_days": int((group["support_near2_all"] >= 3).sum()),
                "support_near2_ge4_days": int((group["support_near2_all"] >= 4).sum()),
                "support_near2_ge3_mean_lift": float(
                    group.loc[group["support_near2_all"] >= 3, "lift_vs_anchor"].mean()
                ),
                "support_near2_ge4_mean_lift": float(
                    group.loc[group["support_near2_all"] >= 4, "lift_vs_anchor"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_lift_vs_anchor", ascending=False)


def changed_day_audit(selected: pd.DataFrame) -> pd.DataFrame:
    changed = selected[
        (selected["method"] == "model_family_consensus_fallback")
        & (selected["changed_from_anchor"] == 1)
    ].copy()
    cols = [
        "fold",
        "date",
        "candidate_model",
        "true_profit",
        "anchor_profit",
        "lift_vs_anchor",
        "charge_start",
        "discharge_start",
        "anchor_charge_start",
        "anchor_discharge_start",
        "support_near2_all",
        "support_near4_all",
        "support_near2_segmented",
        "pair_distance_from_anchor",
        "rule_id",
    ]
    return changed[cols].sort_values("lift_vs_anchor")


def write_report(
    *,
    dataset: pd.DataFrame,
    selected: pd.DataFrame,
    rules: pd.DataFrame,
    search: pd.DataFrame,
    overall: pd.DataFrame,
    by_fold: pd.DataFrame,
    by_month: pd.DataFrame,
    experts: pd.DataFrame,
) -> None:
    fallback = overall[overall["method"] == "model_family_consensus_fallback"].iloc[0]
    oracle = overall[overall["method"] == "oracle_family_top1"].iloc[0]
    passing = search[search["passes_constraints"]].copy()
    top_passing = passing.sort_values(
        ["mean_lift_vs_anchor", "negative_lift_days", "changed_days"],
        ascending=[False, True, True],
    ).head(20)
    changed = changed_day_audit(selected)
    coverage = (
        dataset[dataset["candidate_model"] == ANCHOR]
        .groupby("fold", as_index=False)
        .agg(days=("date", "nunique"), start=("date", "min"), end=("date", "max"))
    )

    lines = [
        "# Model-Family Consensus Fallback",
        "",
        "## Scope",
        "",
        "- Anchor: current online champion `ens_champion_segmented6_prior_5fold`.",
        "- Expert families: CatBoost 180d, XGBoost 180d, segmented_6, segmented_6 margin-core.",
        "- This is a diagnostic-only experiment: no retraining and no submit file generation.",
        (
            "- Coverage is limited to the common 09-12 validation folds because Cat/XGB old "
            "daily reports do not include `valid_2025_jan_feb`."
        ),
        "",
        "## Plan Executed",
        "",
        "- Build one candidate per day from each expert's top1 legal charge/discharge pair.",
        "- Keep champion top1 by default.",
        (
            "- Search conservative fallback rules using only non-held-out folds: low champion "
            "confidence gate plus near-pair support among model families."
        ),
        "- Apply the selected rule to the held-out fold only.",
        "- Reject any rule that changes more than 10% of training days, increases loss days, "
        "or has more negative than positive changed days.",
        "",
        "## Data Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Expert Top1 Standalone Comparison",
        "",
        markdown_table(experts),
        "",
        "## Overall",
        "",
        markdown_table(overall),
        "",
        "## Fold Summary",
        "",
        markdown_table(by_fold, limit=40),
        "",
        "## Month Summary",
        "",
        markdown_table(by_month, limit=40),
        "",
        "## Selected Rules By Held-Out Fold",
        "",
        markdown_table(rules),
        "",
        "## Changed-Day Audit",
        "",
        markdown_table(changed, limit=40),
        "",
        "## Top Passing Train Rules",
        "",
        markdown_table(top_passing, limit=20),
        "",
        "## Decision",
        "",
    ]

    delta = float(fallback["profit_delta_vs_anchor"])
    loss_delta = float(fallback["loss_day_delta_vs_anchor"])
    oracle_gap = float(oracle["profit_delta_vs_anchor"])
    if delta > 100 and loss_delta <= 0:
        lines += [
            "- The fallback clears the local promotion threshold on the common 09-12 folds.",
            (
                "- Still do not submit directly: first regenerate Jan-Feb-compatible expert "
                "daily reports or run a 5-fold version of this diagnostic."
            ),
        ]
    elif delta > 0 and loss_delta <= 0:
        lines += [
            "- The fallback is directionally positive but below the +100/day promotion threshold.",
            "- Keep it as evidence that model-family disagreement has weak signal; no submit.",
        ]
    else:
        lines += [
            "- The fold-safe fallback does not improve the anchor enough, or worsens risk.",
            "- Reject this standalone rule family; no submit.",
        ]
    lines += [
        f"- Oracle among the five top1 model-family choices is still `{oracle_gap:.3f}`/day "
        "above anchor, so the remaining problem is learning when to trust which family, "
        "not whether an alternative family sometimes has the better pair.",
        "",
        "## Artifacts",
        "",
        f"- `{SELECTED_PATH}`",
        f"- `{RULES_PATH}`",
        f"- `{SEARCH_PATH}`",
        f"- `{OVERALL_PATH}`",
        f"- `{BY_FOLD_PATH}`",
        f"- `{BY_MONTH_PATH}`",
        f"- `{EXPERT_SUMMARY_PATH}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    dataset = build_dataset()
    selected, rules, search = crossfold_evaluate(dataset)

    overall = add_anchor_deltas(summarize(selected, ["method"]), [])
    by_fold = add_anchor_deltas(summarize(selected, ["fold", "method"]), ["fold"])
    by_month = add_anchor_deltas(summarize(selected, ["month", "method"]), ["month"])
    experts = expert_summary(dataset)

    selected.to_csv(SELECTED_PATH, index=False)
    rules.to_csv(RULES_PATH, index=False)
    search.to_csv(SEARCH_PATH, index=False)
    overall.to_csv(OVERALL_PATH, index=False)
    by_fold.to_csv(BY_FOLD_PATH, index=False)
    by_month.to_csv(BY_MONTH_PATH, index=False)
    experts.to_csv(EXPERT_SUMMARY_PATH, index=False)
    write_report(
        dataset=dataset,
        selected=selected,
        rules=rules,
        search=search,
        overall=overall,
        by_fold=by_fold,
        by_month=by_month,
        experts=experts,
    )

    print(overall.to_string(index=False))
    print(f"markdown_path={OUT_MD}")


if __name__ == "__main__":
    main()
