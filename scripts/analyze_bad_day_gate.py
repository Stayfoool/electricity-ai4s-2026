from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from electricity.features.bid_space import markdown_table

REPORTS_DIR = Path("reports")
CANDIDATES_PATH = REPORTS_DIR / "weather_gated_rerank_candidates.csv"
DAILY_PATH = REPORTS_DIR / "backtest_ens_champion_segmented6_prior_daily.csv"

TOP_K_OPTIONS = [2, 3, 5, 10]
MASK_MAX_RATE = 0.25
MASK_MIN_DAYS = 2

LABELS = ["loss_day", "high_regret_top25", "high_oracle10_lift_top25", "bottom_profit25"]


@dataclass(frozen=True)
class GateRule:
    action: str
    feature: str
    op: str
    threshold: float
    top_k: int | None
    train_mean_profit: float
    train_loss_days: int
    train_mask_days: int
    train_lift_vs_top1: float


def _candidate_at_rank(candidates: pd.DataFrame, rank: int) -> pd.DataFrame:
    return candidates[candidates["candidate_rank"] == rank].copy()


def _candidate_best_by(candidates: pd.DataFrame, *, top_k: int, score_col: str) -> pd.DataFrame:
    sub = candidates[candidates["candidate_rank"] <= top_k].copy()
    idx = sub.groupby(["fold", "date"])[score_col].idxmax()
    return sub.loc[idx].copy()


def build_dataset() -> pd.DataFrame:
    candidates = pd.read_csv(CANDIDATES_PATH, parse_dates=["date"])
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    top1 = _candidate_at_rank(candidates, 1).rename(
        columns={
            "true_profit": "top1_profit",
            "predicted_spread": "top1_predicted_spread",
            "prior_decision_score": "top1_prior_score",
            "weather_bid_spread": "top1_weather_bid_spread",
            "orig_bid_spread": "top1_orig_bid_spread",
            "charge_start": "top1_charge_start",
            "discharge_start": "top1_discharge_start",
        }
    )
    keep = [
        "date",
        "fold",
        "top1_profit",
        "oracle_profit",
        "regret_vs_oracle",
        "top1_predicted_spread",
        "top1_prior_score",
        "top1_weather_bid_spread",
        "top1_orig_bid_spread",
        "top1_charge_start",
        "top1_discharge_start",
        "wind_charge_mean",
        "wind_discharge_mean",
        "ghi_charge_mean",
        "ghi_discharge_mean",
    ]
    df = top1[keep].copy().rename(columns={"regret_vs_oracle": "top1_regret"})

    grouped = candidates.groupby(["fold", "date"], sort=True)
    agg = grouped.agg(
        top10_pred_spread_mean=("predicted_spread", "mean"),
        top10_pred_spread_std=("predicted_spread", "std"),
        top10_pred_spread_min=("predicted_spread", "min"),
        top10_prior_score_mean=("prior_decision_score", "mean"),
        top10_prior_score_std=("prior_decision_score", "std"),
        top10_prior_score_min=("prior_decision_score", "min"),
        top10_weather_spread_mean=("weather_bid_spread", "mean"),
        top10_weather_spread_std=("weather_bid_spread", "std"),
        top10_weather_spread_max=("weather_bid_spread", "max"),
        top10_weather_delta_max=("weather_bid_spread_delta_from_top1", "max"),
        top10_weather_delta_min=("weather_bid_spread_delta_from_top1", "min"),
        top10_charge_std=("charge_start", "std"),
        top10_discharge_std=("discharge_start", "std"),
        top10_charge_min=("charge_start", "min"),
        top10_charge_max=("charge_start", "max"),
        top10_discharge_min=("discharge_start", "min"),
        top10_discharge_max=("discharge_start", "max"),
    ).reset_index()
    df = df.merge(agg, on=["fold", "date"], how="left")

    rank2 = _candidate_at_rank(candidates, 2)[
        ["date", "predicted_spread", "prior_decision_score", "true_profit"]
    ].rename(
        columns={
            "predicted_spread": "rank2_predicted_spread",
            "prior_decision_score": "rank2_prior_score",
            "true_profit": "rank2_profit",
        }
    )
    rank10 = _candidate_at_rank(candidates, 10)[
        ["date", "predicted_spread", "prior_decision_score", "true_profit"]
    ].rename(
        columns={
            "predicted_spread": "rank10_predicted_spread",
            "prior_decision_score": "rank10_prior_score",
            "true_profit": "rank10_profit",
        }
    )
    df = df.merge(rank2, on="date", how="left").merge(rank10, on="date", how="left")
    df["top1_rank2_pred_gap"] = df["top1_predicted_spread"] - df["rank2_predicted_spread"]
    df["top1_rank2_prior_gap"] = df["top1_prior_score"] - df["rank2_prior_score"]
    df["top1_rank10_prior_gap"] = df["top1_prior_score"] - df["rank10_prior_score"]
    df["top10_pred_spread_range"] = df["top1_predicted_spread"] - df["top10_pred_spread_min"]
    df["top10_prior_score_range"] = df["top1_prior_score"] - df["top10_prior_score_min"]
    df["top10_charge_range"] = df["top10_charge_max"] - df["top10_charge_min"]
    df["top10_discharge_range"] = df["top10_discharge_max"] - df["top10_discharge_min"]
    df["top1_weather_minus_pred"] = df["top1_weather_bid_spread"] - df["top1_predicted_spread"]
    df["top1_orig_bid_minus_pred"] = df["top1_orig_bid_spread"] - df["top1_predicted_spread"]
    df["top1_wind_delta"] = df["wind_discharge_mean"] - df["wind_charge_mean"]
    df["top1_ghi_delta"] = df["ghi_discharge_mean"] - df["ghi_charge_mean"]
    df["top1_gap_slots"] = df["top1_discharge_start"] - df["top1_charge_start"]
    df["top1_charge_hour"] = df["top1_charge_start"] / 4.0
    df["top1_discharge_hour"] = df["top1_discharge_start"] / 4.0
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    oracle_top10 = candidates.loc[grouped["true_profit"].idxmax()].copy()
    oracle_top10 = oracle_top10[["date", "candidate_rank", "true_profit"]].rename(
        columns={"candidate_rank": "oracle10_rank", "true_profit": "oracle10_profit"}
    )
    df = df.merge(oracle_top10, on="date", how="left")
    df["oracle10_lift"] = df["oracle10_profit"] - df["top1_profit"]

    # Use official daily fields for consistency and extra known diagnostics.
    daily_keep = [
        "date",
        "top1_top2_gap",
        "top1_top5_mean_gap",
        "top5_spread_std",
        "profit_ratio_day",
        "charge_start_gap",
        "discharge_start_gap",
    ]
    df = df.merge(daily[daily_keep], on="date", how="left")

    df["loss_day"] = df["top1_profit"] < 0
    df["high_regret_top25"] = df["top1_regret"] >= df["top1_regret"].quantile(0.75)
    df["high_oracle10_lift_top25"] = df["oracle10_lift"] >= df["oracle10_lift"].quantile(0.75)
    df["bottom_profit25"] = df["top1_profit"] <= df["top1_profit"].quantile(0.25)
    return df.sort_values("date").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "fold",
        "top1_profit",
        "oracle_profit",
        "top1_regret",
        "rank2_profit",
        "rank10_profit",
        "oracle10_profit",
        "oracle10_lift",
        "oracle10_rank",
        "profit_ratio_day",
        "charge_start_gap",
        "discharge_start_gap",
        *LABELS,
    }
    out = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def screen_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        y = df[label].astype(int)
        for feature in features:
            x = df[feature].astype(float)
            if x.nunique(dropna=True) < 3 or y.nunique() < 2:
                continue
            valid = x.notna()
            try:
                auc = float(roc_auc_score(y[valid], x[valid]))
            except ValueError:
                continue
            corr = float(x.corr(y, method="spearman"))
            bad = df.loc[df[label], feature].mean()
            good = df.loc[~df[label], feature].mean()
            rows.append(
                {
                    "label": label,
                    "feature": feature,
                    "auc": auc,
                    "auc_distance": abs(auc - 0.5),
                    "spearman": corr,
                    "mean_on_label_true": float(bad),
                    "mean_on_label_false": float(good),
                    "direction": "higher_is_risk" if auc > 0.5 else "lower_is_risk",
                }
            )
    return pd.DataFrame(rows).sort_values(["label", "auc_distance"], ascending=[True, False])


def _mask(df: pd.DataFrame, feature: str, op: str, threshold: float) -> pd.Series:
    if op == "<=":
        return df[feature] <= threshold
    if op == ">":
        return df[feature] > threshold
    raise ValueError(f"unsupported op={op}")


def _candidate_thresholds(values: pd.Series) -> list[float]:
    clean = values.dropna()
    if clean.nunique() < 4:
        return []
    qs = [0.05, 0.1, 0.2, 0.33, 0.5, 0.67, 0.8, 0.9]
    return sorted(set(float(x) for x in clean.quantile(qs).to_list()))


def _replacement_profit(df: pd.DataFrame, action: str, top_k: int | None) -> pd.Series:
    if action == "no_trade":
        return pd.Series(0.0, index=df.index)
    if action == "rank2":
        return df["rank2_profit"].astype(float)
    if action == "rank10":
        return df["rank10_profit"].astype(float)
    raise ValueError(f"unsupported action={action}")


def evaluate_rule(df: pd.DataFrame, rule: GateRule) -> pd.Series:
    mask = _mask(df, rule.feature, rule.op, rule.threshold)
    replacement = _replacement_profit(df, rule.action, rule.top_k)
    return pd.Series(np.where(mask, replacement, df["top1_profit"]), index=df.index)


def _rule_mask_is_usable(mask: pd.Series, train_days: int) -> bool:
    count = int(mask.sum())
    return MASK_MIN_DAYS <= count <= max(MASK_MIN_DAYS, int(train_days * MASK_MAX_RATE))


def learn_rule(train: pd.DataFrame, features: list[str]) -> GateRule:
    base_mean = float(train["top1_profit"].mean())
    best = GateRule("none", features[0], ">", float("inf"), None, base_mean, 999, 0, 0.0)
    actions = ["no_trade", "rank2", "rank10"]
    for feature in features:
        for threshold in _candidate_thresholds(train[feature]):
            for op in ["<=", ">"]:
                mask = _mask(train, feature, op, threshold)
                if not _rule_mask_is_usable(mask, len(train)):
                    continue
                for action in actions:
                    replacement = _replacement_profit(train, action, None)
                    profit = pd.Series(
                        np.where(mask, replacement, train["top1_profit"]), index=train.index
                    )
                    mean_profit = float(profit.mean())
                    loss_days = int((profit < 0).sum())
                    candidate = GateRule(
                        action=action,
                        feature=feature,
                        op=op,
                        threshold=threshold,
                        top_k=None,
                        train_mean_profit=mean_profit,
                        train_loss_days=loss_days,
                        train_mask_days=int(mask.sum()),
                        train_lift_vs_top1=mean_profit - base_mean,
                    )
                    if (
                        candidate.train_mean_profit,
                        -candidate.train_loss_days,
                        -candidate.train_mask_days,
                    ) > (best.train_mean_profit, -best.train_loss_days, -best.train_mask_days):
                        best = candidate
    return best


def summarize_profit(profit: pd.Series, prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_mean_profit": float(profit.mean()),
        f"{prefix}_worst_profit": float(profit.min()),
        f"{prefix}_loss_days": int((profit < 0).sum()),
    }


def crossfold_gate(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rule_rows = []
    daily_rows = []
    for fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != fold].copy()
        valid = df[df["fold"] == fold].copy()
        rule = learn_rule(train, features)
        mask = _mask(valid, rule.feature, rule.op, rule.threshold)
        gated = evaluate_rule(valid, rule)
        base = valid["top1_profit"].astype(float)
        row = {
            "heldout_fold": fold,
            "action": rule.action,
            "feature": rule.feature,
            "op": rule.op,
            "threshold": rule.threshold,
            "train_mean_profit": rule.train_mean_profit,
            "train_lift_vs_top1": rule.train_lift_vs_top1,
            "train_loss_days": rule.train_loss_days,
            "train_mask_days": rule.train_mask_days,
            "valid_mask_days": int(mask.sum()),
            "valid_filtered_loss_days": int(((base < 0) & mask).sum()),
            "valid_filtered_profit_days": int(((base > 0) & mask).sum()),
        }
        row.update(summarize_profit(base, "top1"))
        row.update(summarize_profit(gated, "gate"))
        row["valid_lift_vs_top1"] = float(gated.mean() - base.mean())
        rule_rows.append(row)
        for idx, item in valid.iterrows():
            daily_rows.append(
                {
                    "date": item["date"].date().isoformat(),
                    "fold": fold,
                    "action": rule.action,
                    "feature": rule.feature,
                    "op": rule.op,
                    "threshold": rule.threshold,
                    "feature_value": float(item[rule.feature]),
                    "triggered": bool(mask.loc[idx]),
                    "top1_profit": float(base.loc[idx]),
                    "gate_profit": float(gated.loc[idx]),
                    "lift_vs_top1": float(gated.loc[idx] - base.loc[idx]),
                    "top1_regret": float(item["top1_regret"]),
                    "oracle10_lift": float(item["oracle10_lift"]),
                    "loss_day": bool(item["loss_day"]),
                    "high_regret_top25": bool(item["high_regret_top25"]),
                }
            )
    return pd.DataFrame(rule_rows), pd.DataFrame(daily_rows)


def overall_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, col in [("top1", "top1_profit"), ("bad_day_gate_cv", "gate_profit")]:
        profit = daily[col].astype(float)
        rows.append(
            {
                "method": method,
                "days": len(daily),
                "mean_profit": float(profit.mean()),
                "worst_profit": float(profit.min()),
                "loss_days": int((profit < 0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    base = float(out.loc[out["method"] == "top1", "mean_profit"].iloc[0])
    out["lift_vs_top1"] = out["mean_profit"] - base
    return out


def write_report(
    dataset: pd.DataFrame,
    feature_screen: pd.DataFrame,
    rules: pd.DataFrame,
    daily: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    top_features = feature_screen.groupby("label", group_keys=False).head(8)
    triggered = daily[daily["triggered"]].copy()
    lines = [
        "# Bad-Day Gate Diagnostics",
        "",
        "## Scope",
        "",
        "This report asks whether current champion-prior bad days can be identified "
        "before seeing actual prices. It uses only prediction-stage features for "
        "rules, then evaluates against validation labels.",
        "",
        "Labels inspected:",
        "",
        "- `loss_day`: champion top1 profit < 0.",
        "- `high_regret_top25`: top quartile of regret.",
        "- `high_oracle10_lift_top25`: top quartile of available top-10 improvement.",
        "- `bottom_profit25`: bottom quartile of top1 profit.",
        "",
        "## Baseline Bad-Day Counts",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "days": len(dataset),
                        "loss_days": int(dataset["loss_day"].sum()),
                        "high_regret_days": int(dataset["high_regret_top25"].sum()),
                        "bottom_profit_days": int(dataset["bottom_profit25"].sum()),
                        "mean_top1_profit": float(dataset["top1_profit"].mean()),
                        "mean_oracle10_lift": float(dataset["oracle10_lift"].mean()),
                    }
                ]
            ),
            floatfmt=".4f",
        ),
        "",
        "## Feature Screen Top Signals",
        "",
        markdown_table(top_features, floatfmt=".4f"),
        "",
        "## Cross-Fold Gate Result",
        "",
        markdown_table(overall, floatfmt=".4f"),
        "",
        "## Learned Rules By Held-Out Fold",
        "",
        markdown_table(rules, floatfmt=".4f"),
        "",
        "## Triggered Days",
        "",
        markdown_table(
            triggered[
                [
                    "date",
                    "fold",
                    "action",
                    "feature",
                    "op",
                    "threshold",
                    "feature_value",
                    "top1_profit",
                    "gate_profit",
                    "lift_vs_top1",
                    "top1_regret",
                    "oracle10_lift",
                ]
            ].sort_values("lift_vs_top1"),
            floatfmt=".4f",
        ),
        "",
        "## Decision",
        "",
    ]
    gate_lift = float(overall.loc[overall["method"] == "bad_day_gate_cv", "lift_vs_top1"].iloc[0])
    gate_loss = int(overall.loc[overall["method"] == "bad_day_gate_cv", "loss_days"].iloc[0])
    base_loss = int(overall.loc[overall["method"] == "top1", "loss_days"].iloc[0])
    if gate_lift > 50 and gate_loss <= base_loss:
        lines += [
            "- The gate has enough held-out lift to become a candidate strategy.",
            "- Next step: harden the rule family and validate on 5-fold reports.",
        ]
    else:
        lines += [
            "- The current simple gate is not strong enough to replace champion dispatch.",
            "- Keep the feature screen as evidence for the next targeted bad-day model.",
        ]
    lines += [
        "",
        "## Artifacts",
        "",
        "- `reports/bad_day_gate_dataset.csv`",
        "- `reports/bad_day_gate_feature_screen.csv`",
        "- `reports/bad_day_gate_rules.csv`",
        "- `reports/bad_day_gate_daily.csv`",
        "- `reports/bad_day_gate_summary.csv`",
    ]
    (REPORTS_DIR / "bad_day_gate_diagnostics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    dataset = build_dataset()
    features = feature_columns(dataset)
    feature_screen = screen_features(dataset, features)
    # Restrict rule learning to the strongest simple signals to reduce overfitting.
    selected_features = (
        feature_screen.groupby("feature")["auc_distance"]
        .max()
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )
    rules, daily = crossfold_gate(dataset, selected_features)
    overall = overall_from_daily(daily)

    dataset.to_csv(REPORTS_DIR / "bad_day_gate_dataset.csv", index=False)
    feature_screen.to_csv(REPORTS_DIR / "bad_day_gate_feature_screen.csv", index=False)
    rules.to_csv(REPORTS_DIR / "bad_day_gate_rules.csv", index=False)
    daily.to_csv(REPORTS_DIR / "bad_day_gate_daily.csv", index=False)
    overall.to_csv(REPORTS_DIR / "bad_day_gate_summary.csv", index=False)
    write_report(dataset, feature_screen, rules, daily, overall)

    print(overall.to_string(index=False))
    print(rules.to_string(index=False))
    print(f"dataset_path={REPORTS_DIR / 'bad_day_gate_dataset.csv'}")
    print(f"feature_screen_path={REPORTS_DIR / 'bad_day_gate_feature_screen.csv'}")
    print(f"markdown_path={REPORTS_DIR / 'bad_day_gate_diagnostics.md'}")


if __name__ == "__main__":
    main()
