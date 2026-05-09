from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity.data import load_train_frame

REPORTS_DIR = Path("reports")
CONFIG_PATH = Path("configs/base.yaml")
CHAMPION = "ens_baseline_baseline_last_180d"
MODEL_CANDIDATES = [
    CHAMPION,
    "lgb_baseline",
    "lgb_baseline_last_180d",
    "lgb_business",
    "lgb_business_last_180d",
    "ens_business_business_last_180d",
    "cat_baseline_last_180d",
    "ens_base_base180_cat180",
]
FEATURE_COLUMNS_FOR_GATES = [
    "load_mean",
    "load_std",
    "renewable_ratio_mean",
    "renewable_ratio_std",
    "renewable_ratio_max",
    "net_load_mean",
    "net_load_std",
    "net_load_range",
    "wind_ratio_mean",
    "solar_ratio_mean",
    "solar_ratio_max",
    "tie_line_ratio_mean",
    "non_market_ratio_mean",
    "champion_predicted_spread",
    "lgb180_predicted_spread",
    "champion_lgb180_charge_start_diff",
    "champion_lgb180_discharge_start_diff",
    "champion_cat180_charge_start_diff",
    "champion_cat180_discharge_start_diff",
]


@dataclass(frozen=True)
class GateRule:
    feature: str
    threshold: float
    left_model: str
    right_model: str
    train_mean_profit: float
    train_lift_vs_champion: float


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_daily_model(model: str) -> pd.DataFrame:
    path = REPORTS_DIR / f"backtest_{model}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["date"])
    keep_cols = [
        "date",
        "fold",
        "profit",
        "oracle_profit",
        "regret",
        "charge_start",
        "discharge_start",
        "predicted_spread",
    ]
    df = df[keep_cols].copy()
    return df.rename(
        columns={
            "profit": f"{model}__profit",
            "regret": f"{model}__regret",
            "charge_start": f"{model}__charge_start",
            "discharge_start": f"{model}__discharge_start",
            "predicted_spread": f"{model}__predicted_spread",
        }
    )


def build_daily_feature_frame(cfg: dict) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    raw = load_train_frame(cfg)
    df = raw.copy()
    date = df[time_col].dt.normalize()
    eps = 1e-6
    load = df["系统负荷预测值"].clip(lower=eps)
    renewable = df["风光总加预测值"]
    wind = df["风电预测值"]
    solar = df["光伏预测值"]
    hydro = df["水电预测值"]

    df["date"] = date
    df["slot"] = df[time_col].dt.hour * 4 + df[time_col].dt.minute // 15
    df["net_load"] = df["系统负荷预测值"] - renewable - hydro
    df["renewable_ratio"] = renewable / load
    df["wind_ratio"] = wind / load
    df["solar_ratio"] = solar / load
    df["hydro_ratio"] = hydro / load
    df["tie_line_ratio"] = df["联络线预测值"] / load
    df["non_market_ratio"] = df["非市场化机组预测值"] / load

    rows: list[dict[str, object]] = []
    for day, group in df.groupby("date", sort=True):
        if len(group) != 96:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            continue

        row: dict[str, object] = {
            "date": day,
            "month": int(day.month),
            "dayofweek": int(day.dayofweek),
            "is_weekend": int(day.dayofweek >= 5),
            "load_mean": float(group["系统负荷预测值"].mean()),
            "load_std": float(group["系统负荷预测值"].std()),
            "load_range": float(
                group["系统负荷预测值"].max() - group["系统负荷预测值"].min()
            ),
            "renewable_ratio_mean": float(group["renewable_ratio"].mean()),
            "renewable_ratio_std": float(group["renewable_ratio"].std()),
            "renewable_ratio_max": float(group["renewable_ratio"].max()),
            "net_load_mean": float(group["net_load"].mean()),
            "net_load_std": float(group["net_load"].std()),
            "net_load_range": float(group["net_load"].max() - group["net_load"].min()),
            "wind_ratio_mean": float(group["wind_ratio"].mean()),
            "wind_ratio_std": float(group["wind_ratio"].std()),
            "solar_ratio_mean": float(group["solar_ratio"].mean()),
            "solar_ratio_max": float(group["solar_ratio"].max()),
            "hydro_ratio_mean": float(group["hydro_ratio"].mean()),
            "tie_line_ratio_mean": float(group["tie_line_ratio"].mean()),
            "non_market_ratio_mean": float(group["non_market_ratio"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_gating_dataset(cfg: dict) -> pd.DataFrame:
    base = build_daily_feature_frame(cfg)
    for idx, model in enumerate(MODEL_CANDIDATES):
        daily = load_daily_model(model)
        if idx > 0:
            daily = daily.drop(columns=["fold", "oracle_profit"])
        base = base.merge(daily, on="date", how="inner")

    profit_cols = [f"{model}__profit" for model in MODEL_CANDIDATES]
    base["best_model"] = base[profit_cols].idxmax(axis=1).str.replace("__profit", "", regex=False)
    base["best_profit"] = base[profit_cols].max(axis=1)
    base["champion_gap_to_best"] = base["best_profit"] - base[f"{CHAMPION}__profit"]
    base["champion_predicted_spread"] = base[f"{CHAMPION}__predicted_spread"]
    base["lgb180_predicted_spread"] = base["lgb_baseline_last_180d__predicted_spread"]
    base["champion_lgb180_charge_start_diff"] = (
        base[f"{CHAMPION}__charge_start"] - base["lgb_baseline_last_180d__charge_start"]
    ).abs()
    base["champion_lgb180_discharge_start_diff"] = (
        base[f"{CHAMPION}__discharge_start"]
        - base["lgb_baseline_last_180d__discharge_start"]
    ).abs()
    base["champion_cat180_charge_start_diff"] = (
        base[f"{CHAMPION}__charge_start"] - base["cat_baseline_last_180d__charge_start"]
    ).abs()
    base["champion_cat180_discharge_start_diff"] = (
        base[f"{CHAMPION}__discharge_start"] - base["cat_baseline_last_180d__discharge_start"]
    ).abs()
    return base


def model_profit_col(model: str) -> str:
    return f"{model}__profit"


def select_best_model(df: pd.DataFrame, models: list[str]) -> str:
    means = {model: float(df[model_profit_col(model)].mean()) for model in models}
    return max(means, key=means.get)


def evaluate_rule(df: pd.DataFrame, rule: GateRule) -> pd.Series:
    left_mask = df[rule.feature] <= rule.threshold
    profits = np.where(
        left_mask,
        df[model_profit_col(rule.left_model)],
        df[model_profit_col(rule.right_model)],
    )
    return pd.Series(profits, index=df.index, dtype=float)


def learn_best_single_split_rule(
    train_df: pd.DataFrame,
    *,
    candidate_models: list[str],
    min_bucket_days: int = 12,
) -> GateRule:
    best_rule: GateRule | None = None
    for feature in FEATURE_COLUMNS_FOR_GATES:
        values = train_df[feature].dropna()
        if values.nunique() < 4:
            continue
        thresholds = sorted(set(values.quantile([0.2, 0.33, 0.5, 0.67, 0.8]).to_list()))
        for threshold in thresholds:
            left = train_df[train_df[feature] <= threshold]
            right = train_df[train_df[feature] > threshold]
            if len(left) < min_bucket_days or len(right) < min_bucket_days:
                continue
            left_model = select_best_model(left, candidate_models)
            right_model = select_best_model(right, candidate_models)
            candidate = GateRule(
                feature=feature,
                threshold=float(threshold),
                left_model=left_model,
                right_model=right_model,
                train_mean_profit=float(
                    pd.concat(
                        [
                            left[model_profit_col(left_model)],
                            right[model_profit_col(right_model)],
                        ]
                    ).mean()
                ),
                train_lift_vs_champion=float(
                    pd.concat(
                        [
                            left[model_profit_col(left_model)],
                            right[model_profit_col(right_model)],
                        ]
                    ).mean()
                    - train_df[model_profit_col(CHAMPION)].mean()
                ),
            )
            if best_rule is None or candidate.train_mean_profit > best_rule.train_mean_profit:
                best_rule = candidate
    if best_rule is None:
        champion_mean = float(train_df[model_profit_col(CHAMPION)].mean())
        return GateRule(
            feature="load_mean",
            threshold=float(train_df["load_mean"].median()),
            left_model=CHAMPION,
            right_model=CHAMPION,
            train_mean_profit=champion_mean,
            train_lift_vs_champion=0.0,
        )
    return best_rule


def cross_fold_gate_search(
    df: pd.DataFrame,
    candidate_models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for fold in sorted(df["fold"].unique()):
        train_df = df[df["fold"] != fold].copy()
        valid_df = df[df["fold"] == fold].copy()
        rule = learn_best_single_split_rule(train_df, candidate_models=candidate_models)
        gated_profit = evaluate_rule(valid_df, rule)
        champion_profit = valid_df[model_profit_col(CHAMPION)]
        best_static_model = select_best_model(train_df, candidate_models)
        best_static_profit = valid_df[model_profit_col(best_static_model)]

        fold_rows.append(
            {
                "heldout_fold": fold,
                "feature": rule.feature,
                "threshold": rule.threshold,
                "left_model": rule.left_model,
                "right_model": rule.right_model,
                "train_mean_profit": rule.train_mean_profit,
                "train_lift_vs_champion": rule.train_lift_vs_champion,
                "valid_gated_mean_profit": float(gated_profit.mean()),
                "valid_champion_mean_profit": float(champion_profit.mean()),
                "valid_gated_lift_vs_champion": float(
                    gated_profit.mean() - champion_profit.mean()
                ),
                "best_static_model_from_train": best_static_model,
                "valid_best_static_mean_profit": float(best_static_profit.mean()),
                "valid_days": len(valid_df),
            }
        )

        selected_models = np.where(
            valid_df[rule.feature] <= rule.threshold,
            rule.left_model,
            rule.right_model,
        )
        for idx, row in valid_df.iterrows():
            daily_rows.append(
                {
                    "date": row["date"].date().isoformat(),
                    "fold": fold,
                    "selected_model": selected_models[list(valid_df.index).index(idx)],
                    "gated_profit": float(gated_profit.loc[idx]),
                    "champion_profit": float(champion_profit.loc[idx]),
                    "gated_lift_vs_champion": float(
                        gated_profit.loc[idx] - champion_profit.loc[idx]
                    ),
                    "feature": rule.feature,
                    "feature_value": float(row[rule.feature]),
                    "threshold": rule.threshold,
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(daily_rows)


def summarize_feature_bins(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in FEATURE_COLUMNS_FOR_GATES:
        if df[feature].nunique() < 4:
            continue
        bins = pd.qcut(df[feature], q=3, duplicates="drop")
        for bucket, group in df.groupby(bins, observed=True):
            model_means = {
                model: float(group[model_profit_col(model)].mean()) for model in MODEL_CANDIDATES
            }
            best_model = max(model_means, key=model_means.get)
            rows.append(
                {
                    "feature": feature,
                    "bucket": str(bucket),
                    "days": len(group),
                    "best_model_in_bucket": best_model,
                    "best_model_mean_profit": model_means[best_model],
                    "champion_mean_profit": model_means[CHAMPION],
                    "lift_vs_champion": model_means[best_model] - model_means[CHAMPION],
                    "champion_loss_days": int((group[model_profit_col(CHAMPION)] < 0).sum()),
                    "best_model_share_oracle": float((group["best_model"] == best_model).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["lift_vs_champion", "days"], ascending=False)


def summarize_by_month(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, group in df.groupby("month", sort=True):
        model_means = {
            model: float(group[model_profit_col(model)].mean()) for model in MODEL_CANDIDATES
        }
        best_model = max(model_means, key=model_means.get)
        row: dict[str, object] = {
            "month": int(month),
            "days": len(group),
            "best_model": best_model,
            "best_model_mean_profit": model_means[best_model],
            "champion_mean_profit": model_means[CHAMPION],
            "lift_vs_champion": model_means[best_model] - model_means[CHAMPION],
        }
        for model, value in model_means.items():
            row[f"{model}__mean_profit"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(
    *,
    by_month: pd.DataFrame,
    feature_bins: pd.DataFrame,
    gate_folds: pd.DataFrame,
    gate_daily: pd.DataFrame,
    candidate_models: list[str],
) -> None:
    champion_mean = gate_folds["valid_champion_mean_profit"].mean()
    gated_mean = gate_folds["valid_gated_mean_profit"].mean()
    static_mean = gate_folds["valid_best_static_mean_profit"].mean()
    selected_counts = gate_daily["selected_model"].value_counts().reset_index()
    selected_counts.columns = ["selected_model", "days"]

    lines = [
        "# Gated Diagnostics",
        "",
        "## Scope",
        "",
        "This diagnostic searches simple ex-ante rules using only same-day official "
        "features and model outputs. Hidden true prices are used only for validation.",
        "",
        f"Candidate models: `{', '.join(candidate_models)}`.",
        "",
        "## Cross-Fold Gate Search",
        "",
        "For each held-out validation month, the rule is learned on the other three "
        "months and then evaluated on the held-out month.",
        "",
        gate_folds.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Summary:",
        "",
        f"- Champion mean across held-out folds: `{champion_mean:.4f}`.",
        f"- Single-split gated selector mean: `{gated_mean:.4f}`.",
        f"- Gated lift vs champion: `{gated_mean - champion_mean:.4f}`.",
        f"- Best-static-model-from-train mean: `{static_mean:.4f}`.",
        "",
        "Selected model counts in held-out evaluation:",
        "",
        selected_counts.to_markdown(index=False),
        "",
        "## Month-Level Pattern",
        "",
        by_month.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Month is diagnostic only. It is risky as a direct gate because validation "
        "months are 2025-09..2025-12 while test months are 2026-01..2026-02.",
        "",
        "## Feature Buckets With Largest In-Sample Selector Lift",
        "",
        feature_bins.head(25).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Decision",
        "",
    ]
    if gated_mean > champion_mean:
        lines += [
            "- The simple cross-fold gate improves over champion in this diagnostic.",
            "- Next step should be implementation as a real submit-time selector and a "
            "fresh backtest using generated predictions, not just daily report selection.",
        ]
    else:
        lines += [
            "- The simple cross-fold gate does not beat the current champion.",
            "- Do not promote a gated selector yet.",
            "- Use the feature bucket table only as evidence for targeted feature or "
            "ensemble design, not as a submission rule.",
        ]
    (REPORTS_DIR / "gated_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    df = build_gating_dataset(cfg)
    candidate_models = [
        CHAMPION,
        "lgb_baseline_last_180d",
        "lgb_business_last_180d",
        "cat_baseline_last_180d",
        "ens_base_base180_cat180",
    ]
    by_month = summarize_by_month(df)
    feature_bins = summarize_feature_bins(df)
    gate_folds, gate_daily = cross_fold_gate_search(df, candidate_models)

    df.to_csv(REPORTS_DIR / "gated_daily_dataset.csv", index=False)
    by_month.to_csv(REPORTS_DIR / "gated_month_summary.csv", index=False)
    feature_bins.to_csv(REPORTS_DIR / "gated_feature_bins.csv", index=False)
    gate_folds.to_csv(REPORTS_DIR / "gated_crossfold_rules.csv", index=False)
    gate_daily.to_csv(REPORTS_DIR / "gated_crossfold_daily.csv", index=False)
    write_markdown(
        by_month=by_month,
        feature_bins=feature_bins,
        gate_folds=gate_folds,
        gate_daily=gate_daily,
        candidate_models=candidate_models,
    )

    print(gate_folds.to_string(index=False))
    print(f"dataset_path={REPORTS_DIR / 'gated_daily_dataset.csv'}")
    print(f"month_path={REPORTS_DIR / 'gated_month_summary.csv'}")
    print(f"feature_bins_path={REPORTS_DIR / 'gated_feature_bins.csv'}")
    print(f"rules_path={REPORTS_DIR / 'gated_crossfold_rules.csv'}")
    print(f"daily_path={REPORTS_DIR / 'gated_crossfold_daily.csv'}")
    print(f"markdown_path={REPORTS_DIR / 'gated_diagnostics.md'}")


if __name__ == "__main__":
    main()
