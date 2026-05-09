from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from electricity.data import load_train_frame

REPORTS_DIR = Path("reports")
CONFIG_PATH = Path("configs/base.yaml")

CHAMPION = "ens_champion_segmented6"
EXPERTS = [
    CHAMPION,
    "lgb_segmented_6_last_180d",
    "lgb_segmented_6_margin_core_last_180d",
]

BASE_GATE_FEATURES = [
    "month",
    "dayofweek",
    "is_weekend",
    "load_mean",
    "load_std",
    "load_range",
    "net_load_mean",
    "net_load_std",
    "net_load_range",
    "renewable_ratio_mean",
    "renewable_ratio_std",
    "renewable_ratio_max",
    "wind_ratio_mean",
    "wind_ratio_std",
    "solar_ratio_mean",
    "solar_ratio_max",
    "tie_line_ratio_mean",
    "non_market_ratio_mean",
    "spread_std",
    "charge_start_std",
    "discharge_start_std",
    "champion_segmented6_charge_diff",
    "champion_segmented6_discharge_diff",
    "champion_margin_charge_diff",
    "champion_margin_discharge_diff",
    "champion_segmented6_spread_diff",
    "champion_margin_spread_diff",
    "top1_top2_gap_std",
    "top1_top5_mean_gap_std",
    "champion_top1_top2_gap",
    "segmented6_top1_top2_gap",
    "margin_top1_top2_gap",
    "champion_top1_top5_mean_gap",
    "segmented6_top1_top5_mean_gap",
    "margin_top1_top5_mean_gap",
    "champion_top5_spread_std",
    "segmented6_top5_spread_std",
    "margin_top5_spread_std",
]


@dataclass(frozen=True)
class GateRule:
    feature: str
    threshold: float
    left_expert: str
    right_expert: str
    train_mean_profit: float
    train_lift_vs_champion: float


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def profit_col(expert: str) -> str:
    return f"{expert}__profit"


def load_expert_daily(expert: str) -> pd.DataFrame:
    path = REPORTS_DIR / f"backtest_{expert}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, parse_dates=["date"])
    keep_cols = [
        "date",
        "fold",
        "oracle_profit",
        "profit",
        "regret",
        "charge_start",
        "discharge_start",
        "predicted_spread",
        "top2_spread",
        "top5_spread_mean",
        "top5_spread_std",
        "top1_top2_gap",
        "top1_top5_mean_gap",
    ]
    df = df[keep_cols].copy()
    return df.rename(
        columns={
            "profit": f"{expert}__profit",
            "regret": f"{expert}__regret",
            "charge_start": f"{expert}__charge_start",
            "discharge_start": f"{expert}__discharge_start",
            "predicted_spread": f"{expert}__predicted_spread",
            "top2_spread": f"{expert}__top2_spread",
            "top5_spread_mean": f"{expert}__top5_spread_mean",
            "top5_spread_std": f"{expert}__top5_spread_std",
            "top1_top2_gap": f"{expert}__top1_top2_gap",
            "top1_top5_mean_gap": f"{expert}__top1_top5_mean_gap",
        }
    )


def build_daily_feature_frame(cfg: dict) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    df = load_train_frame(cfg).copy()
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
    df["tie_line_ratio"] = df["联络线预测值"] / load
    df["non_market_ratio"] = df["非市场化机组预测值"] / load

    rows: list[dict[str, object]] = []
    for day, group in df.groupby("date", sort=True):
        if len(group) != 96:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            continue

        rows.append(
            {
                "date": day,
                "month": int(day.month),
                "dayofweek": int(day.dayofweek),
                "is_weekend": int(day.dayofweek >= 5),
                "load_mean": float(group["系统负荷预测值"].mean()),
                "load_std": float(group["系统负荷预测值"].std()),
                "load_range": float(
                    group["系统负荷预测值"].max() - group["系统负荷预测值"].min()
                ),
                "net_load_mean": float(group["net_load"].mean()),
                "net_load_std": float(group["net_load"].std()),
                "net_load_range": float(group["net_load"].max() - group["net_load"].min()),
                "renewable_ratio_mean": float(group["renewable_ratio"].mean()),
                "renewable_ratio_std": float(group["renewable_ratio"].std()),
                "renewable_ratio_max": float(group["renewable_ratio"].max()),
                "wind_ratio_mean": float(group["wind_ratio"].mean()),
                "wind_ratio_std": float(group["wind_ratio"].std()),
                "solar_ratio_mean": float(group["solar_ratio"].mean()),
                "solar_ratio_max": float(group["solar_ratio"].max()),
                "tie_line_ratio_mean": float(group["tie_line_ratio"].mean()),
                "non_market_ratio_mean": float(group["non_market_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_gate_dataset(cfg: dict) -> pd.DataFrame:
    df = build_daily_feature_frame(cfg)
    for idx, expert in enumerate(EXPERTS):
        daily = load_expert_daily(expert)
        if idx > 0:
            daily = daily.drop(columns=["fold", "oracle_profit"])
        df = df.merge(daily, on="date", how="inner")

    profit_cols = [profit_col(expert) for expert in EXPERTS]
    df["best_expert"] = df[profit_cols].idxmax(axis=1).str.replace("__profit", "", regex=False)
    df["best_profit"] = df[profit_cols].max(axis=1)
    df["champion_gap_to_best"] = df["best_profit"] - df[profit_col(CHAMPION)]

    spread_cols = [f"{expert}__predicted_spread" for expert in EXPERTS]
    charge_cols = [f"{expert}__charge_start" for expert in EXPERTS]
    discharge_cols = [f"{expert}__discharge_start" for expert in EXPERTS]
    top1_top2_gap_cols = [f"{expert}__top1_top2_gap" for expert in EXPERTS]
    top1_top5_mean_gap_cols = [f"{expert}__top1_top5_mean_gap" for expert in EXPERTS]
    df["spread_std"] = df[spread_cols].std(axis=1)
    df["charge_start_std"] = df[charge_cols].std(axis=1)
    df["discharge_start_std"] = df[discharge_cols].std(axis=1)
    df["top1_top2_gap_std"] = df[top1_top2_gap_cols].std(axis=1)
    df["top1_top5_mean_gap_std"] = df[top1_top5_mean_gap_cols].std(axis=1)

    segmented = "lgb_segmented_6_last_180d"
    margin = "lgb_segmented_6_margin_core_last_180d"
    df["champion_segmented6_charge_diff"] = (
        df[f"{CHAMPION}__charge_start"] - df[f"{segmented}__charge_start"]
    ).abs()
    df["champion_segmented6_discharge_diff"] = (
        df[f"{CHAMPION}__discharge_start"] - df[f"{segmented}__discharge_start"]
    ).abs()
    df["champion_margin_charge_diff"] = (
        df[f"{CHAMPION}__charge_start"] - df[f"{margin}__charge_start"]
    ).abs()
    df["champion_margin_discharge_diff"] = (
        df[f"{CHAMPION}__discharge_start"] - df[f"{margin}__discharge_start"]
    ).abs()
    df["champion_segmented6_spread_diff"] = (
        df[f"{CHAMPION}__predicted_spread"] - df[f"{segmented}__predicted_spread"]
    ).abs()
    df["champion_margin_spread_diff"] = (
        df[f"{CHAMPION}__predicted_spread"] - df[f"{margin}__predicted_spread"]
    ).abs()
    df["champion_top1_top2_gap"] = df[f"{CHAMPION}__top1_top2_gap"]
    df["segmented6_top1_top2_gap"] = df[f"{segmented}__top1_top2_gap"]
    df["margin_top1_top2_gap"] = df[f"{margin}__top1_top2_gap"]
    df["champion_top1_top5_mean_gap"] = df[f"{CHAMPION}__top1_top5_mean_gap"]
    df["segmented6_top1_top5_mean_gap"] = df[f"{segmented}__top1_top5_mean_gap"]
    df["margin_top1_top5_mean_gap"] = df[f"{margin}__top1_top5_mean_gap"]
    df["champion_top5_spread_std"] = df[f"{CHAMPION}__top5_spread_std"]
    df["segmented6_top5_spread_std"] = df[f"{segmented}__top5_spread_std"]
    df["margin_top5_spread_std"] = df[f"{margin}__top5_spread_std"]
    return df.sort_values("date").reset_index(drop=True)


def select_best_static_expert(df: pd.DataFrame) -> str:
    means = {expert: float(df[profit_col(expert)].mean()) for expert in EXPERTS}
    return max(means, key=means.get)


def selected_profit(df: pd.DataFrame, selected: pd.Series) -> pd.Series:
    values = []
    for idx, expert in selected.items():
        values.append(float(df.loc[idx, profit_col(str(expert))]))
    return pd.Series(values, index=selected.index, dtype=float)


def learn_best_single_rule(train_df: pd.DataFrame, min_bucket_days: int = 15) -> GateRule:
    best_rule: GateRule | None = None
    champion_mean = float(train_df[profit_col(CHAMPION)].mean())
    for feature in BASE_GATE_FEATURES:
        values = train_df[feature].dropna()
        if values.nunique() < 4:
            continue
        thresholds = sorted(set(values.quantile([0.2, 0.33, 0.5, 0.67, 0.8]).to_list()))
        for threshold in thresholds:
            left = train_df[train_df[feature] <= threshold]
            right = train_df[train_df[feature] > threshold]
            if len(left) < min_bucket_days or len(right) < min_bucket_days:
                continue
            left_expert = select_best_static_expert(left)
            right_expert = select_best_static_expert(right)
            profits = pd.concat([left[profit_col(left_expert)], right[profit_col(right_expert)]])
            candidate = GateRule(
                feature=feature,
                threshold=float(threshold),
                left_expert=left_expert,
                right_expert=right_expert,
                train_mean_profit=float(profits.mean()),
                train_lift_vs_champion=float(profits.mean() - champion_mean),
            )
            if best_rule is None or candidate.train_mean_profit > best_rule.train_mean_profit:
                best_rule = candidate

    if best_rule is not None:
        return best_rule
    return GateRule(
        feature="load_mean",
        threshold=float(train_df["load_mean"].median()),
        left_expert=CHAMPION,
        right_expert=CHAMPION,
        train_mean_profit=champion_mean,
        train_lift_vs_champion=0.0,
    )


def apply_rule(df: pd.DataFrame, rule: GateRule) -> pd.Series:
    selected = np.where(df[rule.feature] <= rule.threshold, rule.left_expert, rule.right_expert)
    return pd.Series(selected, index=df.index, dtype=str)


def fit_logistic_gate(train_df: pd.DataFrame) -> Pipeline | None:
    y = train_df["best_expert"].astype(str)
    if y.nunique() < 2:
        return None
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.3,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=2026,
                ),
            ),
        ]
    )
    model.fit(train_df[BASE_GATE_FEATURES], y)
    return model


def summarize_profit(df: pd.DataFrame, profit: pd.Series, prefix: str) -> dict[str, object]:
    oracle = df["oracle_profit"]
    return {
        f"{prefix}_mean_profit": float(profit.mean()),
        f"{prefix}_worst_profit": float(profit.min()),
        f"{prefix}_oracle_ratio": float(profit.mean() / oracle.mean()),
        f"{prefix}_loss_days": int((profit < 0).sum()),
    }


def cross_fold_evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []

    for fold in sorted(df["fold"].unique()):
        train_df = df[df["fold"] != fold].copy()
        valid_df = df[df["fold"] == fold].copy()

        static_expert = select_best_static_expert(train_df)
        static_selected = pd.Series(static_expert, index=valid_df.index, dtype=str)
        static_profit = selected_profit(valid_df, static_selected)

        rule = learn_best_single_rule(train_df)
        rule_selected = apply_rule(valid_df, rule)
        rule_profit = selected_profit(valid_df, rule_selected)

        logistic = fit_logistic_gate(train_df)
        if logistic is None:
            logistic_selected = static_selected
            logistic_mode = "static_fallback"
        else:
            logistic_pred = logistic.predict(valid_df[BASE_GATE_FEATURES])
            logistic_selected = pd.Series(logistic_pred, index=valid_df.index, dtype=str)
            logistic_mode = "logistic"
        logistic_profit = selected_profit(valid_df, logistic_selected)

        champion_profit = valid_df[profit_col(CHAMPION)]
        oracle_profit = valid_df[[profit_col(expert) for expert in EXPERTS]].max(axis=1)
        oracle_selected = (
            valid_df[[profit_col(expert) for expert in EXPERTS]]
            .idxmax(axis=1)
            .str.replace("__profit", "", regex=False)
        )

        row: dict[str, object] = {
            "heldout_fold": fold,
            "valid_days": len(valid_df),
            "best_static_expert_from_train": static_expert,
            "rule_feature": rule.feature,
            "rule_threshold": rule.threshold,
            "rule_left_expert": rule.left_expert,
            "rule_right_expert": rule.right_expert,
            "rule_train_lift_vs_champion": rule.train_lift_vs_champion,
            "logistic_mode": logistic_mode,
        }
        row.update(summarize_profit(valid_df, champion_profit, "champion"))
        row.update(summarize_profit(valid_df, static_profit, "static"))
        row.update(summarize_profit(valid_df, rule_profit, "rule_gate"))
        row.update(summarize_profit(valid_df, logistic_profit, "logistic_gate"))
        row.update(summarize_profit(valid_df, oracle_profit, "oracle_selector"))
        fold_rows.append(row)

        oracle_rows.append(
            {
                "fold": fold,
                "oracle_selector_lift_vs_champion": float(
                    oracle_profit.mean() - champion_profit.mean()
                ),
                "oracle_selected_counts": ";".join(
                    f"{k}:{v}" for k, v in oracle_selected.value_counts().to_dict().items()
                ),
            }
        )

        for idx, row_data in valid_df.iterrows():
            daily_rows.append(
                {
                    "date": row_data["date"].date().isoformat(),
                    "fold": fold,
                    "champion_profit": float(champion_profit.loc[idx]),
                    "static_selected": str(static_selected.loc[idx]),
                    "static_profit": float(static_profit.loc[idx]),
                    "rule_selected": str(rule_selected.loc[idx]),
                    "rule_profit": float(rule_profit.loc[idx]),
                    "logistic_selected": str(logistic_selected.loc[idx]),
                    "logistic_profit": float(logistic_profit.loc[idx]),
                    "oracle_selected": str(oracle_selected.loc[idx]),
                    "oracle_selector_profit": float(oracle_profit.loc[idx]),
                    "best_expert": str(row_data["best_expert"]),
                    "best_profit": float(row_data["best_profit"]),
                }
            )

    return pd.DataFrame(fold_rows), pd.DataFrame(daily_rows), pd.DataFrame(oracle_rows)


def overall_from_daily(daily: pd.DataFrame, method: str) -> dict[str, object]:
    profit = daily[f"{method}_profit"]
    champion = daily["champion_profit"]
    return {
        "method": method,
        "mean_profit": float(profit.mean()),
        "lift_vs_champion": float(profit.mean() - champion.mean()),
        "worst_day_profit": float(profit.min()),
        "loss_days": int((profit < 0).sum()),
        "selected_counts": ";".join(
            f"{k}:{v}" for k, v in daily[f"{method}_selected"].value_counts().to_dict().items()
        )
        if f"{method}_selected" in daily.columns
        else "",
    }


def write_markdown(
    *,
    folds: pd.DataFrame,
    daily: pd.DataFrame,
    overall: pd.DataFrame,
    oracle: pd.DataFrame,
) -> None:
    lines = [
        "# Strategy Gate Diagnostics",
        "",
        "## Scope",
        "",
        "This diagnostic evaluates a small day-level selector over three existing experts. "
        "Hidden true prices are used only to score validation days.",
        "",
        f"Experts: `{', '.join(EXPERTS)}`.",
        "",
        "## Overall Held-Out Result",
        "",
        overall.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Fold Detail",
        "",
        folds.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Oracle Selector Upper Bound",
        "",
        oracle.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Decision",
        "",
    ]

    champion_mean = daily["champion_profit"].mean()
    best_gate = overall[overall["method"].isin(["rule", "logistic"])].sort_values(
        "mean_profit",
        ascending=False,
    )
    if not best_gate.empty and float(best_gate.iloc[0]["mean_profit"]) > champion_mean:
        lines += [
            "- A held-out gate beats the current champion in this diagnostic.",
            "- Next step is to implement a submit-time gate and validate the generated "
            "power schedule, not to replace the champion blindly.",
        ]
    else:
        lines += [
            "- The held-out gates do not beat the current champion.",
            "- Keep this as a diagnostic unless a stricter gate variant improves "
            "held-out profit or stability.",
        ]

    REPORTS_DIR.joinpath("strategy_gate_diagnostics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    dataset = build_gate_dataset(cfg)
    folds, daily, oracle = cross_fold_evaluate(dataset)

    overall = pd.DataFrame(
        [
            overall_from_daily(daily.rename(columns={"champion_profit": "champion_profit"}), name)
            for name in ["static", "rule", "logistic", "oracle_selector"]
        ]
    )
    champion_overall = pd.DataFrame(
        [
            {
                "method": "champion",
                "mean_profit": float(daily["champion_profit"].mean()),
                "lift_vs_champion": 0.0,
                "worst_day_profit": float(daily["champion_profit"].min()),
                "loss_days": int((daily["champion_profit"] < 0).sum()),
                "selected_counts": f"{CHAMPION}:{len(daily)}",
            }
        ]
    )
    overall = pd.concat([champion_overall, overall], ignore_index=True)

    dataset.to_csv(REPORTS_DIR / "strategy_gate_dataset.csv", index=False)
    folds.to_csv(REPORTS_DIR / "strategy_gate_crossfold_summary.csv", index=False)
    daily.to_csv(REPORTS_DIR / "strategy_gate_crossfold_daily.csv", index=False)
    oracle.to_csv(REPORTS_DIR / "strategy_gate_oracle_summary.csv", index=False)
    overall.to_csv(REPORTS_DIR / "strategy_gate_overall.csv", index=False)
    write_markdown(folds=folds, daily=daily, overall=overall, oracle=oracle)

    print(overall.to_string(index=False))
    print(f"dataset_path={REPORTS_DIR / 'strategy_gate_dataset.csv'}")
    print(f"summary_path={REPORTS_DIR / 'strategy_gate_crossfold_summary.csv'}")
    print(f"daily_path={REPORTS_DIR / 'strategy_gate_crossfold_daily.csv'}")
    print(f"markdown_path={REPORTS_DIR / 'strategy_gate_diagnostics.md'}")


if __name__ == "__main__":
    main()
