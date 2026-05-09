from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS_DIR = Path("reports")
MODELS = [
    "ens_baseline_baseline_last_180d",
    "lgb_baseline_last_180d",
    "cat_baseline_last_180d",
    "cat_baseline",
    "xgb_baseline_last_180d",
    "lgb_baseline",
    "lgb_business",
    "ens_business_business_last_180d",
]
CHAMPION = "ens_baseline_baseline_last_180d"
CAT_CANDIDATE = "cat_baseline_last_180d"
LGB_180 = "lgb_baseline_last_180d"


def load_daily(model: str) -> pd.DataFrame:
    path = REPORTS_DIR / f"backtest_{model}_daily.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["model"] = model
    for col in ["charge_start", "discharge_start", "oracle_charge_start", "oracle_discharge_start"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["abs_charge_gap"] = (df["charge_start"] - df["oracle_charge_start"]).abs()
    df["abs_discharge_gap"] = (df["discharge_start"] - df["oracle_discharge_start"]).abs()
    df["both_window_abs_gap"] = df["abs_charge_gap"] + df["abs_discharge_gap"]
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.dayofweek
    return df


def summarize_model(df: pd.DataFrame) -> dict:
    return {
        "model": df["model"].iloc[0],
        "days": len(df),
        "mean_profit": df["profit"].mean(),
        "median_profit": df["profit"].median(),
        "min_profit": df["profit"].min(),
        "loss_days": int((df["profit"] < 0).sum()),
        "mean_oracle_ratio": df["profit_ratio_day"].mean(),
        "mean_regret": df["regret"].mean(),
        "p90_regret": df["regret"].quantile(0.9),
        "mean_abs_charge_gap": df["abs_charge_gap"].mean(),
        "mean_abs_discharge_gap": df["abs_discharge_gap"].mean(),
        "exact_charge_hit_rate": (df["charge_start_gap"] == 0).mean(),
        "exact_discharge_hit_rate": (df["discharge_start_gap"] == 0).mean(),
        "both_within_2_slots_rate": (
            (df["abs_charge_gap"] <= 2) & (df["abs_discharge_gap"] <= 2)
        ).mean(),
    }


def pairwise_complementarity(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for left, right in combinations(frames, 2):
        a = frames[left][["date", "fold", "profit", "regret", "charge_start", "discharge_start"]]
        b = frames[right][["date", "profit", "regret", "charge_start", "discharge_start"]]
        merged = a.merge(b, on="date", suffixes=("_left", "_right"))
        left_profit = merged["profit_left"]
        right_profit = merged["profit_right"]
        different_window = (
            (merged["charge_start_left"] != merged["charge_start_right"])
            | (merged["discharge_start_left"] != merged["discharge_start_right"])
        )
        oracle_selector = np.maximum(left_profit, right_profit)
        rows.append(
            {
                "left_model": left,
                "right_model": right,
                "days": len(merged),
                "left_mean_profit": left_profit.mean(),
                "right_mean_profit": right_profit.mean(),
                "right_minus_left_mean": (right_profit - left_profit).mean(),
                "right_beats_left_days": int((right_profit > left_profit).sum()),
                "left_beats_right_days": int((left_profit > right_profit).sum()),
                "same_profit_days": int((left_profit == right_profit).sum()),
                "different_window_days": int(different_window.sum()),
                "different_window_ratio": different_window.mean(),
                "oracle_selector_mean_profit": oracle_selector.mean(),
                "oracle_selector_lift_vs_best_single": oracle_selector.mean()
                - max(left_profit.mean(), right_profit.mean()),
                "left_loss_days_right_nonloss": int(
                    ((left_profit < 0) & (right_profit >= 0)).sum()
                ),
                "right_loss_days_left_nonloss": int(
                    ((right_profit < 0) & (left_profit >= 0)).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["oracle_selector_lift_vs_best_single", "right_minus_left_mean"], ascending=False
    )


def build_daily_comparison(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = frames[CHAMPION][
        [
            "date",
            "fold",
            "oracle_profit",
            "oracle_charge_start",
            "oracle_discharge_start",
        ]
    ].copy()
    for model in [CHAMPION, LGB_180, CAT_CANDIDATE, "cat_baseline", "xgb_baseline_last_180d"]:
        df = frames[model][
            [
                "date",
                "profit",
                "regret",
                "profit_ratio_day",
                "charge_start",
                "discharge_start",
                "charge_start_gap",
                "discharge_start_gap",
                "predicted_spread",
            ]
        ].copy()
        df = df.rename(columns={col: f"{model}__{col}" for col in df.columns if col != "date"})
        base = base.merge(df, on="date", how="left")
    profit_cols = [c for c in base.columns if c.endswith("__profit")]
    base["best_available_model"] = base[profit_cols].idxmax(axis=1).str.replace(
        "__profit", "", regex=False
    )
    base["best_available_profit"] = base[profit_cols].max(axis=1)
    base["champion_profit_gap_to_best_available"] = (
        base["best_available_profit"] - base[f"{CHAMPION}__profit"]
    )
    return base.sort_values(
        ["champion_profit_gap_to_best_available", "date"], ascending=[False, True]
    )


def write_markdown(
    model_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    champion: pd.DataFrame,
    daily_comparison: pd.DataFrame,
) -> None:
    path = REPORTS_DIR / "daily_diagnostics.md"
    loss_days = champion[champion["profit"] < 0].sort_values("profit")
    high_regret = champion.sort_values("regret", ascending=False).head(15)
    cat_vs_lgb = pairwise[
        (pairwise["left_model"] == LGB_180) & (pairwise["right_model"] == CAT_CANDIDATE)
    ].iloc[0]
    cat_vs_champion = pairwise[
        (pairwise["left_model"] == CHAMPION) & (pairwise["right_model"] == CAT_CANDIDATE)
    ].iloc[0]
    best_help = daily_comparison.head(15)

    lines = [
        "# Daily Diagnostics",
        "",
        "## Model Summary",
        "",
        model_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Champion Error Pattern",
        "",
        f"- Champion loss days: `{len(loss_days)}`.",
        f"- Champion mean profit: `{champion['profit'].mean():.4f}`.",
        f"- Champion mean regret: `{champion['regret'].mean():.4f}`.",
        "- Champion mean absolute charge gap: "
        f"`{champion['abs_charge_gap'].mean():.4f}` slots.",
        "- Champion mean absolute discharge gap: "
        f"`{champion['abs_discharge_gap'].mean():.4f}` slots.",
        "- Champion exact charge window hit rate: "
        f"`{(champion['charge_start_gap'] == 0).mean():.4f}`.",
        "- Champion exact discharge window hit rate: "
        f"`{(champion['discharge_start_gap'] == 0).mean():.4f}`.",
        "",
        "### Champion Loss Days",
        "",
        loss_days[
            [
                "date",
                "fold",
                "profit",
                "oracle_profit",
                "regret",
                "profit_ratio_day",
                "charge_start",
                "discharge_start",
                "oracle_charge_start",
                "oracle_discharge_start",
                "charge_start_gap",
                "discharge_start_gap",
                "predicted_spread",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "### Highest-Regret Champion Days",
        "",
        high_regret[
            [
                "date",
                "fold",
                "profit",
                "oracle_profit",
                "regret",
                "profit_ratio_day",
                "charge_start_gap",
                "discharge_start_gap",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## CatBoost Complementarity",
        "",
        "- CatBoost 180d beats LightGBM 180d on "
        f"`{int(cat_vs_lgb['right_beats_left_days'])}` days; LightGBM 180d beats "
        f"CatBoost 180d on `{int(cat_vs_lgb['left_beats_right_days'])}` days.",
        "- Perfect daily selector upper bound for LGB180 vs Cat180: "
        f"`{cat_vs_lgb['oracle_selector_mean_profit']:.4f}`, lift over best single: "
        f"`{cat_vs_lgb['oracle_selector_lift_vs_best_single']:.4f}`.",
        "- CatBoost 180d beats champion on "
        f"`{int(cat_vs_champion['right_beats_left_days'])}` days; champion beats "
        f"CatBoost 180d on `{int(cat_vs_champion['left_beats_right_days'])}` days.",
        "- Perfect daily selector upper bound for champion vs Cat180: "
        f"`{cat_vs_champion['oracle_selector_mean_profit']:.4f}`, lift over champion: "
        f"`{cat_vs_champion['oracle_selector_lift_vs_best_single']:.4f}`.",
        "",
        "Interpretation: a positive perfect-selector lift means there is some daily "
        "complementarity, but this is not directly usable in test unless we can "
        "predict ex ante when CatBoost is better.",
        "",
        "### Days Where Another Available Model Helps Most Over Champion",
        "",
        best_help[
            [
                "date",
                "fold",
                "oracle_profit",
                f"{CHAMPION}__profit",
                "best_available_model",
                "best_available_profit",
                "champion_profit_gap_to_best_available",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Next Actions From Diagnostics",
        "",
        "- Do not promote CatBoost directly: its mean profit remains below champion.",
        "- Check whether the days where CatBoost beats champion are predictable from "
        "same-day exogenous features; if not, do not ensemble or gate blindly.",
        "- Prioritize features that improve charge/discharge window timing, because "
        "high-regret days mostly come from large window start gaps.",
        "- Next implementation candidate: add NWP weather summary features, then rerun "
        "LGB180 and champion-style ensemble.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    frames = {model: load_daily(model) for model in MODELS}
    model_summary = pd.DataFrame(summarize_model(df) for df in frames.values()).sort_values(
        ["mean_profit", "min_profit"], ascending=False
    )
    pairwise = pairwise_complementarity(frames)
    daily_comparison = build_daily_comparison(frames)

    model_summary.to_csv(REPORTS_DIR / "daily_model_summary.csv", index=False)
    pairwise.to_csv(REPORTS_DIR / "daily_model_complementarity.csv", index=False)
    daily_comparison.to_csv(REPORTS_DIR / "daily_model_comparison.csv", index=False)
    write_markdown(model_summary, pairwise, frames[CHAMPION], daily_comparison)

    print(model_summary.to_string(index=False))
    print(f"summary_path={REPORTS_DIR / 'daily_model_summary.csv'}")
    print(f"complementarity_path={REPORTS_DIR / 'daily_model_complementarity.csv'}")
    print(f"comparison_path={REPORTS_DIR / 'daily_model_comparison.csv'}")
    print(f"markdown_path={REPORTS_DIR / 'daily_diagnostics.md'}")


if __name__ == "__main__":
    main()
