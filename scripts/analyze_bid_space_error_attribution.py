from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity.data import load_train_frame
from electricity.features.bid_space import (
    ACTUAL_COLS,
    PRED_COLS,
    add_bid_space_columns,
    markdown_table,
)

CONFIG_PATH = Path("configs/base.yaml")
REPORTS_DIR = Path("reports")
CHAMPION_DAILY = REPORTS_DIR / "current_champion_daily_error.csv"

COMPONENT_SIGNS = {
    "load": 1.0,
    "renewable": -1.0,
    "tie_line": -1.0,
    "hydro": -1.0,
    "non_market": -1.0,
}
COMPONENT_CN = {
    "load": "系统负荷",
    "renewable": "风光总加",
    "tie_line": "联络线",
    "hydro": "水电",
    "non_market": "非市场化机组",
}
SEGMENT_BINS = [0, 24, 40, 56, 72, 88, 96]
SEGMENT_LABELS = ["00_06", "06_10", "10_14", "14_18", "18_22", "22_24"]


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_attribution_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = add_bid_space_columns(df)
    for name, sign in COMPONENT_SIGNS.items():
        raw_col = f"{name}_raw_error"
        contrib_col = f"{name}_bid_space_contrib"
        abs_contrib_col = f"{name}_abs_bid_space_contrib"
        out[raw_col] = out[PRED_COLS[name]] - out[ACTUAL_COLS[name]]
        out[contrib_col] = sign * out[raw_col]
        out[abs_contrib_col] = out[contrib_col].abs()
    contrib_cols = [f"{name}_bid_space_contrib" for name in COMPONENT_SIGNS]
    out["bid_space_error_reconstructed"] = out[contrib_cols].sum(axis=1)
    out["bid_space_error_reconstruction_delta"] = (
        out["bid_space_error"] - out["bid_space_error_reconstructed"]
    )
    out["segment"] = pd.cut(
        out["slot"],
        bins=SEGMENT_BINS,
        labels=SEGMENT_LABELS,
        right=False,
        include_lowest=True,
    )
    return out


def dominant_component(row: pd.Series) -> str:
    values = {
        name: row[f"{name}_abs_bid_space_contrib"] for name in COMPONENT_SIGNS
    }
    return max(values, key=values.get)


def component_agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg: dict[str, tuple[str, str] | tuple[str, object]] = {
        "rows": ("bid_space_error", "count"),
        "days": ("date", "nunique"),
        "bid_space_error_mean": ("bid_space_error", "mean"),
        "bid_space_error_mae": ("bid_space_error", lambda s: s.abs().mean()),
        "bid_space_error_rmse": ("bid_space_error", lambda s: np.sqrt(np.mean(s**2))),
        "reconstruction_delta_max_abs": (
            "bid_space_error_reconstruction_delta",
            lambda s: s.abs().max(),
        ),
    }
    for name in COMPONENT_SIGNS:
        agg[f"{name}_raw_error_mean"] = (f"{name}_raw_error", "mean")
        agg[f"{name}_contrib_mean"] = (f"{name}_bid_space_contrib", "mean")
        agg[f"{name}_abs_contrib_mean"] = (f"{name}_abs_bid_space_contrib", "mean")
    out = df.groupby(group_cols, observed=True).agg(**agg).reset_index()
    abs_cols = [f"{name}_abs_contrib_mean" for name in COMPONENT_SIGNS]
    out["dominant_abs_component"] = out[abs_cols].idxmax(axis=1).str.replace(
        "_abs_contrib_mean", "", regex=False
    )
    out["dominant_abs_component_cn"] = out["dominant_abs_component"].map(COMPONENT_CN)
    return out


def daily_attribution(df: pd.DataFrame) -> pd.DataFrame:
    daily = component_agg(df, ["date", "month"])
    if CHAMPION_DAILY.exists():
        champion = pd.read_csv(CHAMPION_DAILY, parse_dates=["date"])
        keep = [
            "date",
            "fold",
            "profit",
            "oracle_profit",
            "regret",
            "profit_ratio_day",
            "loss_day",
            "charge_start",
            "discharge_start",
            "oracle_charge_start",
            "oracle_discharge_start",
            "charge_start_gap",
            "discharge_start_gap",
        ]
        daily = daily.merge(champion[keep], on="date", how="left")
        daily["loss_day"] = daily["loss_day"].fillna(False).astype(bool)
        if daily["regret"].notna().any():
            threshold = daily.loc[daily["regret"].notna(), "regret"].quantile(0.9)
            daily["high_regret_day"] = daily["regret"] >= threshold
        else:
            daily["high_regret_day"] = False
    else:
        daily["high_regret_day"] = False
    return daily


def cohort_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if "profit" not in daily.columns:
        return pd.DataFrame()
    valid = daily[daily["profit"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    cohorts = [
        ("all_valid", valid),
        ("loss_days", valid[valid["loss_day"]]),
        ("non_loss_days", valid[~valid["loss_day"]]),
        ("high_regret_p90", valid[valid["high_regret_day"]]),
        ("non_high_regret", valid[~valid["high_regret_day"]]),
    ]
    for name, group in cohorts:
        if group.empty:
            continue
        row: dict[str, object] = {
            "cohort": name,
            "days": len(group),
            "profit_mean": group["profit"].mean(),
            "regret_mean": group["regret"].mean(),
            "bid_space_error_mean": group["bid_space_error_mean"].mean(),
            "bid_space_error_mae": group["bid_space_error_mae"].mean(),
        }
        for component in COMPONENT_SIGNS:
            row[f"{component}_contrib_mean"] = group[f"{component}_contrib_mean"].mean()
            row[f"{component}_abs_contrib_mean"] = group[
                f"{component}_abs_contrib_mean"
            ].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def component_long(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        base = {col: row[col] for col in group_cols}
        for component in COMPONENT_SIGNS:
            rows.append(
                {
                    **base,
                    "component": component,
                    "component_cn": COMPONENT_CN[component],
                    "contrib_mean": row[f"{component}_contrib_mean"],
                    "abs_contrib_mean": row[f"{component}_abs_contrib_mean"],
                    "raw_error_mean": row[f"{component}_raw_error_mean"],
                }
            )
    return pd.DataFrame(rows)


def plot_attribution(month_summary: pd.DataFrame, segment_summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    month_long = component_long(month_summary, ["month"])
    pivot_abs = month_long.pivot(index="month", columns="component", values="abs_contrib_mean")
    pivot_abs = pivot_abs.sort_index()
    pivot_signed = month_long.pivot(index="month", columns="component", values="contrib_mean")
    pivot_signed = pivot_signed.sort_index()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    pivot_abs.plot(kind="bar", stacked=True, ax=axes[0])
    axes[0].set_title("Monthly mean absolute contribution to bid-space error")
    axes[0].set_ylabel("mean absolute contribution")
    axes[0].grid(True, axis="y", alpha=0.25)
    pivot_signed.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Monthly signed contribution to bid-space error")
    axes[1].set_ylabel("signed contribution")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.savefig(REPORTS_DIR / "bid_space_error_component_monthly.png", dpi=160)
    plt.close(fig)

    key_segments = segment_summary[segment_summary["segment"].isin(["10_14", "14_18"])]
    seg_long = component_long(key_segments, ["month", "segment"])
    for segment in ["10_14", "14_18"]:
        part = seg_long[seg_long["segment"] == segment]
        if part.empty:
            continue
        pivot = part.pivot(index="month", columns="component", values="contrib_mean")
        fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
        pivot.sort_index().plot(kind="bar", ax=ax)
        ax.set_title(f"Signed bid-space error contribution by component: {segment}")
        ax.set_ylabel("signed contribution")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.25)
        fig.savefig(REPORTS_DIR / f"bid_space_error_component_{segment}.png", dpi=160)
        plt.close(fig)


def write_report(
    *,
    month_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    daily: pd.DataFrame,
    cohort: pd.DataFrame,
) -> None:
    midday = segment_summary[segment_summary["segment"].isin(["10_14", "14_18"])]
    midday = midday.sort_values("bid_space_error_mean").head(18)
    worst_abs = segment_summary.sort_values("bid_space_error_mae", ascending=False).head(18)
    valid_daily = daily[daily.get("profit", pd.Series(dtype=float)).notna()].copy()
    high_regret = pd.DataFrame()
    loss_days = pd.DataFrame()
    if not valid_daily.empty:
        high_regret = valid_daily.sort_values("regret", ascending=False).head(15)
        loss_days = valid_daily[valid_daily["loss_day"]].sort_values("profit").head(15)

    day_cols = [
        "date",
        "fold",
        "profit",
        "regret",
        "bid_space_error_mean",
        "bid_space_error_mae",
        "dominant_abs_component_cn",
        "load_contrib_mean",
        "renewable_contrib_mean",
        "tie_line_contrib_mean",
        "hydro_contrib_mean",
        "non_market_contrib_mean",
        "charge_start_gap",
        "discharge_start_gap",
    ]
    day_cols = [col for col in day_cols if col in daily.columns]

    max_delta = daily["reconstruction_delta_max_abs"].max()
    lines = [
        "# Bid Space Error Attribution",
        "",
        "This report decomposes forecast bid-space error into component forecast errors.",
        "",
        "Definition:",
        "",
        "`bid_space = load - renewable - tie_line - hydro - non_market`",
        "",
        "So each component contribution to bid-space error is:",
        "",
        "- `load`: forecast minus actual",
        "- `renewable/tie_line/hydro/non_market`: actual minus forecast",
        "",
        f"Reconstruction max absolute delta: `{max_delta:.8f}`.",
        "",
        "## Monthly Summary",
        "",
        markdown_table(month_summary, floatfmt=".4f"),
        "",
        "## Midday Negative Bias Segments",
        "",
        "Most negative rows mean forecast bid_space is lower than actual bid_space.",
        "",
        markdown_table(midday, floatfmt=".4f"),
        "",
        "## Largest Absolute Error Segments",
        "",
        markdown_table(worst_abs, floatfmt=".4f"),
        "",
        "## Champion-Day Cohort Summary",
        "",
        markdown_table(cohort, floatfmt=".4f"),
        "",
        "## Champion High-Regret Days",
        "",
        markdown_table(high_regret[day_cols], floatfmt=".4f"),
        "",
        "## Champion Loss Days",
        "",
        markdown_table(loss_days[day_cols], floatfmt=".4f"),
        "",
        "## Readout",
        "",
        "- Use signed component contributions to explain the direction of bid-space bias.",
        "- Use absolute contributions to identify which component dominates uncertainty.",
        "- If high-regret days have larger component errors, add reliability features "
        "for reranking.",
        "- If a component has systematic month/segment bias, consider bias-corrected "
        "derived features.",
        "",
        "## Artifacts",
        "",
        "- `reports/bid_space_error_month_summary.csv`",
        "- `reports/bid_space_error_segment_summary.csv`",
        "- `reports/bid_space_error_daily_attribution.csv`",
        "- `reports/bid_space_error_cohort_summary.csv`",
        "- `reports/bid_space_error_component_monthly.png`",
        "- `reports/bid_space_error_component_10_14.png`",
        "- `reports/bid_space_error_component_14_18.png`",
    ]
    (REPORTS_DIR / "bid_space_error_attribution.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    df = add_attribution_columns(load_train_frame(cfg))
    month_summary = component_agg(df, ["month"])
    segment_summary = component_agg(df, ["month", "segment"])
    daily = daily_attribution(df)
    cohort = cohort_summary(daily)

    month_summary.to_csv(REPORTS_DIR / "bid_space_error_month_summary.csv", index=False)
    segment_summary.to_csv(REPORTS_DIR / "bid_space_error_segment_summary.csv", index=False)
    daily.to_csv(REPORTS_DIR / "bid_space_error_daily_attribution.csv", index=False)
    cohort.to_csv(REPORTS_DIR / "bid_space_error_cohort_summary.csv", index=False)
    plot_attribution(month_summary, segment_summary)
    write_report(
        month_summary=month_summary,
        segment_summary=segment_summary,
        daily=daily,
        cohort=cohort,
    )
    print(month_summary.to_string(index=False))
    print(REPORTS_DIR / "bid_space_error_attribution.md")


if __name__ == "__main__":
    main()
