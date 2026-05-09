from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity.data import load_train_frame

REPORTS_DIR = Path("reports")
CONFIG_PATH = Path("configs/base.yaml")
CHAMPION = "ens_champion_segmented6"
PAIRS = [
    ("系统负荷实际值", "系统负荷预测值", "load"),
    ("风光总加实际值", "风光总加预测值", "renewable_total"),
    ("联络线实际值", "联络线预测值", "tie_line"),
    ("风电实际值", "风电预测值", "wind"),
    ("光伏实际值", "光伏预测值", "solar"),
    ("水电实际值", "水电预测值", "hydro"),
    ("非市场化机组实际值", "非市场化机组预测值", "non_market"),
]
TIME_BUCKETS = [
    ("night", 0, 24),
    ("morning_ramp", 24, 40),
    ("midday", 40, 56),
    ("afternoon", 56, 72),
    ("evening_peak", 72, 88),
    ("late_night", 88, 96),
]


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_champion_daily() -> pd.DataFrame:
    path = REPORTS_DIR / f"backtest_{CHAMPION}_daily.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_boundary_frame(cfg: dict) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    df = load_train_frame(cfg).copy()
    df["date"] = df[time_col].dt.normalize()
    df["slot"] = df[time_col].dt.hour * 4 + df[time_col].dt.minute // 15
    df["month"] = df[time_col].dt.month
    return df


def daily_error_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day, group in df.groupby("date", sort=True):
        if len(group) != 96:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            continue

        row: dict[str, object] = {
            "date": day,
            "month": int(group["month"].iloc[0]),
        }
        for actual_col, pred_col, alias in PAIRS:
            diff = group[pred_col].to_numpy(dtype=float) - group[actual_col].to_numpy(dtype=float)
            row[f"{alias}_mae_day"] = float(np.mean(np.abs(diff)))
            row[f"{alias}_rmse_day"] = float(np.sqrt(np.mean(diff**2)))
            row[f"{alias}_bias_day"] = float(np.mean(diff))
            denom = np.mean(np.abs(group[actual_col].to_numpy(dtype=float))) + 1e-6
            row[f"{alias}_nmae_day"] = float(np.mean(np.abs(diff)) / denom)
        rows.append(row)
    return pd.DataFrame(rows)


def intraday_bucket_error_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bucket_name, start, end in TIME_BUCKETS:
        part = df[(df["slot"] >= start) & (df["slot"] < end)].copy()
        for day, group in part.groupby("date", sort=True):
            row: dict[str, object] = {
                "date": day,
                "bucket": bucket_name,
                "month": int(group["month"].iloc[0]),
            }
            for actual_col, pred_col, alias in PAIRS:
                diff = group[pred_col].to_numpy(dtype=float) - group[actual_col].to_numpy(
                    dtype=float
                )
                row[f"{alias}_mae"] = float(np.mean(np.abs(diff)))
                row[f"{alias}_bias"] = float(np.mean(diff))
            rows.append(row)
    return pd.DataFrame(rows)


def correlation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target_cols = ["profit", "regret", "charge_start_gap", "discharge_start_gap"]
    for _, _, alias in PAIRS:
        for metric in ["mae_day", "rmse_day", "bias_day", "nmae_day"]:
            feature = f"{alias}_{metric}"
            series = df[feature]
            for target in target_cols:
                corr = float(series.corr(df[target], method="spearman"))
                rows.append(
                    {
                        "feature": feature,
                        "target": target,
                        "spearman_corr": corr,
                        "abs_corr": abs(corr),
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def loss_day_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    loss_mask = df["profit"] < 0
    for _, _, alias in PAIRS:
        for metric in ["mae_day", "rmse_day", "nmae_day"]:
            col = f"{alias}_{metric}"
            rows.append(
                {
                    "feature": col,
                    "loss_day_mean": float(df.loc[loss_mask, col].mean()),
                    "non_loss_day_mean": float(df.loc[~loss_mask, col].mean()),
                    "gap_loss_minus_nonloss": float(
                        df.loc[loss_mask, col].mean() - df.loc[~loss_mask, col].mean()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values("gap_loss_minus_nonloss", ascending=False)


def month_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg: dict[str, tuple[str, str]] = {
        "profit": ("profit", "mean"),
        "regret": ("regret", "mean"),
    }
    for _, _, alias in PAIRS:
        agg[f"{alias}_mae_day"] = (f"{alias}_mae_day", "mean")
        agg[f"{alias}_nmae_day"] = (f"{alias}_nmae_day", "mean")
    out = df.groupby("month", as_index=False).agg(**agg)
    return out


def worst_days(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["date", "profit", "regret", "charge_start_gap", "discharge_start_gap"]
    for _, _, alias in PAIRS:
        keep += [f"{alias}_mae_day", f"{alias}_nmae_day"]
    return df.sort_values(["regret", "profit"], ascending=[False, True])[keep].head(20)


def bucket_loss_comparison(bucket_df: pd.DataFrame, champion_df: pd.DataFrame) -> pd.DataFrame:
    merged = bucket_df.merge(champion_df[["date", "profit", "regret"]], on="date", how="inner")
    rows: list[dict[str, object]] = []
    loss_mask = merged["profit"] < 0
    for bucket_name, group in merged.groupby("bucket", sort=False):
        for _, _, alias in PAIRS:
            col = f"{alias}_mae"
            rows.append(
                {
                    "bucket": bucket_name,
                    "feature": col,
                    "loss_day_mean": float(group.loc[loss_mask.loc[group.index], col].mean()),
                    "non_loss_day_mean": float(group.loc[~loss_mask.loc[group.index], col].mean()),
                    "gap_loss_minus_nonloss": float(
                        group.loc[loss_mask.loc[group.index], col].mean()
                        - group.loc[~loss_mask.loc[group.index], col].mean()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["gap_loss_minus_nonloss", "bucket"], ascending=[False, True]
    )


def write_markdown(
    *,
    month_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    loss_df: pd.DataFrame,
    worst_df: pd.DataFrame,
    bucket_loss_df: pd.DataFrame,
) -> None:
    top_corr = corr_df.head(20)
    top_loss = loss_df.head(20)
    top_bucket = bucket_loss_df.head(20)
    lines = [
        "# Boundary Error Impact",
        "",
        f"Champion analyzed: `{CHAMPION}`.",
        "",
        "## Month Summary",
        "",
        month_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Strongest Spearman Correlations",
        "",
        top_corr.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Error Metrics That Worsen on Loss Days",
        "",
        top_loss.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Intraday Error Buckets With Largest Loss-Day Gap",
        "",
        top_bucket.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Worst-Regret Days With Boundary Error Context",
        "",
        worst_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation Rules",
        "",
        "- Correlation is diagnostic, not causal proof.",
        "- High loss-day gap means boundary forecast error tends to be larger on days "
        "where the champion loses money.",
        "- Intraday bucket analysis is especially relevant for storage because window "
        "selection is time-local.",
    ]
    (REPORTS_DIR / "boundary_error_impact.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    boundary_df = load_boundary_frame(cfg)
    champion_df = load_champion_daily()

    daily_df = daily_error_frame(boundary_df)
    merged = champion_df.merge(daily_df, on="date", how="inner")
    intraday_df = intraday_bucket_error_frame(boundary_df)

    corr_df = correlation_summary(merged)
    loss_df = loss_day_comparison(merged)
    month_df = month_summary(merged)
    worst_df = worst_days(merged)
    bucket_loss_df = bucket_loss_comparison(intraday_df, champion_df)

    merged.to_csv(REPORTS_DIR / "boundary_error_daily.csv", index=False)
    intraday_df.to_csv(REPORTS_DIR / "boundary_error_intraday.csv", index=False)
    corr_df.to_csv(REPORTS_DIR / "boundary_error_correlations.csv", index=False)
    loss_df.to_csv(REPORTS_DIR / "boundary_error_lossday_gap.csv", index=False)
    month_df.to_csv(REPORTS_DIR / "boundary_error_month_summary.csv", index=False)
    worst_df.to_csv(REPORTS_DIR / "boundary_error_worst_days.csv", index=False)
    bucket_loss_df.to_csv(REPORTS_DIR / "boundary_error_bucket_loss_gap.csv", index=False)
    write_markdown(
        month_df=month_df,
        corr_df=corr_df,
        loss_df=loss_df,
        worst_df=worst_df,
        bucket_loss_df=bucket_loss_df,
    )

    print(month_df.to_string(index=False))
    print(f"daily_path={REPORTS_DIR / 'boundary_error_daily.csv'}")
    print(f"intraday_path={REPORTS_DIR / 'boundary_error_intraday.csv'}")
    print(f"corr_path={REPORTS_DIR / 'boundary_error_correlations.csv'}")
    print(f"loss_gap_path={REPORTS_DIR / 'boundary_error_lossday_gap.csv'}")
    print(f"month_path={REPORTS_DIR / 'boundary_error_month_summary.csv'}")
    print(f"worst_path={REPORTS_DIR / 'boundary_error_worst_days.csv'}")
    print(f"bucket_loss_path={REPORTS_DIR / 'boundary_error_bucket_loss_gap.csv'}")
    print(f"markdown_path={REPORTS_DIR / 'boundary_error_impact.md'}")


if __name__ == "__main__":
    main()
