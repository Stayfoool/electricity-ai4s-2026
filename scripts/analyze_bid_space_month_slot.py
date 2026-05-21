from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity.data import load_train_frame
from electricity.features.bid_space import add_bid_space_columns, markdown_table

CONFIG_PATH = Path("configs/base.yaml")
REPORTS_DIR = Path("reports")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_frame() -> pd.DataFrame:
    cfg = load_config()
    df = add_bid_space_columns(load_train_frame(cfg))
    df["price_dev_day"] = df["price"] - df.groupby("date")["price"].transform("mean")
    df["bid_fct_dev_day"] = df["bid_space_fct"] - df.groupby("date")[
        "bid_space_fct"
    ].transform("mean")
    df["bid_act_dev_day"] = df["bid_space_act"] - df.groupby("date")[
        "bid_space_act"
    ].transform("mean")
    return df


def month_slot_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["month", "slot"], as_index=False)
    out = grouped.agg(
        rows=("price", "count"),
        price_mean=("price", "mean"),
        price_std=("price", "std"),
        bid_fct_mean=("bid_space_fct", "mean"),
        bid_act_mean=("bid_space_act", "mean"),
        bid_error_mean=("bid_space_error", "mean"),
        bid_error_mae=("bid_space_error", lambda s: s.abs().mean()),
        price_dev_mean=("price_dev_day", "mean"),
        bid_fct_dev_mean=("bid_fct_dev_day", "mean"),
        bid_act_dev_mean=("bid_act_dev_day", "mean"),
    )
    return out


def month_level_shape_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, group in df.groupby("month"):
        by_slot = month_slot_summary(group)
        rows.append(
            {
                "month": int(month),
                "days": int(group["date"].nunique()),
                "price_range_by_slot": by_slot["price_mean"].max()
                - by_slot["price_mean"].min(),
                "bid_fct_range_by_slot": by_slot["bid_fct_mean"].max()
                - by_slot["bid_fct_mean"].min(),
                "bid_act_range_by_slot": by_slot["bid_act_mean"].max()
                - by_slot["bid_act_mean"].min(),
                "slot_curve_corr_fct_price": by_slot["bid_fct_mean"].corr(
                    by_slot["price_mean"]
                ),
                "slot_curve_corr_act_price": by_slot["bid_act_mean"].corr(
                    by_slot["price_mean"]
                ),
                "slot_curve_spearman_fct_price": by_slot["bid_fct_mean"].corr(
                    by_slot["price_mean"], method="spearman"
                ),
                "slot_curve_spearman_act_price": by_slot["bid_act_mean"].corr(
                    by_slot["price_mean"], method="spearman"
                ),
                "mean_bid_error_mae": group["bid_space_error"].abs().mean(),
                "price_peak_slot": int(by_slot.loc[by_slot["price_mean"].idxmax(), "slot"]),
                "price_valley_slot": int(by_slot.loc[by_slot["price_mean"].idxmin(), "slot"]),
                "bid_fct_peak_slot": int(by_slot.loc[by_slot["bid_fct_mean"].idxmax(), "slot"]),
                "bid_fct_valley_slot": int(by_slot.loc[by_slot["bid_fct_mean"].idxmin(), "slot"]),
                "bid_act_peak_slot": int(by_slot.loc[by_slot["bid_act_mean"].idxmax(), "slot"]),
                "bid_act_valley_slot": int(by_slot.loc[by_slot["bid_act_mean"].idxmin(), "slot"]),
            }
        )
    return pd.DataFrame(rows)


def intraday_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 24, 40, 56, 72, 88, 96]
    labels = ["00_06", "06_10", "10_14", "14_18", "18_22", "22_24"]
    work = df.copy()
    work["segment"] = pd.cut(
        work["slot"], bins=bins, labels=labels, right=False, include_lowest=True
    )
    out = (
        work.groupby(["month", "segment"], observed=True)
        .agg(
            rows=("price", "count"),
            price_mean=("price", "mean"),
            price_std=("price", "std"),
            bid_fct_mean=("bid_space_fct", "mean"),
            bid_act_mean=("bid_space_act", "mean"),
            bid_error_mae=("bid_space_error", lambda s: s.abs().mean()),
            corr_fct_price=(
                "bid_space_fct",
                lambda s: s.corr(work.loc[s.index, "price"]),
            ),
            corr_act_price=(
                "bid_space_act",
                lambda s: s.corr(work.loc[s.index, "price"]),
            ),
            spearman_fct_price=(
                "bid_space_fct",
                lambda s: s.corr(work.loc[s.index, "price"], method="spearman"),
            ),
            spearman_act_price=(
                "bid_space_act",
                lambda s: s.corr(work.loc[s.index, "price"], method="spearman"),
            ),
        )
        .reset_index()
    )
    return out


def month_slot_corr_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, month_group in df.groupby("month"):
        for slot, group in month_group.groupby("slot"):
            if len(group) < 5:
                continue
            rows.append(
                {
                    "month": int(month),
                    "slot": int(slot),
                    "rows": len(group),
                    "corr_fct_price": group["bid_space_fct"].corr(group["price"]),
                    "corr_act_price": group["bid_space_act"].corr(group["price"]),
                    "spearman_fct_price": group["bid_space_fct"].corr(
                        group["price"], method="spearman"
                    ),
                    "spearman_act_price": group["bid_space_act"].corr(
                        group["price"], method="spearman"
                    ),
                    "price_std": group["price"].std(),
                    "bid_error_mae": group["bid_space_error"].abs().mean(),
                }
            )
    return pd.DataFrame(rows)


def _pivot(summary: pd.DataFrame, value: str) -> pd.DataFrame:
    return summary.pivot(index="month", columns="slot", values=value).sort_index()


def plot_heatmaps(summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plots = [
        (summary, "price_mean", "Monthly intraday actual price mean"),
        (summary, "bid_fct_mean", "Monthly intraday forecast bid space mean"),
        (summary, "bid_act_mean", "Monthly intraday actual bid space mean"),
        (summary, "bid_error_mean", "Monthly intraday bid space error"),
        (corr, "spearman_fct_price", "Month x slot Spearman: forecast bid space vs price"),
        (corr, "spearman_act_price", "Month x slot Spearman: actual bid space vs price"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
    for ax, (frame, value, title) in zip(axes.ravel(), plots, strict=True):
        data = _pivot(frame, value)
        im = ax.imshow(data.to_numpy(), aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_ylabel("month")
        ax.set_xlabel("slot")
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_yticklabels(data.index)
        ax.set_xticks(np.arange(0, 96, 12))
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(REPORTS_DIR / "bid_space_month_slot_heatmaps.png", dpi=160)
    plt.close(fig)


def plot_monthly_curves(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    months = sorted(summary["month"].unique())
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True, constrained_layout=True)
    first_twin = None
    for ax, month in zip(axes.ravel(), months, strict=False):
        group = summary[summary["month"] == month].sort_values("slot")
        ax.plot(group["slot"], group["price_mean"], label="price", color="black")
        ax2 = ax.twinx()
        if first_twin is None:
            first_twin = ax2
        ax2.plot(group["slot"], group["bid_fct_mean"], label="bid_fct", color="tab:blue")
        ax2.plot(group["slot"], group["bid_act_mean"], label="bid_act", color="tab:orange")
        ax.set_title(f"month {month:02d}")
        ax.set_xticks(np.arange(0, 96, 24))
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="upper left")
    if first_twin is not None:
        first_twin.legend(loc="upper right")
    fig.savefig(REPORTS_DIR / "bid_space_monthly_intraday_curves.png", dpi=160)
    plt.close(fig)


def write_report(
    *,
    shape_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    corr_summary: pd.DataFrame,
) -> None:
    weak_months = shape_summary.sort_values("slot_curve_spearman_fct_price").head(6)
    weak_segments = segment_summary.sort_values("spearman_fct_price").head(15)
    volatile_slots = corr_summary.sort_values("price_std", ascending=False).head(15)
    lines = [
        "# Bid Space Month-Slot Diagnostics",
        "",
        "This report focuses on monthly intraday shape: month x 96-slot bid space, ",
        "actual price, forecast error, and bid-space/price correlation.",
        "",
        "## Monthly Shape Summary",
        "",
        markdown_table(shape_summary, floatfmt=".4f"),
        "",
        "## Weakest Months By Forecast Bid-Space Intraday Shape",
        "",
        markdown_table(weak_months, floatfmt=".4f"),
        "",
        "## Weakest Month-Segment Correlations",
        "",
        markdown_table(weak_segments, floatfmt=".4f"),
        "",
        "## Most Volatile Month-Slot Points",
        "",
        markdown_table(volatile_slots, floatfmt=".4f"),
        "",
        "## Readout",
        "",
        "- Use this report to locate months and day segments where bid_space is unreliable.",
        "- Weak forecast-bid-space segments are candidates for reranker caution features.",
        "- If actual bid_space is also weak in a segment, the missing signal is not only "
        "forecast error.",
        "",
        "## Artifacts",
        "",
        "- `reports/bid_space_month_slot_summary.csv`",
        "- `reports/bid_space_month_level_shape_summary.csv`",
        "- `reports/bid_space_month_segment_summary.csv`",
        "- `reports/bid_space_month_slot_corr.csv`",
        "- `reports/bid_space_month_slot_heatmaps.png`",
        "- `reports/bid_space_monthly_intraday_curves.png`",
    ]
    (REPORTS_DIR / "bid_space_month_slot_diagnostics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load_frame()
    summary = month_slot_summary(df)
    shape_summary = month_level_shape_summary(df)
    segment_summary = intraday_segment_summary(df)
    corr_summary = month_slot_corr_summary(df)

    summary.to_csv(REPORTS_DIR / "bid_space_month_slot_summary.csv", index=False)
    shape_summary.to_csv(REPORTS_DIR / "bid_space_month_level_shape_summary.csv", index=False)
    segment_summary.to_csv(REPORTS_DIR / "bid_space_month_segment_summary.csv", index=False)
    corr_summary.to_csv(REPORTS_DIR / "bid_space_month_slot_corr.csv", index=False)
    plot_heatmaps(summary, corr_summary)
    plot_monthly_curves(summary)
    write_report(
        shape_summary=shape_summary,
        segment_summary=segment_summary,
        corr_summary=corr_summary,
    )
    print(shape_summary.to_string(index=False))
    print(REPORTS_DIR / "bid_space_month_slot_diagnostics.md")


if __name__ == "__main__":
    main()
