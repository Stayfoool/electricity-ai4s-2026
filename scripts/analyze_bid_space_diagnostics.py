from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity.data import load_train_frame
from electricity.dispatch import optimize_day
from electricity.eval.metrics import daily_profit
from electricity.features.bid_space import add_bid_space_columns, markdown_table

CONFIG_PATH = Path("configs/base.yaml")
REPORTS_DIR = Path("reports")
BLOCK_SIZE = 8


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    frame = pd.concat([x, y], axis=1).dropna()
    if len(frame) < 3 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1], method=method))


def point_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, group in [("all", df), *[(f"month_{m:02d}", g) for m, g in df.groupby("month")]]:
        rows.append(
            {
                "scope": scope,
                "rows": len(group),
                "bid_fct_mean": group["bid_space_fct"].mean(),
                "bid_act_mean": group["bid_space_act"].mean(),
                "bid_error_bias": group["bid_space_error"].mean(),
                "bid_error_mae": group["bid_space_error"].abs().mean(),
                "bid_error_rmse": np.sqrt(np.mean(group["bid_space_error"] ** 2)),
                "corr_fct_price": _safe_corr(group["bid_space_fct"], group["price"]),
                "corr_act_price": _safe_corr(group["bid_space_act"], group["price"]),
                "spearman_fct_price": _safe_corr(
                    group["bid_space_fct"], group["price"], method="spearman"
                ),
                "spearman_act_price": _safe_corr(
                    group["bid_space_act"], group["price"], method="spearman"
                ),
                "corr_fct_act": _safe_corr(group["bid_space_fct"], group["bid_space_act"]),
                "spearman_fct_act": _safe_corr(
                    group["bid_space_fct"], group["bid_space_act"], method="spearman"
                ),
            }
        )
    return pd.DataFrame(rows)


def price_bin_stats(df: pd.DataFrame, *, bins: int = 20) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for col in ["bid_space_fct", "bid_space_act"]:
        bucket = pd.qcut(df[col], q=bins, duplicates="drop")
        stat = (
            df.groupby(bucket, observed=True)
            .agg(
                bid_mean=(col, "mean"),
                price_mean=("price", "mean"),
                price_std=("price", "std"),
                count=("price", "count"),
            )
            .reset_index(drop=True)
        )
        stat.insert(0, "source", col)
        stat.insert(1, "bin_id", np.arange(len(stat)))
        rows.append(stat)
    return pd.concat(rows, ignore_index=True)


def _window_means(values: np.ndarray) -> np.ndarray:
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    return (prefix[BLOCK_SIZE:] - prefix[:-BLOCK_SIZE]) / BLOCK_SIZE


def _pair_profit_from_starts(prices: np.ndarray, charge_start: int, discharge_start: int) -> float:
    power = np.zeros(96, dtype=float)
    power[charge_start : charge_start + BLOCK_SIZE] = -1000.0
    power[discharge_start : discharge_start + BLOCK_SIZE] = 1000.0
    return daily_profit(prices, power)


def window_and_daily_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_rows: list[dict[str, object]] = []
    day_rows: list[dict[str, object]] = []

    for date, group in df.groupby("date", sort=True):
        if len(group) != 96:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            continue

        price = group["price"].to_numpy(dtype=float)
        bid_fct = group["bid_space_fct"].to_numpy(dtype=float)
        bid_act = group["bid_space_act"].to_numpy(dtype=float)
        price_win = _window_means(price)
        fct_win = _window_means(bid_fct)
        act_win = _window_means(bid_act)

        for start in range(len(price_win)):
            window_rows.append(
                {
                    "date": date,
                    "month": int(group["month"].iloc[0]),
                    "window_start": start,
                    "price_window_mean": price_win[start],
                    "bid_fct_window_mean": fct_win[start],
                    "bid_act_window_mean": act_win[start],
                    "bid_window_error": fct_win[start] - act_win[start],
                }
            )

        oracle = optimize_day(price)
        # For 89 window means, rank legal pair starts directly.
        fct_pair = rank_window_pairs(fct_win)[0]
        act_pair = rank_window_pairs(act_win)[0]
        price_pair = rank_window_pairs(price_win)[0]
        fct_profit = _pair_profit_from_starts(price, fct_pair[1], fct_pair[2])
        act_profit = _pair_profit_from_starts(price, act_pair[1], act_pair[2])
        win_oracle_profit = _pair_profit_from_starts(price, price_pair[1], price_pair[2])
        oracle_profit = daily_profit(price, oracle.power)

        day_rows.append(
            {
                "date": date,
                "month": int(group["month"].iloc[0]),
                "bid_point_spearman": _safe_corr(
                    pd.Series(bid_fct), pd.Series(bid_act), method="spearman"
                ),
                "bid_price_spearman_fct": _safe_corr(
                    pd.Series(bid_fct), pd.Series(price), method="spearman"
                ),
                "bid_price_spearman_act": _safe_corr(
                    pd.Series(bid_act), pd.Series(price), method="spearman"
                ),
                "window_spearman_fct_price": _safe_corr(
                    pd.Series(fct_win), pd.Series(price_win), method="spearman"
                ),
                "window_spearman_act_price": _safe_corr(
                    pd.Series(act_win), pd.Series(price_win), method="spearman"
                ),
                "bid_error_mae_day": float(np.mean(np.abs(bid_fct - bid_act))),
                "bid_window_error_mae_day": float(np.mean(np.abs(fct_win - act_win))),
                "oracle_charge_start": oracle.charge_start,
                "oracle_discharge_start": oracle.discharge_start,
                "price_window_charge_start": price_pair[1],
                "price_window_discharge_start": price_pair[2],
                "bid_fct_charge_start": fct_pair[1],
                "bid_fct_discharge_start": fct_pair[2],
                "bid_act_charge_start": act_pair[1],
                "bid_act_discharge_start": act_pair[2],
                "fct_charge_gap": fct_pair[1] - price_pair[1],
                "fct_discharge_gap": fct_pair[2] - price_pair[2],
                "act_charge_gap": act_pair[1] - price_pair[1],
                "act_discharge_gap": act_pair[2] - price_pair[2],
                "profit_by_bid_fct_pair": fct_profit,
                "profit_by_bid_act_pair": act_profit,
                "window_oracle_profit": win_oracle_profit,
                "oracle_profit": oracle_profit,
                "regret_bid_fct_pair": win_oracle_profit - fct_profit,
                "regret_bid_act_pair": win_oracle_profit - act_profit,
                "bid_fct_spread": fct_pair[0],
                "bid_act_spread": act_pair[0],
                "price_window_spread": price_pair[0],
            }
        )

    return pd.DataFrame(window_rows), pd.DataFrame(day_rows)


def rank_window_pairs(window_values: np.ndarray) -> list[tuple[float, int, int]]:
    """Rank legal charge/discharge 8-slot-window pairs using 89 window values."""
    if len(window_values) != 89:
        raise ValueError(f"expected 89 window values, got {len(window_values)}")
    out: list[tuple[float, int, int]] = []
    for tc in range(0, 81):
        for td in range(tc + BLOCK_SIZE, 89):
            out.append((float(window_values[td] - window_values[tc]), tc, td))
    out.sort(key=lambda item: item[0], reverse=True)
    return out


def summarize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = [("all", daily), *[(f"month_{m:02d}", g) for m, g in daily.groupby("month")]]
    for scope, group in scopes:
        rows.append(
            {
                "scope": scope,
                "days": len(group),
                "window_spearman_fct_price_mean": group["window_spearman_fct_price"].mean(),
                "window_spearman_act_price_mean": group["window_spearman_act_price"].mean(),
                "bid_window_error_mae_day_mean": group["bid_window_error_mae_day"].mean(),
                "profit_by_bid_fct_pair_mean": group["profit_by_bid_fct_pair"].mean(),
                "profit_by_bid_act_pair_mean": group["profit_by_bid_act_pair"].mean(),
                "window_oracle_profit_mean": group["window_oracle_profit"].mean(),
                "regret_bid_fct_pair_mean": group["regret_bid_fct_pair"].mean(),
                "regret_bid_act_pair_mean": group["regret_bid_act_pair"].mean(),
                "mean_abs_fct_charge_gap": group["fct_charge_gap"].abs().mean(),
                "mean_abs_fct_discharge_gap": group["fct_discharge_gap"].abs().mean(),
                "mean_abs_act_charge_gap": group["act_charge_gap"].abs().mean(),
                "mean_abs_act_discharge_gap": group["act_discharge_gap"].abs().mean(),
                "loss_days_bid_fct_pair": int((group["profit_by_bid_fct_pair"] < 0).sum()),
                "loss_days_bid_act_pair": int((group["profit_by_bid_act_pair"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def make_plots(point: pd.DataFrame, bins: pd.DataFrame, daily: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    REPORTS_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for source, group in bins.groupby("source"):
        ax.plot(group["bid_mean"], group["price_mean"], marker="o", label=source)
    ax.set_xlabel("bid space bin mean")
    ax.set_ylabel("actual price mean")
    ax.set_title("Actual price vs forecast/actual bid space bins")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "bid_space_vs_price_bins.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    by_hour = point.groupby("hour")["bid_space_error"].agg(["mean", "std"])
    ax.plot(by_hour.index, by_hour["mean"], marker="o")
    ax.fill_between(
        by_hour.index,
        by_hour["mean"] - by_hour["std"],
        by_hour["mean"] + by_hour["std"],
        alpha=0.2,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("hour")
    ax.set_ylabel("forecast - actual bid space")
    ax.set_title("Bid space forecast error by hour")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "bid_space_error_by_hour.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        daily["window_spearman_fct_price"],
        daily["profit_by_bid_fct_pair"],
        s=12,
        alpha=0.55,
    )
    ax.set_xlabel("daily Spearman: forecast bid-space window vs price window")
    ax.set_ylabel("profit from forecast bid-space pair")
    ax.set_title("Window ranking quality vs realized profit")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "bid_space_window_rank_vs_profit.png", dpi=160)
    plt.close(fig)



def write_markdown(
    *,
    point_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    top_regret = daily.sort_values("regret_bid_fct_pair", ascending=False).head(15)
    lines = [
        "# Bid Space Diagnostics",
        "",
        "## Scope",
        "",
        "This report uses labelled 2025 training rows only. Actual boundary values "
        "and actual prices are used for diagnostics, not as submit-time features.",
        "",
        "Definitions:",
        "",
        "- `bid_space_fct = 系统负荷预测值 - 风光总加预测值 - 联络线预测值 "
        "- 水电预测值 - 非市场化机组预测值`.",
        "- `bid_space_act = 系统负荷实际值 - 风光总加实际值 - 联络线实际值 "
        "- 水电实际值 - 非市场化机组实际值`.",
        "- Window metrics use 8-slot / 2-hour rolling means, matching the storage "
        "block constraint.",
        "",
        "## Point-Level Summary",
        "",
        markdown_table(point_summary, floatfmt=".4f"),
        "",
        "## Window / Daily Dispatch Summary",
        "",
        markdown_table(daily_summary, floatfmt=".4f"),
        "",
        "## Worst Regret Days When Selecting Pair By Forecast Bid Space",
        "",
        markdown_table(
            top_regret[
                [
                    "date",
                    "month",
                    "window_spearman_fct_price",
                    "window_spearman_act_price",
                    "bid_window_error_mae_day",
                    "bid_fct_charge_start",
                    "bid_fct_discharge_start",
                    "price_window_charge_start",
                    "price_window_discharge_start",
                    "profit_by_bid_fct_pair",
                    "window_oracle_profit",
                    "regret_bid_fct_pair",
                ]
            ],
            floatfmt=".4f",
        ),
        "",
        "## Interpretation",
        "",
        "- `bid_space_act` measures the business signal upper bound: if it tracks "
        "price much better than `bid_space_fct`, forecast error is a major bottleneck.",
        "- The window summary is more important than point correlation because "
        "the competition payoff is determined by two 8-slot windows.",
        "- If `profit_by_bid_fct_pair` is far below `window_oracle_profit`, "
        "directly dispatching from bid space is weaker than price-model dispatch "
        "even if bid space is directionally correlated with price.",
        "- If `bid_space_act` pair is also weak, the issue is not just forecast "
        "error; bid space alone misses price drivers.",
        "",
        "## Artifacts",
        "",
        "- `reports/bid_space_point_stats.csv`",
        "- `reports/bid_space_price_bins.csv`",
        "- `reports/bid_space_window_stats.csv`",
        "- `reports/bid_space_daily_dispatch.csv`",
        "- `reports/bid_space_daily_summary.csv`",
        "- `reports/bid_space_vs_price_bins.png`",
        "- `reports/bid_space_error_by_hour.png`",
        "- `reports/bid_space_window_rank_vs_profit.png`",
    ]
    (REPORTS_DIR / "bid_space_diagnostics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    df = add_bid_space_columns(load_train_frame(cfg))
    point_summary = point_stats(df)
    bins = price_bin_stats(df)
    windows, daily = window_and_daily_frames(df)
    daily_summary = summarize_daily(daily)

    point_summary.to_csv(REPORTS_DIR / "bid_space_point_stats.csv", index=False)
    bins.to_csv(REPORTS_DIR / "bid_space_price_bins.csv", index=False)
    windows.to_csv(REPORTS_DIR / "bid_space_window_stats.csv", index=False)
    daily.to_csv(REPORTS_DIR / "bid_space_daily_dispatch.csv", index=False)
    daily_summary.to_csv(REPORTS_DIR / "bid_space_daily_summary.csv", index=False)
    make_plots(df, bins, daily)
    write_markdown(point_summary=point_summary, daily_summary=daily_summary, daily=daily)

    print(point_summary.head(13).to_string(index=False))
    print(daily_summary.head(13).to_string(index=False))
    print("wrote reports/bid_space_diagnostics.md")


if __name__ == "__main__":
    main()
