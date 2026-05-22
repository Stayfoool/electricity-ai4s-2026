from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path("reports")

REGIMES = {
    "standard_09_12": {
        "folds": {"valid_2025_09", "valid_2025_10", "valid_2025_11", "valid_2025_12"}
    },
    "jan_feb_like": {"folds": {"valid_2025_jan_feb"}},
    "winter_11_12_jan_feb": {
        "folds": {"valid_2025_11", "valid_2025_12", "valid_2025_jan_feb"}
    },
    "late_winter_12_jan_feb": {"folds": {"valid_2025_12", "valid_2025_jan_feb"}},
    "all_5fold": {"folds": None},
}


def model_name_from_path(path: Path) -> str:
    name = path.name
    if not name.startswith("backtest_") or not name.endswith("_daily.csv"):
        raise ValueError(f"expected backtest_*_daily.csv, got {path}")
    return name.removeprefix("backtest_").removesuffix("_daily.csv")


def summarize_daily(daily: pd.DataFrame, *, model: str, regime: str) -> dict[str, object]:
    if daily.empty:
        return {
            "model": model,
            "regime": regime,
            "days": 0,
            "mean_profit": float("nan"),
            "worst_day_profit": float("nan"),
            "loss_days": 0,
            "mean_oracle_profit": float("nan"),
            "oracle_ratio": float("nan"),
            "mean_regret": float("nan"),
            "p90_regret": float("nan"),
            "mean_abs_charge_gap": float("nan"),
            "mean_abs_discharge_gap": float("nan"),
            "trade_days": 0,
        }
    out = {
        "model": model,
        "regime": regime,
        "days": int(len(daily)),
        "mean_profit": float(daily["profit"].mean()),
        "worst_day_profit": float(daily["profit"].min()),
        "loss_days": int((daily["profit"] < -1e-9).sum()),
        "mean_oracle_profit": float(daily["oracle_profit"].mean()),
        "oracle_ratio": float(daily["profit"].mean() / daily["oracle_profit"].mean()),
        "mean_regret": float(daily["regret"].mean()),
        "p90_regret": float(daily["regret"].quantile(0.9)),
        "mean_abs_charge_gap": float(daily["charge_start_gap"].abs().mean()),
        "mean_abs_discharge_gap": float(daily["discharge_start_gap"].abs().mean()),
        "trade_days": int(daily["traded"].sum()) if "traded" in daily else int(len(daily)),
    }
    return out


def summarize_file(path: Path) -> list[dict[str, object]]:
    model = model_name_from_path(path)
    daily = pd.read_csv(path)
    if "fold" not in daily.columns:
        raise ValueError(f"daily report missing fold column: {path}")
    rows = []
    for regime, spec in REGIMES.items():
        folds = spec["folds"]
        subset = daily if folds is None else daily[daily["fold"].isin(folds)]
        rows.append(summarize_daily(subset, model=model, regime=regime))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="backtest_*5fold*_daily.csv",
        help="glob pattern under reports/ for daily reports",
    )
    parser.add_argument(
        "--output",
        default="reports/validation_regime_summary.csv",
        help="output CSV path",
    )
    parser.add_argument(
        "--markdown",
        default="reports/validation_regime_summary.md",
        help="output Markdown path",
    )
    args = parser.parse_args()

    paths = sorted(REPORTS_DIR.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"no reports matched {args.pattern}")

    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(summarize_file(path))
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["regime", "mean_profit"], ascending=[True, False])

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    lines = [
        "# Validation Regime Summary",
        "",
        "This report re-summarizes daily backtest outputs under multiple validation regimes.",
        "It is intended to reduce overfitting to a single trailing validation split.",
        "",
        "## Regimes",
        "",
        "- `standard_09_12`: rolling validation on September to December 2025.",
        "- `jan_feb_like`: pseudo-test fold using January-February 2025.",
        "- `winter_11_12_jan_feb`: winter-focused view combining November, December, "
        "and Jan-Feb-like.",
        "- `late_winter_12_jan_feb`: stricter late-winter view combining December "
        "and Jan-Feb-like.",
        "- `all_5fold`: all available 5-fold validation days.",
        "",
    ]
    display_cols = [
        "model",
        "days",
        "mean_profit",
        "worst_day_profit",
        "loss_days",
        "oracle_ratio",
        "mean_regret",
        "p90_regret",
        "mean_abs_charge_gap",
        "mean_abs_discharge_gap",
    ]
    def markdown_table(df: pd.DataFrame) -> str:
        headers = list(df.columns)
        out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            cells = []
            for col in headers:
                value = row[col]
                if isinstance(value, float):
                    cells.append(f"{value:.4f}")
                else:
                    cells.append(str(value))
            out.append("| " + " | ".join(cells) + " |")
        return "\n".join(out)

    for regime in REGIMES:
        block = summary[summary["regime"] == regime][display_cols].copy()
        lines.extend([f"## {regime}", "", markdown_table(block), ""])
    lines.extend(["## Artifacts", "", f"- `{out_csv}`"])
    out_md = Path(args.markdown)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_path={out_csv}")
    print(f"markdown_path={out_md}")


if __name__ == "__main__":
    main()
