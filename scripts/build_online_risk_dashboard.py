from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

RAW_FEATURE_COLS = [
    "系统负荷预测值",
    "风光总加预测值",
    "联络线预测值",
    "风电预测值",
    "光伏预测值",
    "水电预测值",
    "非市场化机组预测值",
]

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

ONLINE_MODEL_ALIASES = {
    "champion": "ens_champion_segmented6_prior_5fold",
    "holiday_only": "ens_champion_segmented6_prior_5fold_holiday_only",
}


def model_name_from_daily_path(path: Path) -> str:
    name = path.name
    if not name.startswith("backtest_") or not name.endswith("_daily.csv"):
        raise ValueError(f"expected backtest_*_daily.csv, got {path}")
    return name.removeprefix("backtest_").removesuffix("_daily.csv")


def read_csv_from_zip(zip_path: Path, member: str, *, parse_dates: list[str]) -> pd.DataFrame:
    with ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            return pd.read_csv(f, parse_dates=parse_dates)


def load_boundary_frames(material_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    zip_path = material_dir / "to_sais_new.zip"
    if zip_path.exists():
        train = read_csv_from_zip(
            zip_path,
            "to_sais_new/train/mengxi_boundary_anon_filtered.csv",
            parse_dates=["times"],
        )
        test = read_csv_from_zip(
            zip_path,
            "to_sais_new/test/test_in_feature_ori.csv",
            parse_dates=["times"],
        )
        return train, test

    train_path = material_dir / "to_sais_new/train/mengxi_boundary_anon_filtered.csv"
    test_path = material_dir / "to_sais_new/test/test_in_feature_ori.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "cannot find competition boundary data under "
            f"{material_dir}; expected to_sais_new.zip or unzipped CSVs"
        )
    return (
        pd.read_csv(train_path, parse_dates=["times"]),
        pd.read_csv(test_path, parse_dates=["times"]),
    )


def add_bid_space(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bid_space"] = (
        out["系统负荷预测值"]
        - out["风光总加预测值"]
        - out["联络线预测值"]
        - out["水电预测值"]
        - out["非市场化机组预测值"]
    )
    return out


def daily_signature(df: pd.DataFrame) -> pd.DataFrame:
    work = add_bid_space(df)
    work["date"] = work["times"].dt.normalize()
    work["month"] = work["times"].dt.month
    work["dayofweek"] = work["times"].dt.dayofweek

    value_cols = [*RAW_FEATURE_COLS, "bid_space"]
    agg = work.groupby("date", as_index=False)[value_cols].agg(["mean", "std", "min", "max"])
    agg.columns = ["date", *[f"{col}_{stat}" for col, stat in agg.columns[1:]]]

    cal = (
        work.groupby("date", as_index=False)
        .agg(month=("month", "first"), dayofweek=("dayofweek", "first"))
        .assign(
            is_weekend=lambda d: (d["dayofweek"] >= 5).astype(int),
            month_sin=lambda d: np.sin(2 * np.pi * d["month"] / 12),
            month_cos=lambda d: np.cos(2 * np.pi * d["month"] / 12),
            dow_sin=lambda d: np.sin(2 * np.pi * d["dayofweek"] / 7),
            dow_cos=lambda d: np.cos(2 * np.pi * d["dayofweek"] / 7),
        )
    )
    out = agg.merge(
        cal[["date", "month_sin", "month_cos", "dow_sin", "dow_cos", "is_weekend"]],
        on="date",
        how="left",
    )
    out["date"] = pd.to_datetime(out["date"])
    return out


def standardized_nearest_test_distance(
    train_sig: pd.DataFrame,
    test_sig: pd.DataFrame,
) -> pd.DataFrame:
    feature_cols = [c for c in train_sig.columns if c != "date"]
    combined = pd.concat([train_sig[feature_cols], test_sig[feature_cols]], ignore_index=True)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0).replace(0.0, 1.0).fillna(1.0)

    train_x = ((train_sig[feature_cols] - mean) / std).to_numpy(dtype=float)
    test_x = ((test_sig[feature_cols] - mean) / std).to_numpy(dtype=float)

    # Small matrices: 365 train days x 59 test days. Vectorized distance is safe.
    diff = train_x[:, None, :] - test_x[None, :, :]
    dist = np.sqrt(np.mean(diff * diff, axis=2))
    nearest_idx = dist.argmin(axis=1)
    out = train_sig[["date"]].copy()
    out["nearest_test_date"] = test_sig.iloc[nearest_idx]["date"].to_numpy()
    out["test_like_distance"] = dist.min(axis=1)

    scale = float(np.median(out["test_like_distance"]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    out["test_like_weight_raw"] = np.exp(-out["test_like_distance"] / scale)
    return out


def load_daily_reports(reports_dir: Path, pattern: str) -> dict[str, pd.DataFrame]:
    reports: dict[str, pd.DataFrame] = {}
    for path in sorted(reports_dir.glob(pattern)):
        model = model_name_from_daily_path(path)
        daily = pd.read_csv(path, parse_dates=["date"])
        if "fold" not in daily.columns:
            continue
        reports[model] = daily
    if not reports:
        raise FileNotFoundError(f"no daily reports matched {reports_dir / pattern}")
    return reports


def summarize_daily(daily: pd.DataFrame, *, model: str, regime: str) -> dict[str, object]:
    if daily.empty:
        return {
            "model": model,
            "regime": regime,
            "days": 0,
            "mean_profit": np.nan,
            "p10_profit": np.nan,
            "worst_day_profit": np.nan,
            "loss_days": 0,
            "oracle_ratio": np.nan,
            "mean_regret": np.nan,
            "p90_regret": np.nan,
            "mean_abs_charge_gap": np.nan,
            "mean_abs_discharge_gap": np.nan,
        }
    return {
        "model": model,
        "regime": regime,
        "days": int(len(daily)),
        "mean_profit": float(daily["profit"].mean()),
        "p10_profit": float(daily["profit"].quantile(0.1)),
        "worst_day_profit": float(daily["profit"].min()),
        "loss_days": int((daily["profit"] < -1e-9).sum()),
        "oracle_ratio": float(daily["profit"].sum() / daily["oracle_profit"].sum()),
        "mean_regret": float(daily["regret"].mean()),
        "p90_regret": float(daily["regret"].quantile(0.9)),
        "mean_abs_charge_gap": float(daily["charge_start_gap"].abs().mean()),
        "mean_abs_discharge_gap": float(daily["discharge_start_gap"].abs().mean()),
    }


def summarize_regimes(reports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, daily in reports.items():
        for regime, spec in REGIMES.items():
            folds = spec["folds"]
            subset = daily if folds is None else daily[daily["fold"].isin(folds)]
            rows.append(summarize_daily(subset, model=model, regime=regime))
    return pd.DataFrame(rows)


def summarize_test_like(
    reports: dict[str, pd.DataFrame],
    day_weights: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    weights = day_weights[["date", "test_like_distance", "test_like_weight_raw"]].copy()
    for model, daily in reports.items():
        merged = daily.merge(weights, on="date", how="left")
        merged["test_like_weight_raw"] = merged["test_like_weight_raw"].fillna(0.0)
        weight_sum = float(merged["test_like_weight_raw"].sum())
        if weight_sum <= 0:
            weighted_profit = np.nan
            weighted_regret = np.nan
            weighted_oracle = np.nan
            effective_days = np.nan
            weighted_loss_days = np.nan
        else:
            w = merged["test_like_weight_raw"].to_numpy(dtype=float)
            weighted_profit = float(np.average(merged["profit"], weights=w))
            weighted_regret = float(np.average(merged["regret"], weights=w))
            weighted_oracle = float(np.average(merged["oracle_profit"], weights=w))
            effective_days = float((w.sum() ** 2) / np.sum(w * w))
            weighted_loss_days = float(
                np.average((merged["profit"] < -1e-9).astype(float), weights=w)
            )
        rows.append(
            {
                "model": model,
                "test_like_days": int(len(merged)),
                "test_like_effective_days": effective_days,
                "test_like_mean_distance": float(merged["test_like_distance"].mean()),
                "test_like_weighted_profit": weighted_profit,
                "test_like_weighted_oracle_ratio": (
                    weighted_profit / weighted_oracle
                    if weighted_oracle and weighted_oracle > 0
                    else np.nan
                ),
                "test_like_weighted_regret": weighted_regret,
                "test_like_weighted_loss_rate": weighted_loss_days,
            }
        )
    return pd.DataFrame(rows)


def summarize_feature_drift(day_weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, spec in REGIMES.items():
        folds = spec["folds"]
        if folds is None:
            subset = day_weights
        else:
            month_map = {
                "valid_2025_09": 9,
                "valid_2025_10": 10,
                "valid_2025_11": 11,
                "valid_2025_12": 12,
                "valid_2025_jan_feb": None,
            }
            masks = []
            for fold in folds:
                month = month_map.get(fold)
                if month is None:
                    masks.append(day_weights["date"].dt.month.isin([1, 2]))
                else:
                    masks.append(day_weights["date"].dt.month == month)
            mask = masks[0].copy()
            for item in masks[1:]:
                mask = mask | item
            subset = day_weights[mask]
        rows.append(
            {
                "regime": regime,
                "days": int(len(subset)),
                "mean_test_like_distance": float(subset["test_like_distance"].mean()),
                "median_test_like_distance": float(subset["test_like_distance"].median()),
                "mean_weight_raw": float(subset["test_like_weight_raw"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_test_like_distance")


def load_online_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["submitted_at", "model", "dashboard_model", "online_score", "notes"]
        )
    scores = pd.read_csv(path)
    scores["dashboard_model"] = scores["model"].map(ONLINE_MODEL_ALIASES)
    missing = scores["dashboard_model"].isna()
    scores.loc[missing, "dashboard_model"] = scores.loc[missing, "model"]
    return scores


def wide_regime_summary(regime_summary: pd.DataFrame) -> pd.DataFrame:
    keep_metrics = [
        "mean_profit",
        "p10_profit",
        "worst_day_profit",
        "loss_days",
        "oracle_ratio",
        "mean_regret",
        "p90_regret",
        "mean_abs_charge_gap",
        "mean_abs_discharge_gap",
    ]
    pieces = []
    for regime in REGIMES:
        block = regime_summary[regime_summary["regime"] == regime][["model", *keep_metrics]].copy()
        block = block.rename(columns={col: f"{regime}_{col}" for col in keep_metrics})
        pieces.append(block)
    out = pieces[0]
    for block in pieces[1:]:
        out = out.merge(block, on="model", how="outer")
    return out


def add_champion_deltas(dashboard: pd.DataFrame, champion_model: str) -> pd.DataFrame:
    out = dashboard.copy()
    base_rows = out[out["model"] == champion_model]
    if base_rows.empty:
        return out
    base = base_rows.iloc[0]
    delta_cols = [
        "standard_09_12_mean_profit",
        "jan_feb_like_mean_profit",
        "winter_11_12_jan_feb_mean_profit",
        "late_winter_12_jan_feb_mean_profit",
        "all_5fold_mean_profit",
        "test_like_weighted_profit",
        "all_5fold_loss_days",
        "all_5fold_p90_regret",
        "all_5fold_worst_day_profit",
    ]
    for col in delta_cols:
        if col in out.columns and col in base:
            out[f"delta_vs_champion__{col}"] = out[col] - base[col]
    if "online_score" in out.columns and pd.notna(base.get("online_score", np.nan)):
        out["delta_vs_champion__online_score"] = out["online_score"] - base["online_score"]
    return out


def classify_risk(row: pd.Series) -> str:
    all_delta = row.get("delta_vs_champion__all_5fold_mean_profit", np.nan)
    jan_delta = row.get("delta_vs_champion__jan_feb_like_mean_profit", np.nan)
    standard_delta = row.get("delta_vs_champion__standard_09_12_mean_profit", np.nan)
    loss_delta = row.get("delta_vs_champion__all_5fold_loss_days", np.nan)
    test_like_delta = row.get("delta_vs_champion__test_like_weighted_profit", np.nan)

    if row.get("model") == "ens_champion_segmented6_prior_5fold":
        return "baseline_champion"
    if (
        pd.notna(all_delta)
        and all_delta >= 100
        and pd.notna(test_like_delta)
        and test_like_delta >= 0
    ):
        return "submit_candidate"
    if (
        pd.notna(jan_delta)
        and jan_delta > 100
        and pd.notna(standard_delta)
        and standard_delta < -200
    ):
        return "winter_overfit_risk"
    if pd.notna(loss_delta) and loss_delta > 1:
        return "loss_day_risk"
    if pd.notna(all_delta) and all_delta < -250:
        return "broad_validation_weak"
    return "watch"


def markdown_table(df: pd.DataFrame, *, max_rows: int = 20, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "(empty)"
    show = df.head(max_rows)
    headers = list(show.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in show.iterrows():
        cells = []
        for col in headers:
            value = row[col]
            if isinstance(value, float) or isinstance(value, np.floating):
                if np.isnan(value):
                    cells.append("nan")
                else:
                    cells.append(format(float(value), floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    *,
    path: Path,
    dashboard: pd.DataFrame,
    feature_drift: pd.DataFrame,
    online_scores: pd.DataFrame,
) -> None:
    display_cols = [
        "model",
        "risk_class",
        "online_score",
        "delta_vs_champion__online_score",
        "all_5fold_mean_profit",
        "delta_vs_champion__all_5fold_mean_profit",
        "jan_feb_like_mean_profit",
        "delta_vs_champion__jan_feb_like_mean_profit",
        "standard_09_12_mean_profit",
        "delta_vs_champion__standard_09_12_mean_profit",
        "test_like_weighted_profit",
        "delta_vs_champion__test_like_weighted_profit",
        "all_5fold_loss_days",
        "all_5fold_p90_regret",
    ]
    display_cols = [col for col in display_cols if col in dashboard.columns]
    top = dashboard[display_cols].copy()
    top = top.sort_values(
        ["online_score", "all_5fold_mean_profit", "test_like_weighted_profit"],
        ascending=[False, False, False],
        na_position="last",
    )

    lines = [
        "# Online Risk Dashboard",
        "",
        "Purpose: explain why local validation can diverge from online score and avoid "
        "submitting candidates that only win one validation regime.",
        "",
        "## Key Readout",
        "",
        "- `test_like_weighted_profit` uses only visible 2026 Jan-Feb test features. "
        "It reweights validation days by similarity to test-day feature signatures.",
        "- `risk_class=baseline_champion` is the current online best reference.",
        "- `winter_overfit_risk` means Jan-Feb-like improves but broad 9-12 validation weakens.",
        "- This table is a promotion filter, not a substitute for online scoring.",
        "",
        "## Candidate Dashboard",
        "",
        markdown_table(top, max_rows=30),
        "",
        "## Feature Drift By Validation Regime",
        "",
        "Lower distance means the validation regime is more similar to the 2026 Jan-Feb "
        "test feature distribution.",
        "",
        markdown_table(feature_drift, max_rows=20),
        "",
        "## Online Score Ledger",
        "",
        markdown_table(online_scores, max_rows=20),
        "",
        "## Promotion Rule",
        "",
        "- Do not promote a candidate based only on `jan_feb_like_mean_profit`.",
        "- Prefer candidates that preserve `all_5fold_mean_profit`, `standard_09_12_mean_profit`, "
        "and `test_like_weighted_profit`.",
        "- Penalize extra loss days and higher p90 regret because online has only 59 days.",
        "- If local lift is below 100 and risk metrics do not improve, skip submission.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--material-dir", default="/Users/yw/ai/electricity/eletricmaterial")
    parser.add_argument("--daily-pattern", default="backtest_*5fold*_daily.csv")
    parser.add_argument("--online-scores", default="reports/online_submission_scores.csv")
    parser.add_argument("--champion-model", default="ens_champion_segmented6_prior_5fold")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    train_boundary, test_boundary = load_boundary_frames(Path(args.material_dir))
    train_sig = daily_signature(train_boundary)
    test_sig = daily_signature(test_boundary)
    day_weights = standardized_nearest_test_distance(train_sig, test_sig)

    daily_reports = load_daily_reports(reports_dir, args.daily_pattern)
    regime_summary = summarize_regimes(daily_reports)
    test_like = summarize_test_like(daily_reports, day_weights)
    feature_drift = summarize_feature_drift(day_weights)

    online_scores = load_online_scores(Path(args.online_scores))
    online_ledger = online_scores.copy()

    dashboard = wide_regime_summary(regime_summary).merge(test_like, on="model", how="left")
    if not online_scores.empty:
        dashboard = dashboard.merge(
            online_scores[["dashboard_model", "online_score", "submitted_at", "notes"]],
            left_on="model",
            right_on="dashboard_model",
            how="left",
        ).drop(columns=["dashboard_model"])
    dashboard = add_champion_deltas(dashboard, args.champion_model)
    dashboard["risk_class"] = dashboard.apply(classify_risk, axis=1)
    dashboard = dashboard.sort_values(
        ["online_score", "all_5fold_mean_profit", "test_like_weighted_profit"],
        ascending=[False, False, False],
        na_position="last",
    )

    out_dashboard = reports_dir / "online_risk_dashboard.csv"
    out_regimes = reports_dir / "online_risk_by_regime.csv"
    out_drift = reports_dir / "online_feature_drift_by_regime.csv"
    out_day_weights = reports_dir / "online_test_like_day_weights.csv"
    out_ledger = reports_dir / "online_score_ledger.csv"
    out_md = reports_dir / "online_risk_dashboard.md"

    dashboard.to_csv(out_dashboard, index=False)
    regime_summary.to_csv(out_regimes, index=False)
    feature_drift.to_csv(out_drift, index=False)
    day_weights.to_csv(out_day_weights, index=False)
    online_ledger.to_csv(out_ledger, index=False)
    write_report(
        path=out_md,
        dashboard=dashboard,
        feature_drift=feature_drift,
        online_scores=online_ledger,
    )

    print(f"dashboard={out_dashboard}")
    print(f"regimes={out_regimes}")
    print(f"feature_drift={out_drift}")
    print(f"day_weights={out_day_weights}")
    print(f"online_ledger={out_ledger}")
    print(f"report={out_md}")


if __name__ == "__main__":
    main()
