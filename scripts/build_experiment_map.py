from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

CONFIGS_DIR = Path("configs")
REPORTS_DIR = Path("reports")
OUT_CSV = REPORTS_DIR / "experiment_map.csv"
OUT_MD = REPORTS_DIR / "experiment_map.md"
BASELINE_MODEL = "ens_champion_segmented6_prior_5fold"
ONLINE_LEDGER = REPORTS_DIR / "online_score_ledger.csv"
EXPERIMENT_INDEX = REPORTS_DIR / "experiment_index.csv"

STANDARD_FOLDS = {"valid_2025_09", "valid_2025_10", "valid_2025_11", "valid_2025_12"}
JAN_FEB_FOLDS = {"valid_2025_jan_feb"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def flatten_members(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if "ensemble" in cfg:
        return list(cfg.get("ensemble", {}).get("members", []))
    return [cfg]


def model_name(cfg: dict[str, Any], path: Path) -> str:
    name = cfg.get("model", {}).get("name")
    return str(name) if name else path.stem


def report_candidates(name: str) -> list[Path]:
    candidates = [
        REPORTS_DIR / f"backtest_{name}.csv",
        REPORTS_DIR / f"{name}.csv",
    ]
    if name.startswith("ensemble_"):
        candidates.append(REPORTS_DIR / f"backtest_{name}.csv")
    return [p for p in candidates if p.exists()]


def daily_candidates(name: str) -> list[Path]:
    candidates = [
        REPORTS_DIR / f"backtest_{name}_daily.csv",
        REPORTS_DIR / f"{name}_daily.csv",
    ]
    return [p for p in candidates if p.exists()]


def safe_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_summary_report(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    out: dict[str, Any] = {"summary_path": str(path)}
    if df.empty:
        return out
    numeric_cols = [
        "avg_profit",
        "oracle_ratio",
        "loss_days",
        "avg_regret",
        "curve_z_rmse",
        "mean_abs_charge_gap",
        "mean_abs_discharge_gap",
    ]
    for col in numeric_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            out[f"{col}_mean"] = float(values.mean())
            out[f"{col}_min"] = float(values.min())
            out[f"{col}_max"] = float(values.max())
    if "fold" in df.columns:
        out["folds"] = "|".join(df["fold"].astype(str).tolist())
        out["has_jan_feb_like"] = bool(df["fold"].astype(str).isin(JAN_FEB_FOLDS).any())
        jan = df[df["fold"].astype(str).isin(JAN_FEB_FOLDS)]
        if not jan.empty and "avg_profit" in jan.columns:
            out["jan_feb_like_avg_profit"] = float(pd.to_numeric(jan["avg_profit"]).mean())
        standard = df[df["fold"].astype(str).isin(STANDARD_FOLDS)]
        if not standard.empty and "avg_profit" in standard.columns:
            out["standard_09_12_avg_profit"] = float(pd.to_numeric(standard["avg_profit"]).mean())
    return out


def summarize_daily_report(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    out: dict[str, Any] = {"daily_path": str(path)}
    if df.empty or "profit" not in df.columns:
        return out
    profit = pd.to_numeric(df["profit"], errors="coerce")
    out["daily_days"] = int(len(df))
    out["daily_mean_profit"] = float(profit.mean())
    out["daily_worst_profit"] = float(profit.min())
    out["daily_loss_days"] = int((profit < -1e-9).sum())
    if "regret" in df.columns:
        regret = pd.to_numeric(df["regret"], errors="coerce")
        out["daily_mean_regret"] = float(regret.mean())
        out["daily_p90_regret"] = float(regret.quantile(0.9))
    if "fold" in df.columns:
        fold_series = df["fold"].astype(str)
        jan = df[fold_series.isin(JAN_FEB_FOLDS)]
        if not jan.empty:
            out["jan_feb_like_daily_mean_profit"] = float(
                pd.to_numeric(jan["profit"], errors="coerce").mean()
            )
            out["jan_feb_like_daily_loss_days"] = int(
                (pd.to_numeric(jan["profit"], errors="coerce") < -1e-9).sum()
            )
        standard = df[fold_series.isin(STANDARD_FOLDS)]
        if not standard.empty:
            out["standard_09_12_daily_mean_profit"] = float(
                pd.to_numeric(standard["profit"], errors="coerce").mean()
            )
            out["standard_09_12_daily_loss_days"] = int(
                (pd.to_numeric(standard["profit"], errors="coerce") < -1e-9).sum()
            )
    return out


def classify_cfg(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    members = flatten_members(cfg)
    member_names = [str(m.get("model", {}).get("name", "")) for m in members]
    backends = [str(m.get("model", {}).get("backend", "")) for m in members]
    target_modes = [str(m.get("model", {}).get("target_mode", "")) for m in members]
    train_windows = [m.get("model", {}).get("train_window_days") for m in members]
    feature_flags: dict[str, int] = defaultdict(int)
    for member in members:
        for key, value in (member.get("feature_sets", {}) or {}).items():
            if value:
                feature_flags[str(key)] += 1
    if len(members) > 1:
        family = "ensemble"
    elif "pair_spread" in name:
        family = "pair_window"
    elif "window_mean" in name:
        family = "window"
    elif any("segment_boundaries" in m.get("model", {}) for m in members):
        family = "segmented"
    elif name.startswith("cat") or any(b == "catboost" for b in backends):
        family = "catboost"
    elif name.startswith("xgb") or any(b == "xgboost" for b in backends):
        family = "xgboost"
    elif name.startswith("linear") or any(b == "linear" for b in backends):
        family = "linear"
    else:
        family = "lightgbm"

    target_family = "absolute"
    if any("zscore" in mode for mode in target_modes):
        target_family = "shape_zscore"
    elif any("centered" in mode for mode in target_modes):
        target_family = "shape_centered"
    elif "pair_spread" in name:
        target_family = "pair_spread"
    elif "window_mean" in name:
        target_family = "window_mean"
    elif len(members) > 1:
        target_family = "ensemble"

    fold_names = [str(f.get("name", "")) for f in cfg.get("folds", [])]
    return {
        "model": name,
        "model_family": family,
        "target_family": target_family,
        "member_count": len(members),
        "member_names": "|".join(member_names),
        "train_windows": "|".join("all" if x is None else str(x) for x in train_windows),
        "feature_flags": "|".join(sorted(feature_flags)) if feature_flags else "base",
        "has_bid_space": bool(feature_flags.get("bid_space")),
        "has_business": bool(feature_flags.get("business")),
        "has_capacity": bool(feature_flags.get("capacity")),
        "has_holiday": bool(feature_flags.get("holiday")),
        "has_lag": bool(feature_flags.get("lag")),
        "has_weekly_relative": bool(feature_flags.get("weekly_relative")),
        "has_nwp": bool(feature_flags.get("nwp")),
        "has_nwp_residual": bool(feature_flags.get("nwp_residual")),
        "has_bias_correction": bool(feature_flags.get("bias_correction")),
        "has_weather_correction": bool(feature_flags.get("weather_correction")),
        "has_jan_feb_fold_cfg": "valid_2025_jan_feb" in fold_names,
        "fold_count_cfg": len(fold_names),
    }


def load_existing_notes() -> dict[str, dict[str, Any]]:
    notes: dict[str, dict[str, Any]] = {}
    if EXPERIMENT_INDEX.exists():
        exp = pd.read_csv(EXPERIMENT_INDEX)
        config_counts = exp["config_path"].value_counts() if "config_path" in exp.columns else {}
        for _, row in exp.iterrows():
            model = row.get("run_id")
            if isinstance(model, str):
                notes[model] = {
                    "index_status": row.get("status"),
                    "index_alias": row.get("alias"),
                    "index_notes": row.get("notes"),
                }
            cfg_path = row.get("config_path")
            if isinstance(cfg_path, str) and int(config_counts.get(cfg_path, 0)) == 1:
                notes[cfg_path] = {
                    "index_status": row.get("status"),
                    "index_alias": row.get("alias"),
                    "index_notes": row.get("notes"),
                }
    return notes


def load_online_scores() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not ONLINE_LEDGER.exists():
        return out
    ledger = pd.read_csv(ONLINE_LEDGER)
    for _, row in ledger.iterrows():
        cfg_path = row.get("config_path")
        if isinstance(cfg_path, str):
            out[cfg_path] = {
                "online_score": safe_float(row.get("online_score")),
                "online_model": row.get("model"),
                "online_notes": row.get("notes"),
            }
    return out


def decision_bucket(row: dict[str, Any], baseline: dict[str, Any]) -> str:
    status = str(row.get("index_status") or "").lower()
    if status in {"promoted", "rejected", "candidate"}:
        return status
    online = row.get("online_score")
    if online is not None and not pd.isna(online):
        return "online_tested"
    profit = row.get("avg_profit_mean")
    if profit is None:
        profit = row.get("daily_mean_profit")
    base_profit = baseline.get("avg_profit_mean") or baseline.get("daily_mean_profit")
    if (
        profit is None
        or base_profit is None
        or pd.isna(profit)
        or pd.isna(base_profit)
    ):
        return "unmeasured"
    delta = float(profit) - float(base_profit)
    if delta >= 100:
        return "candidate"
    if delta <= -100:
        return "global_failed"
    return "near_champion"


def opportunity_tags(row: dict[str, Any]) -> str:
    tags: list[str] = []
    status = row.get("decision_bucket")
    if status in {"global_failed", "rejected"}:
        if row.get("has_nwp") or row.get("has_weather_correction") or row.get("has_nwp_residual"):
            tags.append("weather_as_global_failed_try_as_risk_or_rerank")
        if row.get("has_bid_space") or "margin" in str(row.get("feature_flags", "")):
            tags.append("bidspace_as_global_failed_try_as_aux_signal")
        if row.get("target_family") in {"window_mean", "pair_spread"}:
            tags.append("window_pair_global_failed_try_as_topk_expert")
    if row.get("has_jan_feb_fold_cfg") and row.get("jan_feb_like_avg_profit") is not None:
        tags.append("has_jan_feb_evidence")
    if row.get("model_family") in {"catboost", "xgboost"}:
        tags.append("model_family_diversity")
    if row.get("has_weekly_relative") or row.get("has_lag"):
        tags.append("temporal_relative_candidate")
    return "|".join(tags)


def build_rows() -> list[dict[str, Any]]:
    notes = load_existing_notes()
    online = load_online_scores()
    rows: list[dict[str, Any]] = []
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        cfg = load_yaml(path)
        name = model_name(cfg, path)
        row: dict[str, Any] = {
            "config_path": str(path),
            **classify_cfg(cfg, name),
        }
        summaries = report_candidates(name)
        if summaries:
            row.update(summarize_summary_report(summaries[0]))
        dailies = daily_candidates(name)
        if dailies:
            row.update(summarize_daily_report(dailies[0]))
        if str(path) in notes:
            row.update(notes[str(path)])
        if name in notes:
            row.update({k: v for k, v in notes[name].items() if k not in row or pd.isna(row[k])})
        if str(path) in online:
            row.update(online[str(path)])
        rows.append(row)
    return rows


def markdown_table(df: pd.DataFrame, cols: list[str], *, limit: int = 20) -> str:
    block = df[cols].head(limit).copy()
    headers = cols
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in block.iterrows():
        cells = []
        for col in cols:
            value = row.get(col)
            if isinstance(value, float):
                cells.append("" if pd.isna(value) else f"{value:.3f}")
            else:
                text = "" if pd.isna(value) else str(value)
                cells.append(text.replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(df: pd.DataFrame, baseline: dict[str, Any]) -> None:
    measured = df[df["decision_bucket"] != "unmeasured"].copy()
    by_bucket = df["decision_bucket"].value_counts(dropna=False).reset_index()
    top_cols = [
        "model",
        "decision_bucket",
        "model_family",
        "feature_flags",
        "avg_profit_mean",
        "jan_feb_like_avg_profit",
        "daily_loss_days",
        "online_score",
        "opportunity_tags",
    ]
    top = measured.sort_values(
        ["online_score", "avg_profit_mean", "jan_feb_like_avg_profit"],
        ascending=[False, False, False],
        na_position="last",
    )
    unmeasured = df[df["decision_bucket"] == "unmeasured"].copy()
    opportunity = df[df["opportunity_tags"].astype(str) != ""].copy()
    opportunity = opportunity.sort_values(
        ["decision_bucket", "avg_profit_mean", "jan_feb_like_avg_profit"],
        ascending=[True, False, False],
        na_position="last",
    )
    lines = [
        "# Experiment Map",
        "",
        "This map summarizes existing configs, backtest reports, online scores, and manual notes.",
        "It is intended to expose matrix-style experiment coverage and avoid linear local search.",
        "",
        "## Baseline",
        "",
        f"- Baseline model for local comparison: `{BASELINE_MODEL}`.",
        f"- Baseline avg_profit_mean: `{baseline.get('avg_profit_mean')}`.",
        f"- Baseline Jan-Feb-like avg_profit: `{baseline.get('jan_feb_like_avg_profit')}`.",
        "",
        "## Decision Buckets",
        "",
        markdown_table(by_bucket, ["decision_bucket", "count"], limit=20),
        "",
        "## Top Measured Experiments",
        "",
        markdown_table(top, top_cols, limit=30),
        "",
        "## Reusable Opportunity Tags",
        "",
        markdown_table(opportunity, top_cols, limit=30),
        "",
        "## Unmeasured Configs",
        "",
        markdown_table(
            unmeasured,
            ["model", "model_family", "feature_flags", "target_family", "config_path"],
            limit=40,
        ),
        "",
        "## Artifacts",
        "",
        f"- `{OUT_CSV}`",
        f"- `{OUT_MD}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    rows = build_rows()
    df = pd.DataFrame(rows)
    baseline_row = df[df["model"] == BASELINE_MODEL]
    baseline = baseline_row.iloc[0].to_dict() if not baseline_row.empty else {}
    df["decision_bucket"] = [decision_bucket(row._asdict(), baseline) for row in df.itertuples()]
    df["opportunity_tags"] = [opportunity_tags(row._asdict()) for row in df.itertuples()]
    sort_cols = ["decision_bucket", "online_score", "avg_profit_mean", "jan_feb_like_avg_profit"]
    df = df.sort_values(sort_cols, ascending=[True, False, False, False], na_position="last")
    df.to_csv(OUT_CSV, index=False)
    write_markdown(df, baseline)
    print(f"map_path={OUT_CSV}")
    print(f"markdown_path={OUT_MD}")
    print(f"rows={len(df)}")


if __name__ == "__main__":
    main()
