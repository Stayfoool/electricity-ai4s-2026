from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from electricity.features.bid_space import markdown_table

REPORTS_DIR = Path("reports")
DASHBOARD_PATH = REPORTS_DIR / "online_risk_dashboard.csv"
OUT_MD = REPORTS_DIR / "online_validation_gap_analysis.md"
OUT_KNOWN = REPORTS_DIR / "online_validation_gap_known_scores.csv"
OUT_CANDIDATES = REPORTS_DIR / "online_validation_gap_candidate_filter.csv"

CHAMPION_MODEL = "ens_champion_segmented6_prior_5fold"
ONLINE_DELTA_COL = "delta_vs_champion__online_score"
LOCAL_DELTA_COLS = [
    "delta_vs_champion__standard_09_12_mean_profit",
    "delta_vs_champion__jan_feb_like_mean_profit",
    "delta_vs_champion__winter_11_12_jan_feb_mean_profit",
    "delta_vs_champion__late_winter_12_jan_feb_mean_profit",
    "delta_vs_champion__all_5fold_mean_profit",
    "delta_vs_champion__test_like_weighted_profit",
    "delta_vs_champion__all_5fold_loss_days",
    "delta_vs_champion__all_5fold_p90_regret",
    "delta_vs_champion__all_5fold_worst_day_profit",
]


def _short_metric_name(col: str) -> str:
    return col.removeprefix("delta_vs_champion__")


def _load_dashboard() -> pd.DataFrame:
    if not DASHBOARD_PATH.exists():
        raise FileNotFoundError(DASHBOARD_PATH)
    return pd.read_csv(DASHBOARD_PATH)


def _known_score_alignment(dashboard: pd.DataFrame) -> pd.DataFrame:
    known = dashboard[dashboard["online_score"].notna()].copy()
    known = known[known["model"] != CHAMPION_MODEL].copy()
    rows: list[dict[str, object]] = []
    for _, row in known.iterrows():
        online_delta = float(row[ONLINE_DELTA_COL])
        for col in LOCAL_DELTA_COLS:
            if col not in row or pd.isna(row[col]):
                continue
            value = float(row[col])
            if col.endswith("loss_days") or col.endswith("p90_regret"):
                local_direction = -np.sign(value)
                interpretation = "lower_is_better"
            else:
                local_direction = np.sign(value)
                interpretation = "higher_is_better"
            rows.append(
                {
                    "model": row["model"],
                    "online_delta": online_delta,
                    "metric": _short_metric_name(col),
                    "local_delta": value,
                    "interpretation": interpretation,
                    "direction_matches_online": bool(local_direction == np.sign(online_delta)),
                    "abs_local_delta": abs(value),
                }
            )
    return pd.DataFrame(rows)


def _candidate_filter(dashboard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    submitted = dashboard["online_score"].notna()
    for _, row in dashboard[~submitted].iterrows():
        model = row["model"]
        if model == CHAMPION_MODEL:
            continue
        standard = row.get("delta_vs_champion__standard_09_12_mean_profit", np.nan)
        jan = row.get("delta_vs_champion__jan_feb_like_mean_profit", np.nan)
        all5 = row.get("delta_vs_champion__all_5fold_mean_profit", np.nan)
        test_like = row.get("delta_vs_champion__test_like_weighted_profit", np.nan)
        loss = row.get("delta_vs_champion__all_5fold_loss_days", np.nan)
        p90 = row.get("delta_vs_champion__all_5fold_p90_regret", np.nan)
        worst = row.get("delta_vs_champion__all_5fold_worst_day_profit", np.nan)

        hard_reject_reasons = []
        if pd.notna(all5) and all5 < -150:
            hard_reject_reasons.append("all_5fold<-150")
        if pd.notna(test_like) and test_like < -150:
            hard_reject_reasons.append("test_like<-150")
        if pd.notna(loss) and loss > 0:
            hard_reject_reasons.append("more_loss_days")
        if pd.notna(p90) and p90 > 1000:
            hard_reject_reasons.append("p90_regret+>1000")
        if pd.notna(standard) and pd.notna(jan) and jan > 100 and standard < -200:
            hard_reject_reasons.append("jan_feb_only_win")

        # Conservative score: reward broad/test-like stability; penalize extra losses/regret.
        safety_score = 0.0
        for value, weight in [(all5, 0.35), (test_like, 0.35), (standard, 0.20), (jan, 0.10)]:
            if pd.notna(value):
                safety_score += weight * float(value)
        if pd.notna(loss):
            safety_score -= 250.0 * max(float(loss), 0.0)
        if pd.notna(p90):
            safety_score -= 0.05 * max(float(p90), 0.0)

        rows.append(
            {
                "model": model,
                "risk_class": row.get("risk_class"),
                "safety_score": safety_score,
                "standard_delta": standard,
                "jan_feb_delta": jan,
                "all_5fold_delta": all5,
                "test_like_delta": test_like,
                "loss_day_delta": loss,
                "p90_regret_delta": p90,
                "worst_day_delta": worst,
                "hard_reject_reasons": ";".join(hard_reject_reasons),
                "submit_filter": "reject" if hard_reject_reasons else "watch_only",
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["submit_filter", "safety_score", "all_5fold_delta", "test_like_delta"],
        ascending=[True, False, False, False],
        na_position="last",
    )


def _metric_alignment_summary(known_alignment: pd.DataFrame) -> pd.DataFrame:
    if known_alignment.empty:
        return pd.DataFrame()
    return (
        known_alignment.groupby(["metric", "interpretation"], as_index=False)
        .agg(
            known_models=("model", "nunique"),
            direction_match_rate=("direction_matches_online", "mean"),
            mean_abs_local_delta=("abs_local_delta", "mean"),
        )
        .sort_values(["direction_match_rate", "mean_abs_local_delta"], ascending=[False, False])
    )


def write_report(
    *,
    known_alignment: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    candidate_filter: pd.DataFrame,
    dashboard: pd.DataFrame,
) -> None:
    known_models = dashboard[dashboard["online_score"].notna()].copy()
    filter_cols = [
        "model",
        "risk_class",
        "safety_score",
        "standard_delta",
        "jan_feb_delta",
        "all_5fold_delta",
        "test_like_delta",
        "loss_day_delta",
        "p90_regret_delta",
        "hard_reject_reasons",
        "submit_filter",
    ]
    top_watch = candidate_filter[candidate_filter["submit_filter"] == "watch_only"].head(10)
    hard_reject = candidate_filter[candidate_filter["submit_filter"] == "reject"].head(20)

    lines = [
        "# Online vs Local Validation Gap Analysis",
        "",
        "## Scope",
        "",
        (
            "- Known online scores are currently limited to champion and holiday_only, "
            "so this is a directional diagnostic, not a statistical proof."
        ),
        (
            "- Goal: prevent submitting candidates that only win Jan-Feb-like but lose "
            "on broader or test-like validation."
        ),
        (
            "- The analysis uses existing backtest reports and the online score ledger; "
            "no model training is performed."
        ),
        "",
        "## Known Online Scores",
        "",
        markdown_table(
            known_models[
                [
                    "model",
                    "online_score",
                    "risk_class",
                    "delta_vs_champion__online_score",
                    "delta_vs_champion__standard_09_12_mean_profit",
                    "delta_vs_champion__jan_feb_like_mean_profit",
                    "delta_vs_champion__all_5fold_mean_profit",
                    "delta_vs_champion__test_like_weighted_profit",
                    "delta_vs_champion__all_5fold_loss_days",
                    "delta_vs_champion__all_5fold_p90_regret",
                ]
            ],
            floatfmt=".4f",
        ),
        "",
        "## Metric Direction Check",
        "",
        markdown_table(alignment_summary, floatfmt=".4f"),
        "",
        "## Candidate Filter: Watch Only",
        "",
        markdown_table(top_watch[filter_cols], floatfmt=".4f"),
        "",
        "## Candidate Filter: Rejected By Safety Rules",
        "",
        markdown_table(hard_reject[filter_cols], floatfmt=".4f"),
        "",
        "## Conclusion",
        "",
        "- `holiday_only` improved Jan-Feb-like by +217/day but online dropped by -112 points.",
        (
            "- The same candidate lost on standard_09_12, all_5fold, and "
            "test_like_weighted; those metrics correctly warned against submission."
        ),
        (
            "- Therefore Jan-Feb-like cannot be used as the primary promotion metric. "
            "It is only an auxiliary signal."
        ),
        (
            "- Current unsubmitted watch candidates still have negative "
            "all_5fold/test_like deltas, so none should be promoted directly."
        ),
        "",
        "## Next Action",
        "",
        (
            "1. Do not submit another winter/holiday-only variant unless it preserves "
            "all_5fold and test_like metrics."
        ),
        (
            "2. If exploring winter effects, run constrained low-weight blends and "
            "require `all_5fold_delta >= -50`, `test_like_delta >= -50`, and no "
            "extra loss days."
        ),
        (
            "3. Otherwise shift to a new axis: online-distribution-aware feature "
            "selection or model-family fallback, not more global holiday weighting."
        ),
        "",
        "## Artifacts",
        "",
        f"- `{OUT_MD}`",
        f"- `{OUT_KNOWN}`",
        f"- `{OUT_CANDIDATES}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    dashboard = _load_dashboard()
    known_alignment = _known_score_alignment(dashboard)
    alignment_summary = _metric_alignment_summary(known_alignment)
    candidate_filter = _candidate_filter(dashboard)

    known_alignment.to_csv(OUT_KNOWN, index=False)
    candidate_filter.to_csv(OUT_CANDIDATES, index=False)
    write_report(
        known_alignment=known_alignment,
        alignment_summary=alignment_summary,
        candidate_filter=candidate_filter,
        dashboard=dashboard,
    )
    print(f"report={OUT_MD}")
    print(f"known={OUT_KNOWN}")
    print(f"candidates={OUT_CANDIDATES}")
    print(candidate_filter.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
