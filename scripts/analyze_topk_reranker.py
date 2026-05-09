from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPORTS_DIR = Path("reports")
MODEL = "ens_champion_segmented6"
TOP_K = 10

DAY_FEATURES = [
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
    "champion_top1_top2_gap",
    "champion_top1_top5_mean_gap",
    "champion_top5_spread_std",
]

CANDIDATE_FEATURES = [
    "candidate_rank",
    "charge_start",
    "discharge_start",
    "predicted_spread",
    "predicted_spread_delta_from_top1",
    "charge_delta_from_top1",
    "discharge_delta_from_top1",
    "same_charge_as_top1",
    "same_discharge_as_top1",
]


def build_candidate_dataset() -> pd.DataFrame:
    candidates = pd.read_csv(REPORTS_DIR / "topk_pair_candidates.csv", parse_dates=["date"])
    days = pd.read_csv(REPORTS_DIR / "strategy_gate_dataset.csv", parse_dates=["date"])
    candidates = candidates[
        (candidates["model"] == MODEL) & (candidates["candidate_rank"] <= TOP_K)
    ].copy()
    top1 = candidates[candidates["candidate_rank"] == 1][
        ["date", "predicted_spread", "charge_start", "discharge_start"]
    ].rename(
        columns={
            "predicted_spread": "top1_predicted_spread",
            "charge_start": "top1_charge_start",
            "discharge_start": "top1_discharge_start",
        }
    )
    out = candidates.merge(top1, on="date", how="left")
    out["predicted_spread_delta_from_top1"] = (
        out["top1_predicted_spread"] - out["predicted_spread"]
    )
    out["charge_delta_from_top1"] = (out["charge_start"] - out["top1_charge_start"]).abs()
    out["discharge_delta_from_top1"] = (
        out["discharge_start"] - out["top1_discharge_start"]
    ).abs()
    out["same_charge_as_top1"] = (out["charge_delta_from_top1"] == 0).astype(int)
    out["same_discharge_as_top1"] = (out["discharge_delta_from_top1"] == 0).astype(int)

    day_cols = ["date", *DAY_FEATURES]
    out = out.merge(days[day_cols], on="date", how="left")
    out["target_true_profit"] = out["true_profit"]
    return out.sort_values(["date", "candidate_rank"]).reset_index(drop=True)


def fit_reranker(train_df: pd.DataFrame) -> Pipeline:
    features = [*CANDIDATE_FEATURES, *DAY_FEATURES]
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=80,
                    learning_rate=0.03,
                    max_depth=2,
                    min_samples_leaf=12,
                    random_state=2026,
                ),
            ),
        ]
    )
    model.fit(train_df[features], train_df["target_true_profit"])
    return model


def choose_by_prediction(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    idx = df.groupby("date")[score_col].idxmax()
    return df.loc[idx].copy()


def cross_fold_evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [*CANDIDATE_FEATURES, *DAY_FEATURES]
    fold_rows: list[dict[str, object]] = []
    selected_rows: list[pd.DataFrame] = []

    for fold in sorted(df["fold"].unique()):
        train_df = df[df["fold"] != fold].copy()
        valid_df = df[df["fold"] == fold].copy()
        reranker = fit_reranker(train_df)
        valid_df["rerank_score"] = reranker.predict(valid_df[features])

        top1 = valid_df[valid_df["candidate_rank"] == 1].copy()
        reranked = choose_by_prediction(valid_df, "rerank_score")
        oracle_top10 = choose_by_prediction(valid_df, "true_profit")

        for selected_name, selected in [
            ("top1", top1),
            ("reranker", reranked),
            ("oracle_top10", oracle_top10),
        ]:
            profit = selected["true_profit"].astype(float)
            fold_rows.append(
                {
                    "heldout_fold": fold,
                    "method": selected_name,
                    "valid_days": len(selected),
                    "mean_profit": float(profit.mean()),
                    "worst_profit": float(profit.min()),
                    "loss_days": int((profit < 0).sum()),
                    "mean_rank": float(selected["candidate_rank"].mean()),
                    "rank1_days": int((selected["candidate_rank"] == 1).sum()),
                }
        )

        reranked["selected_method"] = "reranker"
        reranked["top1_profit"] = (
            top1.set_index("date").loc[reranked["date"], "true_profit"].to_numpy()
        )
        reranked["oracle_top10_profit"] = (
            oracle_top10.set_index("date").loc[reranked["date"], "true_profit"].to_numpy()
        )
        reranked["lift_vs_top1"] = reranked["true_profit"] - reranked["top1_profit"]
        reranked["miss_vs_oracle_top10"] = (
            reranked["oracle_top10_profit"] - reranked["true_profit"]
        )
        selected_rows.append(
            reranked[
                [
                    "date",
                    "fold",
                    "candidate_rank",
                    "charge_start",
                    "discharge_start",
                    "predicted_spread",
                    "rerank_score",
                    "true_profit",
                    "top1_profit",
                    "oracle_top10_profit",
                    "lift_vs_top1",
                    "miss_vs_oracle_top10",
                ]
            ].copy()
        )

    return pd.DataFrame(fold_rows), pd.concat(selected_rows, ignore_index=True)


def build_overall(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in folds.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "mean_profit": float(
                    np.average(group["mean_profit"], weights=group["valid_days"])
                ),
                "worst_fold_profit": float(group["mean_profit"].min()),
                "loss_days": int(group["loss_days"].sum()),
                "mean_selected_rank": float(
                    np.average(group["mean_rank"], weights=group["valid_days"])
                ),
                "rank1_days": int(group["rank1_days"].sum()),
            }
        )
    overall = pd.DataFrame(rows)
    top1_mean = float(overall.loc[overall["method"] == "top1", "mean_profit"].iloc[0])
    overall["lift_vs_top1"] = overall["mean_profit"] - top1_mean
    return overall


def write_markdown(*, overall: pd.DataFrame, folds: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Top-K Reranker Diagnostics",
        "",
        "## Scope",
        "",
        f"Model: `{MODEL}`. Candidate set: predicted top `{TOP_K}` legal pairs. "
        "The reranker is trained only on held-in months and evaluated on held-out months.",
        "",
        "## Overall Held-Out Result",
        "",
        overall.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Fold Detail",
        "",
        folds.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Reranker Selected Rank Counts",
        "",
        selected["candidate_rank"].value_counts().sort_index().rename_axis("rank").reset_index(
            name="days"
        ).to_markdown(index=False),
        "",
        "## Decision",
        "",
    ]
    rerank_lift = float(overall.loc[overall["method"] == "reranker", "lift_vs_top1"].iloc[0])
    if rerank_lift > 0:
        lines += [
            "- The held-out reranker improves over predicted top1.",
            "- Next step is to harden it into a submit-time strategy only if fold-level "
            "results are stable.",
        ]
    else:
        lines += [
            "- The held-out reranker does not beat predicted top1.",
            "- Keep top-K as diagnostic evidence rather than changing dispatch.",
        ]
    (REPORTS_DIR / "topk_reranker_diagnostics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    dataset = build_candidate_dataset()
    folds, selected = cross_fold_evaluate(dataset)
    overall = build_overall(folds)

    dataset.to_csv(REPORTS_DIR / "topk_reranker_dataset.csv", index=False)
    folds.to_csv(REPORTS_DIR / "topk_reranker_crossfold_summary.csv", index=False)
    selected.to_csv(REPORTS_DIR / "topk_reranker_selected_daily.csv", index=False)
    overall.to_csv(REPORTS_DIR / "topk_reranker_overall.csv", index=False)
    write_markdown(overall=overall, folds=folds, selected=selected)

    print(overall.to_string(index=False))
    print(f"dataset_path={REPORTS_DIR / 'topk_reranker_dataset.csv'}")
    print(f"summary_path={REPORTS_DIR / 'topk_reranker_crossfold_summary.csv'}")
    print(f"selected_path={REPORTS_DIR / 'topk_reranker_selected_daily.csv'}")
    print(f"markdown_path={REPORTS_DIR / 'topk_reranker_diagnostics.md'}")


if __name__ == "__main__":
    main()
