from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import analyze_current_prior_topk_pair_reranker as base
import pandas as pd

from electricity.data import load_train_frame
from electricity.features import build_feature_frame, nwp_residual_source_columns
from electricity.features.bid_space import markdown_table

REPORTS_DIR = Path("reports")
DATASET_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_foldsafe_dataset.csv"
SELECTED_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_foldsafe_selected_daily.csv"
FOLD_SUMMARY_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_foldsafe_by_fold.csv"
REGIME_SUMMARY_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_foldsafe_by_regime.csv"
OVERALL_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_foldsafe_overall.csv"
PARAMS_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_foldsafe_params.csv"
OUT_MD = REPORTS_DIR / "current_prior_topk_pair_reranker_foldsafe_diagnostics.md"


def _raw_nwp_feature_frame(cfg: dict) -> pd.DataFrame:
    """Build features with raw NWP columns but without global NWP residuals."""
    build_spec = deepcopy(cfg)
    feature_sets = dict(build_spec.get("feature_sets", {}))
    feature_sets["nwp"] = True
    feature_sets["nwp_residual"] = False
    build_spec["feature_sets"] = feature_sets

    time_col = build_spec["data"]["time_col"]
    return build_feature_frame(load_train_frame(build_spec), build_spec, time_col=time_col)


def _add_fold_safe_nwp_residuals(
    df: pd.DataFrame,
    cfg: dict,
    *,
    train_mask: pd.Series,
) -> pd.DataFrame:
    """Add NWP residuals using month-hour means estimated outside held-out dates.

    If the held-out month does not exist in the training side, fall back to an
    hour-of-day mean, then to a global training mean. This avoids using held-out
    feature distribution to normalize held-out rows.
    """
    out = df.copy()
    cols = nwp_residual_source_columns(cfg)
    missing = sorted(set(cols) - set(out.columns))
    if missing:
        raise ValueError(f"cannot build fold-safe NWP residual features, missing: {missing}")
    if "month" not in out.columns or "hour" not in out.columns:
        raise ValueError("fold-safe NWP residual features require month/hour time features")

    train = out.loc[train_mask].copy()
    if train.empty:
        raise ValueError("fold-safe NWP residual training side is empty")

    month_hour_means = train.groupby(["month", "hour"], as_index=False)[cols].mean()
    hour_means = train.groupby("hour", as_index=False)[cols].mean()
    global_means = train[cols].mean()

    month_hour_means = month_hour_means.rename(
        columns={col: f"__mh_mean_{col}" for col in cols}
    )
    hour_means = hour_means.rename(columns={col: f"__h_mean_{col}" for col in cols})

    out = out.merge(month_hour_means, on=["month", "hour"], how="left")
    out = out.merge(hour_means, on="hour", how="left")
    temp_cols: list[str] = []
    for col in cols:
        mh_col = f"__mh_mean_{col}"
        h_col = f"__h_mean_{col}"
        temp_cols.extend([mh_col, h_col])
        baseline = out[mh_col].fillna(out[h_col]).fillna(float(global_means[col]))
        out[f"{col}_mh_resid"] = out[col] - baseline

    return out.drop(columns=temp_cols)


def _fold_safe_slot_frame(
    raw_df: pd.DataFrame,
    cfg: dict,
    candidates: pd.DataFrame,
    *,
    heldout_fold: str,
) -> tuple[pd.DataFrame, list[str]]:
    time_col = cfg["data"]["time_col"]
    heldout_dates = pd.to_datetime(
        candidates.loc[candidates["fold"] == heldout_fold, "date"]
    ).dt.normalize()
    heldout_date_set = set(heldout_dates)
    date = raw_df[time_col].dt.normalize()
    train_mask = ~date.isin(heldout_date_set)

    df = _add_fold_safe_nwp_residuals(raw_df, cfg, train_mask=train_mask)
    point_cols = list(cfg["model"]["pair_point_feature_cols"])
    missing = sorted(set(point_cols) - set(df.columns))
    if missing:
        raise ValueError(f"feature frame missing pair_point_feature_cols: {missing}")

    out = df[[time_col, *point_cols]].copy()
    out["date"] = out[time_col].dt.normalize()
    out["slot"] = out[time_col].dt.hour * 4 + out[time_col].dt.minute // 15
    return out.sort_values(["date", "slot"]).reset_index(drop=True), point_cols


def _build_dataset_for_heldout(
    *,
    raw_df: pd.DataFrame,
    cfg: dict,
    candidates: pd.DataFrame,
    heldout_fold: str,
) -> pd.DataFrame:
    slot_df, point_cols = _fold_safe_slot_frame(
        raw_df,
        cfg,
        candidates,
        heldout_fold=heldout_fold,
    )
    day_maps = base._day_feature_maps(slot_df, point_cols)
    dataset = base._add_pair_signal_features(candidates, day_maps, point_cols)
    return dataset.sort_values(["fold", "date", "candidate_rank"]).reset_index(drop=True)


def crossfold_evaluate_foldsafe(
    *,
    raw_df: pd.DataFrame,
    cfg: dict,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_frames = []
    param_rows = []
    dataset_frames = []

    for heldout_fold in sorted(candidates["fold"].unique()):
        dataset = _build_dataset_for_heldout(
            raw_df=raw_df,
            cfg=cfg,
            candidates=candidates,
            heldout_fold=str(heldout_fold),
        )
        dataset["residual_fit_scope"] = f"exclude_{heldout_fold}"
        dataset_frames.append(dataset[dataset["fold"] == heldout_fold].copy())

        features = base.feature_columns(dataset)
        train = dataset[dataset["fold"] != heldout_fold].copy()
        valid = dataset[dataset["fold"] == heldout_fold].copy()

        anchor = base.select_daily(
            valid.assign(anchor_score=-valid["candidate_rank"]),
            score_col="anchor_score",
            target="lift",
            threshold=0.0,
            method="anchor_top1",
        )
        oracle = base.select_daily(
            valid.assign(oracle_score=valid["true_profit"]),
            score_col="oracle_score",
            target="true_profit",
            threshold=-1_000_000.0,
            method="oracle_top10",
        )
        selected_frames.extend([anchor, oracle])

        for target in base.MODEL_TARGETS:
            oof = base._inner_oof_predictions(train, features=features, target=target)
            threshold = base._best_threshold_from_oof(
                oof,
                target=target,
                score_col="rerank_score",
            )
            model = base.fit_model(train, features, target=target)
            scored_valid = valid.copy()
            scored_valid["rerank_score"] = model.predict(scored_valid[features])
            method = f"gbr_{target}"
            selected = base.select_daily(
                scored_valid,
                score_col="rerank_score",
                target=target,
                threshold=threshold,
                method=method,
            )
            selected_frames.append(selected)
            param_rows.append(
                {
                    "heldout_fold": heldout_fold,
                    "method": method,
                    "threshold": threshold,
                    "feature_count": len(features),
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                }
            )

    dataset_all = pd.concat(dataset_frames, ignore_index=True)
    selected_all = pd.concat(selected_frames, ignore_index=True)
    selected_all["date"] = pd.to_datetime(selected_all["date"])
    by_fold = base.summarize_selection(selected_all, group_cols=["fold", "method"])
    return dataset_all, selected_all, by_fold, pd.DataFrame(param_rows)


def _read_original_overall() -> pd.DataFrame | None:
    if not base.OVERALL_PATH.exists():
        return None
    return pd.read_csv(base.OVERALL_PATH)


def _comparison_table(foldsafe_overall: pd.DataFrame) -> pd.DataFrame:
    original = _read_original_overall()
    if original is None:
        return pd.DataFrame()

    cols = ["method", "mean_profit", "loss_days", "changed_days", "profit_delta_vs_anchor"]
    left = original[original["regime"] == "all_5fold"][cols].rename(
        columns={
            "mean_profit": "original_mean_profit",
            "loss_days": "original_loss_days",
            "changed_days": "original_changed_days",
            "profit_delta_vs_anchor": "original_delta_vs_anchor",
        }
    )
    right = foldsafe_overall[foldsafe_overall["regime"] == "all_5fold"][cols].rename(
        columns={
            "mean_profit": "foldsafe_mean_profit",
            "loss_days": "foldsafe_loss_days",
            "changed_days": "foldsafe_changed_days",
            "profit_delta_vs_anchor": "foldsafe_delta_vs_anchor",
        }
    )
    out = left.merge(right, on="method", how="outer")
    out["mean_profit_change"] = out["foldsafe_mean_profit"] - out["original_mean_profit"]
    out["delta_change"] = out["foldsafe_delta_vs_anchor"] - out["original_delta_vs_anchor"]
    return out.sort_values("foldsafe_mean_profit", ascending=False)


def write_report(
    *,
    dataset: pd.DataFrame,
    selected: pd.DataFrame,
    by_fold: pd.DataFrame,
    by_regime: pd.DataFrame,
    overall: pd.DataFrame,
    params: pd.DataFrame,
) -> None:
    comparison = _comparison_table(overall)
    rank_counts = (
        selected.groupby(["method", "candidate_rank"], as_index=False)
        .size()
        .rename(columns={"size": "days"})
    )
    learned = overall[~overall["method"].isin(["anchor_top1", "oracle_top10"])].copy()
    best_learned = learned.sort_values("mean_profit", ascending=False).iloc[0]
    anchor = overall[overall["method"] == "anchor_top1"].iloc[0]

    lines = [
        "# Current Prior Top-K Pair Reranker Fold-Safe NWP Residual",
        "",
        "## Scope",
        "",
        "- Anchor and candidate set are unchanged from the current prior top-K reranker.",
        "- Candidate set: champion-prior top10 legal charge/discharge pairs per day.",
        "- Difference from the original reranker: NWP month-hour residual means are fitted "
        "outside each outer held-out fold.",
        "- Fallback when held-out months are absent from training side: hour-of-day mean, then "
        "global training mean.",
        "- This is an outer fold-safe leakage check, not a submit strategy.",
        "",
        "## Overall",
        "",
        markdown_table(overall, floatfmt=".4f"),
        "",
    ]
    if not comparison.empty:
        lines += [
            "## Original vs Fold-Safe",
            "",
            markdown_table(comparison, floatfmt=".4f"),
            "",
        ]
    lines += [
        "## Regime Summary",
        "",
        markdown_table(by_regime, floatfmt=".4f"),
        "",
        "## Fold Summary",
        "",
        markdown_table(by_fold, floatfmt=".4f"),
        "",
        "## Selected Rank Counts",
        "",
        markdown_table(rank_counts, floatfmt=".4f"),
        "",
        "## Chosen Thresholds",
        "",
        markdown_table(params, floatfmt=".4f"),
        "",
        "## Decision",
        "",
    ]
    if float(best_learned["profit_delta_vs_anchor"]) > 100:
        lines += [
            "- Fold-safe reranker still beats anchor by more than 100/day.",
            "- Next step: inspect changed days and consider a conservative submit-time gate.",
        ]
    elif float(best_learned["mean_profit"]) > float(anchor["mean_profit"]):
        lines += [
            "- Fold-safe reranker remains positive but too small for submission.",
            "- Keep as diagnostic evidence only.",
        ]
    else:
        lines += [
            "- Fold-safe learned rerankers do not beat anchor top1.",
            "- Do not submit a learned reranker from this experiment.",
        ]
    lines += [
        "",
        "## Artifacts",
        "",
        f"- `{DATASET_PATH}` rows: `{len(dataset)}`",
        f"- `{SELECTED_PATH}` rows: `{len(selected)}`",
        f"- `{FOLD_SUMMARY_PATH}`",
        f"- `{REGIME_SUMMARY_PATH}`",
        f"- `{OVERALL_PATH}`",
        f"- `{PARAMS_PATH}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    cfg = base.load_config()
    candidates = pd.read_csv(base.CANDIDATE_PATH)
    candidates = base._add_candidate_features(candidates)
    raw_df = _raw_nwp_feature_frame(cfg)

    dataset, selected, by_fold, params = crossfold_evaluate_foldsafe(
        raw_df=raw_df,
        cfg=cfg,
        candidates=candidates,
    )
    by_regime = base.summarize_regimes(selected)
    overall = base.summarize_overall(by_regime)

    dataset.to_csv(DATASET_PATH, index=False)
    selected.to_csv(SELECTED_PATH, index=False)
    by_fold.to_csv(FOLD_SUMMARY_PATH, index=False)
    by_regime.to_csv(REGIME_SUMMARY_PATH, index=False)
    overall.to_csv(OVERALL_PATH, index=False)
    params.to_csv(PARAMS_PATH, index=False)
    write_report(
        dataset=dataset,
        selected=selected,
        by_fold=by_fold,
        by_regime=by_regime,
        overall=overall,
        params=params,
    )
    print(overall.to_string(index=False))
    print(f"markdown_path={OUT_MD}")


if __name__ == "__main__":
    main()
