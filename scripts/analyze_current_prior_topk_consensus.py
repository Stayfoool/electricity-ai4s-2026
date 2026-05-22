from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity.data import load_train_frame
from electricity.dispatch import rank_day_pairs
from electricity.eval.backtest import (
    _add_sample_weight,
    _add_target_variants,
    _fold_train_frame,
    _maybe_build_prior,
    _model_spec_frame,
    _predict_member_fold,
    _target_col_for_mode,
)
from electricity.eval.segmented_utils import (
    add_slot_and_segment,
    predict_segmented_fold,
    segment_boundaries,
    segmented_feature_build_spec,
    segmented_feature_columns,
)
from electricity.features import (
    apply_fold_bias_correction,
    apply_fold_weather_correction,
    build_feature_frame,
    feature_columns,
)
from electricity.models import predict_model, train_model

REPORTS_DIR = Path("reports")
ANCHOR_MODEL = "ens_champion_segmented6_prior"
EXPERT_CONFIGS = {
    ANCHOR_MODEL: Path("configs/ensemble_champion_segmented6_prior_5fold.yaml"),
    "lgb_segmented_6_last_180d": Path("configs/lgb_segmented_6_last_180d.yaml"),
    "lgb_segmented_6_margin_core_last_180d": Path(
        "configs/lgb_segmented_6_margin_core_last_180d.yaml"
    ),
}
TOP_K = 10
BLOCK_SIZE = 8
POWER = 1000.0
RULES = [
    "anchor_top1",
    "support_then_anchor_rank",
    "support2_then_anchor_rank",
    "unanimous_then_anchor_rank",
    "support_score",
    "anchor_rank_le5_support2",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pair_profit(true_prices: np.ndarray, charge_start: int, discharge_start: int) -> float:
    charge_sum = true_prices[charge_start : charge_start + BLOCK_SIZE].sum()
    discharge_sum = true_prices[discharge_start : discharge_start + BLOCK_SIZE].sum()
    return float((discharge_sum - charge_sum) * POWER)


def _with_anchor_folds(cfg: dict, anchor_cfg: dict) -> dict:
    out = dict(cfg)
    out["folds"] = anchor_cfg["folds"]
    return out


def _prepare_ensemble_specs(cfg: dict) -> list[dict]:
    specs = []
    for raw_spec in cfg["ensemble"]["members"]:
        spec = dict(raw_spec)
        spec.setdefault("paths", cfg["paths"])
        if "sample_weighting" in cfg:
            spec.setdefault("sample_weighting", cfg["sample_weighting"])
        if "weekly_relative_features" in cfg:
            spec.setdefault("weekly_relative_features", cfg["weekly_relative_features"])
        specs.append(spec)
    return specs


def predict_ensemble_config(cfg: dict) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    specs = _prepare_ensemble_specs(cfg)
    weights = np.asarray([float(spec.get("weight", 1.0)) for spec in specs], dtype=float)
    weights = weights / weights.sum()

    raw_df = load_train_frame(cfg)
    member_frames = [
        _model_spec_frame(raw_df, spec, time_col=time_col, target_col=target_col)
        for spec in specs
    ]
    member_feature_cols = [feature_columns(spec) for spec in specs]
    member_target_cols = [
        _target_col_for_mode(spec["model"].get("target_mode", "absolute")) for spec in specs
    ]

    rows: list[pd.DataFrame] = []
    for fold in cfg["folds"]:
        preds: list[np.ndarray] = []
        valid_df_for_eval: pd.DataFrame | None = None
        for spec, frame, cols, train_target_col in zip(
            specs, member_frames, member_feature_cols, member_target_cols, strict=True
        ):
            fold_train_end = pd.Timestamp(fold["train_end"])
            fold_frame = apply_fold_bias_correction(
                frame,
                spec,
                time_col=time_col,
                train_end=fold_train_end,
            )
            fold_frame = apply_fold_weather_correction(
                fold_frame,
                spec,
                time_col=time_col,
                train_end=fold_train_end,
            )
            pred, valid_df, _, _ = _predict_member_fold(
                spec=spec,
                frame=fold_frame,
                feature_cols=cols,
                train_target_col=train_target_col,
                time_col=time_col,
                fold=fold,
            )
            preds.append(pred)
            if valid_df_for_eval is None:
                valid_df_for_eval = valid_df[[time_col, target_col]].copy()
        if valid_df_for_eval is None:
            raise ValueError(f"no validation frame for fold={fold['name']}")
        valid_df_for_eval["pred"] = np.average(np.vstack(preds), axis=0, weights=weights)
        valid_df_for_eval["fold"] = fold["name"]
        rows.append(valid_df_for_eval)
    return pd.concat(rows, ignore_index=True)


def predict_single_config(cfg: dict) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    target_mode = cfg["model"].get("target_mode", "absolute")
    train_target_col = _target_col_for_mode(target_mode)
    train_window_days = cfg["model"].get("train_window_days")
    boundaries = cfg["model"].get("segment_boundaries")
    build_spec = segmented_feature_build_spec(cfg)
    cols = feature_columns(build_spec)

    raw_df = load_train_frame(cfg)
    df = _add_target_variants(
        build_feature_frame(raw_df, build_spec, time_col=time_col),
        time_col=time_col,
        target_col=target_col,
    )
    if boundaries is not None:
        boundary_values = segment_boundaries(cfg)
        df = add_slot_and_segment(df, time_col=time_col, boundaries=boundary_values)
        feature_cols_by_segment = segmented_feature_columns(cfg, boundary_values)

    rows: list[pd.DataFrame] = []
    for fold in cfg["folds"]:
        train_end = pd.Timestamp(fold["train_end"])
        valid_start = pd.Timestamp(fold["valid_start"])
        valid_end = pd.Timestamp(fold["valid_end"])
        fold_train_start = (
            pd.Timestamp(fold["train_start"]) if fold.get("train_start") else None
        )
        fold_df = apply_fold_bias_correction(df, cfg, time_col=time_col, train_end=train_end)
        fold_df = apply_fold_weather_correction(
            fold_df,
            cfg,
            time_col=time_col,
            train_end=train_end,
        )
        train_df = _fold_train_frame(
            fold_df,
            time_col=time_col,
            train_end=train_end,
            train_window_days=train_window_days,
            train_start=fold_train_start,
            valid_start=valid_start,
            valid_end=valid_end,
        )
        valid_df = fold_df[
            (fold_df[time_col] >= valid_start) & (fold_df[time_col] <= valid_end)
        ].copy()
        if train_df.empty or valid_df.empty:
            raise ValueError(f"empty train/valid split for fold={fold['name']}")
        train_df = _add_sample_weight(train_df, cfg, time_col=time_col)
        valid_df = valid_df.reset_index(drop=True)

        if boundaries is not None:
            preds, _, _ = predict_segmented_fold(
                train_df,
                valid_df,
                feature_cols_by_segment=feature_cols_by_segment,
                target_col=train_target_col,
                cfg=cfg,
                boundaries=boundary_values,
            )
        else:
            model = train_model(
                train_df,
                valid_df,
                feature_cols=cols,
                target_col=train_target_col,
                cfg=cfg,
            )
            preds = predict_model(model, valid_df, cols, cfg)

        out = valid_df[[time_col, target_col]].copy()
        out["pred"] = preds
        out["fold"] = fold["name"]
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def predict_config(cfg: dict) -> pd.DataFrame:
    if "ensemble" in cfg:
        return predict_ensemble_config(cfg)
    return predict_single_config(cfg)


def fold_priors(
    anchor_cfg: dict,
) -> dict[str, tuple[np.ndarray | None, np.ndarray | None, float, float]]:
    time_col = anchor_cfg["data"]["time_col"]
    target_col = anchor_cfg["data"]["target_col"]
    raw_df = load_train_frame(anchor_cfg)
    out = {}
    for fold in anchor_cfg["folds"]:
        out[fold["name"]] = _maybe_build_prior(
            anchor_cfg,
            raw_df,
            time_col=time_col,
            target_col=target_col,
            train_end=pd.Timestamp(fold["train_end"]),
            valid_start=pd.Timestamp(fold["valid_start"]),
            valid_end=pd.Timestamp(fold["valid_end"]),
        )
    return out


def rank_with_prior(
    prices: np.ndarray,
    prior: tuple[np.ndarray | None, np.ndarray | None, float, float],
    *,
    top_k: int | None = TOP_K,
) -> list[tuple[float, int, int]]:
    log_pc, log_pd, lc_w, ld_w = prior
    return rank_day_pairs(
        prices,
        top_k=top_k,
        log_prior_charge=log_pc,
        log_prior_discharge=log_pd,
        lambda_charge=lc_w,
        lambda_discharge=ld_w,
    )


def build_model_day_candidates(
    model: str,
    frame: pd.DataFrame,
    priors: dict[str, tuple[np.ndarray | None, np.ndarray | None, float, float]],
    *,
    time_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["date"] = work[time_col].dt.normalize()
    work["slot"] = work[time_col].dt.hour * 4 + work[time_col].dt.minute // 15

    candidate_rows: list[dict[str, object]] = []
    day_rows: list[dict[str, object]] = []
    for (fold, date), group in work.groupby(["fold", "date"], sort=True):
        if len(group) != 96:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            continue

        true_prices = group[target_col].to_numpy(dtype=float)
        pred_prices = group["pred"].to_numpy(dtype=float)
        ranked = rank_with_prior(pred_prices, priors[str(fold)], top_k=TOP_K)
        true_ranked = rank_day_pairs(true_prices)
        oracle_spread, oracle_charge, oracle_discharge = true_ranked[0]
        oracle_profit = oracle_spread * POWER

        rows_for_day: list[dict[str, object]] = []
        for rank, (pred_spread, charge_start, discharge_start) in enumerate(
            ranked,
            start=1,
        ):
            true_profit = pair_profit(true_prices, charge_start, discharge_start)
            row = {
                "model": model,
                "fold": fold,
                "date": date.date().isoformat(),
                "candidate_rank": rank,
                "charge_start": int(charge_start),
                "discharge_start": int(discharge_start),
                "predicted_spread": float(pred_spread),
                "true_profit": true_profit,
                "oracle_profit": float(oracle_profit),
                "regret_vs_oracle": float(oracle_profit - true_profit),
                "is_oracle_pair": bool(
                    charge_start == oracle_charge and discharge_start == oracle_discharge
                ),
            }
            candidate_rows.append(row)
            rows_for_day.append(row)

        top1 = rows_for_day[0]
        topk_best = max(rows_for_day, key=lambda row: float(row["true_profit"]))
        day_rows.append(
            {
                "model": model,
                "fold": fold,
                "date": date.date().isoformat(),
                "top1_profit": float(top1["true_profit"]),
                "top10_best_profit": float(topk_best["true_profit"]),
                "top10_lift_vs_top1": float(topk_best["true_profit"])
                - float(top1["true_profit"]),
                "top10_best_rank": int(topk_best["candidate_rank"]),
                "oracle_profit": float(oracle_profit),
                "oracle_in_top10": any(bool(row["is_oracle_pair"]) for row in rows_for_day),
                "top1_is_oracle": bool(top1["is_oracle_pair"]),
            }
        )
    return pd.DataFrame(candidate_rows), pd.DataFrame(day_rows)


def build_anchor_limited_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    anchor = candidate_df[candidate_df["model"] == ANCHOR_MODEL].copy()
    anchor = anchor.rename(
        columns={
            "candidate_rank": "anchor_rank",
            "predicted_spread": "anchor_predicted_spread",
        }
    )
    index_cols = ["fold", "date", "charge_start", "discharge_start"]
    top1 = anchor[anchor["anchor_rank"] == 1][
        ["fold", "date", "anchor_predicted_spread"]
    ].rename(columns={"anchor_predicted_spread": "anchor_top1_spread"})
    out = anchor.merge(top1, on=["fold", "date"], how="left")
    out["anchor_spread_delta_from_top1"] = (
        out["anchor_top1_spread"] - out["anchor_predicted_spread"]
    )

    support_models = [model for model in EXPERT_CONFIGS if model != ANCHOR_MODEL]
    out["support_count"] = 1
    out["rank_score"] = TOP_K + 1 - out["anchor_rank"]
    support_rank_cols = []
    support_present_cols = []

    for model in support_models:
        support = candidate_df[candidate_df["model"] == model][
            [*index_cols, "candidate_rank", "predicted_spread"]
        ].rename(
            columns={
                "candidate_rank": f"{model}_rank",
                "predicted_spread": f"{model}_predicted_spread",
            }
        )
        out = out.merge(support, on=index_cols, how="left")
        present_col = f"{model}_present"
        rank_col = f"{model}_rank"
        out[present_col] = out[rank_col].notna().astype(int)
        out["support_count"] += out[present_col]
        out["rank_score"] += np.where(out[rank_col].notna(), TOP_K + 1 - out[rank_col], 0)
        support_rank_cols.append(rank_col)
        support_present_cols.append(present_col)

    rank_cols = ["anchor_rank", *support_rank_cols]
    out["mean_rank"] = out[rank_cols].mean(axis=1, skipna=True)
    out["support2"] = out["support_count"] >= 2
    out["unanimous"] = out["support_count"] == len(EXPERT_CONFIGS)
    out["support_present_cols_sum"] = out[support_present_cols].sum(axis=1)
    return out


def select_first(group: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> pd.Series:
    return group.sort_values(sort_cols, ascending=ascending).iloc[0]


def select_for_rule(group: pd.DataFrame, rule: str) -> pd.Series:
    anchor_top1 = group[group["anchor_rank"] == 1].iloc[0]
    if rule == "anchor_top1":
        return anchor_top1

    if rule == "support_then_anchor_rank":
        return select_first(
            group,
            ["support_count", "anchor_rank", "mean_rank", "anchor_spread_delta_from_top1"],
            [False, True, True, True],
        )

    if rule == "support2_then_anchor_rank":
        supported = group[group["support_count"] >= 2]
        if supported.empty:
            return anchor_top1
        return select_first(
            supported,
            ["anchor_rank", "support_count", "mean_rank", "anchor_spread_delta_from_top1"],
            [True, False, True, True],
        )

    if rule == "unanimous_then_anchor_rank":
        unanimous = group[group["unanimous"]]
        if unanimous.empty:
            return anchor_top1
        return select_first(
            unanimous,
            ["anchor_rank", "mean_rank", "anchor_spread_delta_from_top1"],
            [True, True, True],
        )

    if rule == "support_score":
        return select_first(
            group,
            ["support_count", "rank_score", "anchor_rank", "anchor_spread_delta_from_top1"],
            [False, False, True, True],
        )

    if rule == "anchor_rank_le5_support2":
        supported = group[(group["anchor_rank"] <= 5) & (group["support_count"] >= 2)]
        if supported.empty:
            return anchor_top1
        return select_first(
            supported,
            ["support_count", "anchor_rank", "mean_rank", "anchor_spread_delta_from_top1"],
            [False, True, True, True],
        )

    raise ValueError(f"unknown rule={rule}")


def build_rule_daily(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (fold, date), group in candidates.groupby(["fold", "date"], sort=True):
        anchor_top1 = group[group["anchor_rank"] == 1].iloc[0]
        oracle_top10_profit = float(group["true_profit"].max())
        for rule in RULES:
            selected = select_for_rule(group, rule)
            true_profit = float(selected["true_profit"])
            rows.append(
                {
                    "fold": fold,
                    "date": date,
                    "rule": rule,
                    "charge_start": int(selected["charge_start"]),
                    "discharge_start": int(selected["discharge_start"]),
                    "anchor_rank": int(selected["anchor_rank"]),
                    "support_count": int(selected["support_count"]),
                    "rank_score": float(selected["rank_score"]),
                    "mean_rank": float(selected["mean_rank"]),
                    "anchor_predicted_spread": float(selected["anchor_predicted_spread"]),
                    "anchor_spread_delta_from_top1": float(
                        selected["anchor_spread_delta_from_top1"]
                    ),
                    "true_profit": true_profit,
                    "anchor_top1_profit": float(anchor_top1["true_profit"]),
                    "oracle_profit": float(selected["oracle_profit"]),
                    "oracle_top10_profit": oracle_top10_profit,
                    "lift_vs_anchor_top1": true_profit - float(anchor_top1["true_profit"]),
                    "miss_vs_oracle_top10": oracle_top10_profit - true_profit,
                    "changed_from_anchor": int(
                        int(selected["charge_start"]) != int(anchor_top1["charge_start"])
                        or int(selected["discharge_start"])
                        != int(anchor_top1["discharge_start"])
                    ),
                    "is_oracle_pair": bool(selected["is_oracle_pair"]),
                }
            )
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rule, group in daily.groupby("rule", sort=False):
        profit = group["true_profit"].astype(float)
        lift = group["lift_vs_anchor_top1"].astype(float)
        rows.append(
            {
                "rule": rule,
                "days": len(group),
                "mean_profit": float(profit.mean()),
                "worst_profit": float(profit.min()),
                "loss_days": int((profit < 0).sum()),
                "mean_lift_vs_anchor": float(lift.mean()),
                "positive_lift_days": int((lift > 0).sum()),
                "negative_lift_days": int((lift < 0).sum()),
                "changed_days": int(group["changed_from_anchor"].sum()),
                "mean_anchor_rank": float(group["anchor_rank"].mean()),
                "mean_support_count": float(group["support_count"].mean()),
                "mean_miss_vs_oracle_top10": float(group["miss_vs_oracle_top10"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_profit", ascending=False)


def summarize_by_fold(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (fold, rule), group in daily.groupby(["fold", "rule"], sort=True):
        profit = group["true_profit"].astype(float)
        lift = group["lift_vs_anchor_top1"].astype(float)
        rows.append(
            {
                "fold": fold,
                "rule": rule,
                "days": len(group),
                "mean_profit": float(profit.mean()),
                "worst_profit": float(profit.min()),
                "loss_days": int((profit < 0).sum()),
                "mean_lift_vs_anchor": float(lift.mean()),
                "changed_days": int(group["changed_from_anchor"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _regime_mask(df: pd.DataFrame, regime: str) -> pd.Series:
    dates = pd.to_datetime(df["date"])
    folds = df["fold"].astype(str)
    if regime == "standard_09_12":
        return folds.isin(["valid_2025_09", "valid_2025_10", "valid_2025_11", "valid_2025_12"])
    if regime == "jan_feb_like":
        return folds == "valid_2025_jan_feb"
    if regime == "winter_11_12_jan_feb":
        return (folds == "valid_2025_jan_feb") | dates.dt.month.isin([11, 12])
    if regime == "late_winter_12_jan_feb":
        return (folds == "valid_2025_jan_feb") | (dates.dt.month == 12)
    if regime == "all_5fold":
        return pd.Series(True, index=df.index)
    raise ValueError(f"unknown regime={regime}")


def summarize_regimes(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    regimes = [
        "standard_09_12",
        "jan_feb_like",
        "winter_11_12_jan_feb",
        "late_winter_12_jan_feb",
        "all_5fold",
    ]
    for regime in regimes:
        subset = daily[_regime_mask(daily, regime)]
        for rule, group in subset.groupby("rule", sort=False):
            profit = group["true_profit"].astype(float)
            lift = group["lift_vs_anchor_top1"].astype(float)
            rows.append(
                {
                    "regime": regime,
                    "rule": rule,
                    "days": len(group),
                    "mean_profit": float(profit.mean()),
                    "worst_profit": float(profit.min()),
                    "loss_days": int((profit < 0).sum()),
                    "mean_lift_vs_anchor": float(lift.mean()),
                    "changed_days": int(group["changed_from_anchor"].sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["regime", "mean_profit"], ascending=[True, False])


def summarize_topk_upper_bound(day_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in day_df.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "days": len(group),
                "top1_mean_profit": float(group["top1_profit"].mean()),
                "top10_best_mean_profit": float(group["top10_best_profit"].mean()),
                "top10_lift_mean": float(group["top10_lift_vs_top1"].mean()),
                "positive_lift_days": int((group["top10_lift_vs_top1"] > 0).sum()),
                "oracle_in_top10_rate": float(group["oracle_in_top10"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("top10_lift_mean", ascending=False)


def write_markdown(
    *,
    topk_summary: pd.DataFrame,
    rule_summary: pd.DataFrame,
    by_fold: pd.DataFrame,
    by_regime: pd.DataFrame,
) -> None:
    anchor_mean = float(
        rule_summary.loc[rule_summary["rule"] == "anchor_top1", "mean_profit"].iloc[0]
    )
    best = rule_summary.iloc[0]
    lines = [
        "# Current Prior Top-K Consensus Diagnostics",
        "",
        "## Scope",
        "",
        "Anchor model is the current online-best champion with dispatch prior:",
        "",
        f"- `{ANCHOR_MODEL}`",
        "",
        "All rules are constrained to the anchor model's predicted top-10 legal pairs. "
        "Other experts only provide support/rank signals. Fold-specific dispatch priors "
        "are rebuilt from training labels exactly as in the champion-prior backtest.",
        "",
        "## Top-K Upper Bound",
        "",
        topk_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Rule Summary",
        "",
        rule_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Fold Summary",
        "",
        by_fold.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Regime Summary",
        "",
        by_regime.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Decision",
        "",
    ]
    if float(best["mean_profit"]) > anchor_mean + 50:
        lines += [
            "- At least one anchor-limited consensus rule materially beats anchor top1.",
            "- Next step: generate a submit file for the best conservative rule and test online.",
        ]
    elif float(best["mean_profit"]) > anchor_mean:
        lines += [
            "- Anchor-limited consensus is positive but small.",
            "- Do not submit yet unless fold/regime stability is clearly favorable.",
        ]
    else:
        lines += [
            "- Anchor-limited consensus does not beat anchor top1.",
            "- Keep current champion-prior submit as the safer path.",
        ]

    (REPORTS_DIR / "current_prior_topk_consensus_diagnostics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    anchor_cfg = load_config(EXPERT_CONFIGS[ANCHOR_MODEL])
    time_col = anchor_cfg["data"]["time_col"]
    target_col = anchor_cfg["data"]["target_col"]
    priors = fold_priors(anchor_cfg)

    all_candidates: list[pd.DataFrame] = []
    all_days: list[pd.DataFrame] = []
    for model, path in EXPERT_CONFIGS.items():
        print(f"running_model={model}", flush=True)
        cfg = _with_anchor_folds(load_config(path), anchor_cfg)
        pred_df = predict_config(cfg)
        candidates, days = build_model_day_candidates(
            model,
            pred_df,
            priors,
            time_col=time_col,
            target_col=target_col,
        )
        all_candidates.append(candidates)
        all_days.append(days)

    candidate_df = pd.concat(all_candidates, ignore_index=True)
    day_df = pd.concat(all_days, ignore_index=True)
    anchor_candidates = build_anchor_limited_candidates(candidate_df)
    rule_daily = build_rule_daily(anchor_candidates)
    topk_summary = summarize_topk_upper_bound(day_df)
    rule_summary = summarize(rule_daily)
    by_fold = summarize_by_fold(rule_daily)
    by_regime = summarize_regimes(rule_daily)

    candidate_df.to_csv(REPORTS_DIR / "current_prior_topk_pair_candidates.csv", index=False)
    day_df.to_csv(REPORTS_DIR / "current_prior_topk_pair_daily.csv", index=False)
    topk_summary.to_csv(REPORTS_DIR / "current_prior_topk_pair_summary.csv", index=False)
    anchor_candidates.to_csv(
        REPORTS_DIR / "current_prior_topk_anchor_limited_candidates.csv",
        index=False,
    )
    rule_daily.to_csv(REPORTS_DIR / "current_prior_topk_consensus_daily.csv", index=False)
    rule_summary.to_csv(REPORTS_DIR / "current_prior_topk_consensus_summary.csv", index=False)
    by_fold.to_csv(REPORTS_DIR / "current_prior_topk_consensus_by_fold.csv", index=False)
    by_regime.to_csv(REPORTS_DIR / "current_prior_topk_consensus_by_regime.csv", index=False)
    write_markdown(
        topk_summary=topk_summary,
        rule_summary=rule_summary,
        by_fold=by_fold,
        by_regime=by_regime,
    )

    print(rule_summary.to_string(index=False))
    print(f"summary_path={REPORTS_DIR / 'current_prior_topk_consensus_summary.csv'}")
    print(
        "markdown_path="
        f"{REPORTS_DIR / 'current_prior_topk_consensus_diagnostics.md'}"
    )


if __name__ == "__main__":
    main()
