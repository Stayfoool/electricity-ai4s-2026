from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from electricity.data import load_train_frame
from electricity.features import build_feature_frame
from electricity.features.bid_space import markdown_table

CONFIG_PATH = Path("configs/lgb_pair_spread_bidspace_weather_resid_5fold.yaml")
CANDIDATE_PATH = Path("reports/current_prior_topk_anchor_limited_candidates.csv")
REPORTS_DIR = Path("reports")
DATASET_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_dataset.csv"
SELECTED_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_selected_daily.csv"
FOLD_SUMMARY_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_by_fold.csv"
REGIME_SUMMARY_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_by_regime.csv"
OVERALL_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_overall.csv"
PARAMS_PATH = REPORTS_DIR / "current_prior_topk_pair_reranker_params.csv"
OUT_MD = REPORTS_DIR / "current_prior_topk_pair_reranker_diagnostics.md"

BLOCK_SIZE = 8
POINTS_PER_DAY = 96
THRESHOLD_GRID = [-1_000_000.0, 0.0, 100.0, 250.0, 500.0, 750.0, 1000.0]
MODEL_TARGETS = ["true_profit", "lift"]

BASE_FEATURES = [
    "candidate_rank",
    "anchor_predicted_spread",
    "anchor_spread_delta_from_top1",
    "support_count",
    "rank_score",
    "mean_rank",
    "support_present_cols_sum",
    "lgb_segmented_6_last_180d_rank",
    "lgb_segmented_6_last_180d_predicted_spread",
    "lgb_segmented_6_last_180d_present",
    "lgb_segmented_6_margin_core_last_180d_rank",
    "lgb_segmented_6_margin_core_last_180d_predicted_spread",
    "lgb_segmented_6_margin_core_last_180d_present",
    "charge_start",
    "discharge_start",
    "charge_end",
    "discharge_end",
    "charge_hour",
    "discharge_hour",
    "gap_slots",
    "gap_hours",
    "charge_delta_from_top1",
    "discharge_delta_from_top1",
    "same_charge_as_top1",
    "same_discharge_as_top1",
    "charge_start_sin",
    "charge_start_cos",
    "discharge_start_sin",
    "discharge_start_cos",
    "gap_sin",
    "gap_cos",
    "month",
    "dayofweek",
    "is_weekend",
    "top1_top2_gap",
    "top1_top5_mean_gap",
    "top5_spread_std",
]


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _window_means(values: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    return (prefix[block_size:] - prefix[:-block_size]) / block_size


def _rank_pct(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(pct=True).to_numpy(dtype=float)


def _date_slot_frame(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    time_col = cfg["data"]["time_col"]
    df = build_feature_frame(load_train_frame(cfg), cfg, time_col=time_col)
    point_cols = list(cfg["model"]["pair_point_feature_cols"])
    missing = sorted(set(point_cols) - set(df.columns))
    if missing:
        raise ValueError(f"feature frame missing pair_point_feature_cols: {missing}")
    out = df[[time_col, *point_cols]].copy()
    out["date"] = out[time_col].dt.normalize()
    out["slot"] = out[time_col].dt.hour * 4 + out[time_col].dt.minute // 15
    out = out.sort_values(["date", "slot"]).reset_index(drop=True)
    return out, point_cols


def _day_feature_maps(slot_df: pd.DataFrame, point_cols: list[str]) -> dict[pd.Timestamp, dict]:
    maps: dict[pd.Timestamp, dict] = {}
    for date, group in slot_df.groupby("date", sort=True):
        if len(group) != POINTS_PER_DAY:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(POINTS_PER_DAY)):
            continue
        col_maps = {}
        for col in point_cols:
            values = group[col].to_numpy(dtype=float)
            windows = _window_means(values)
            col_maps[col] = {
                "window": windows,
                "rank": _rank_pct(windows),
                "day_mean": float(np.mean(values)),
                "day_std": float(np.std(values)),
                "day_range": float(np.max(values) - np.min(values)),
            }
        maps[pd.Timestamp(date)] = col_maps
    return maps


def _add_candidate_features(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["candidate_rank"] = out["anchor_rank"].astype(int)
    out["charge_end"] = out["charge_start"] + BLOCK_SIZE - 1
    out["discharge_end"] = out["discharge_start"] + BLOCK_SIZE - 1
    out["charge_hour"] = out["charge_start"] / 4.0
    out["discharge_hour"] = out["discharge_start"] / 4.0
    out["gap_slots"] = out["discharge_start"] - out["charge_start"]
    out["gap_hours"] = out["gap_slots"] / 4.0
    out["month"] = out["date"].dt.month
    out["dayofweek"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["charge_start_sin"] = np.sin(2 * np.pi * out["charge_start"] / POINTS_PER_DAY)
    out["charge_start_cos"] = np.cos(2 * np.pi * out["charge_start"] / POINTS_PER_DAY)
    out["discharge_start_sin"] = np.sin(
        2 * np.pi * out["discharge_start"] / POINTS_PER_DAY
    )
    out["discharge_start_cos"] = np.cos(
        2 * np.pi * out["discharge_start"] / POINTS_PER_DAY
    )
    out["gap_sin"] = np.sin(2 * np.pi * out["gap_slots"] / POINTS_PER_DAY)
    out["gap_cos"] = np.cos(2 * np.pi * out["gap_slots"] / POINTS_PER_DAY)

    top1 = out[out["candidate_rank"] == 1][
        [
            "fold",
            "date",
            "charge_start",
            "discharge_start",
            "anchor_predicted_spread",
            "true_profit",
        ]
    ].rename(
        columns={
            "charge_start": "top1_charge_start",
            "discharge_start": "top1_discharge_start",
            "anchor_predicted_spread": "top1_anchor_predicted_spread",
            "true_profit": "top1_true_profit",
        }
    )
    out = out.merge(top1, on=["fold", "date"], how="left")
    out["charge_delta_from_top1"] = (
        out["charge_start"] - out["top1_charge_start"]
    ).abs()
    out["discharge_delta_from_top1"] = (
        out["discharge_start"] - out["top1_discharge_start"]
    ).abs()
    out["same_charge_as_top1"] = (out["charge_delta_from_top1"] == 0).astype(int)
    out["same_discharge_as_top1"] = (out["discharge_delta_from_top1"] == 0).astype(int)
    out["target_lift"] = out["true_profit"] - out["top1_true_profit"]

    rank_stats = (
        out.groupby(["fold", "date"], as_index=False)
        .agg(
            top1_top2_gap=(
                "anchor_predicted_spread",
                lambda s: float(s.iloc[0] - s.iloc[1]) if len(s) > 1 else 0.0,
            ),
            top1_top5_mean_gap=(
                "anchor_predicted_spread",
                lambda s: float(s.iloc[0] - s.head(5).mean()),
            ),
            top5_spread_std=("anchor_predicted_spread", lambda s: float(s.head(5).std(ddof=0))),
        )
    )
    return out.merge(rank_stats, on=["fold", "date"], how="left")


def _add_pair_signal_features(
    candidates: pd.DataFrame,
    day_maps: dict[pd.Timestamp, dict],
    point_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in candidates.itertuples(index=False):
        date = pd.Timestamp(row.date)
        if date not in day_maps:
            raise ValueError(f"missing point features for date={date.date()}")
        charge_start = int(row.charge_start)
        discharge_start = int(row.discharge_start)
        feature_row: dict[str, float] = {}
        for col in point_cols:
            col_map = day_maps[date][col]
            windows = col_map["window"]
            ranks = col_map["rank"]
            charge = float(windows[charge_start])
            discharge = float(windows[discharge_start])
            feature_row[f"charge_{col}_mean"] = charge
            feature_row[f"discharge_{col}_mean"] = discharge
            feature_row[f"diff_{col}_mean"] = discharge - charge
            feature_row[f"charge_{col}_rank_pct"] = float(ranks[charge_start])
            feature_row[f"discharge_{col}_rank_pct"] = float(ranks[discharge_start])
            feature_row[f"diff_{col}_rank_pct"] = (
                feature_row[f"discharge_{col}_rank_pct"]
                - feature_row[f"charge_{col}_rank_pct"]
            )
            feature_row[f"day_{col}_mean"] = col_map["day_mean"]
            feature_row[f"day_{col}_std"] = col_map["day_std"]
            feature_row[f"day_{col}_range"] = col_map["day_range"]
        rows.append(feature_row)
    return pd.concat([candidates.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def build_dataset() -> pd.DataFrame:
    cfg = load_config()
    candidates = pd.read_csv(CANDIDATE_PATH)
    candidates = _add_candidate_features(candidates)
    slot_df, point_cols = _date_slot_frame(cfg)
    day_maps = _day_feature_maps(slot_df, point_cols)
    dataset = _add_pair_signal_features(candidates, day_maps, point_cols)
    return dataset.sort_values(["fold", "date", "candidate_rank"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {
        "model",
        "fold",
        "date",
        "true_profit",
        "oracle_profit",
        "regret_vs_oracle",
        "is_oracle_pair",
        "top1_true_profit",
        "target_lift",
        "target",
    }
    cols = [
        col
        for col in df.columns
        if col not in blocked and pd.api.types.is_numeric_dtype(df[col])
    ]
    return cols


def fit_model(train: pd.DataFrame, features: list[str], *, target: str) -> Pipeline:
    regressor = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=10,
        random_state=2026,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )
    y_col = "true_profit" if target == "true_profit" else "target_lift"
    model.fit(train[features], train[y_col])
    return model


def _score_delta(df: pd.DataFrame, *, target: str, score_col: str) -> pd.Series:
    if target == "lift":
        return df[score_col]
    top1 = df[df["candidate_rank"] == 1][["date", score_col]].rename(
        columns={score_col: "__top1_score"}
    )
    scored = df[["date", score_col]].merge(top1, on="date", how="left")
    return scored[score_col] - scored["__top1_score"]


def select_daily(
    df: pd.DataFrame,
    *,
    score_col: str,
    target: str,
    threshold: float,
    method: str,
) -> pd.DataFrame:
    selected_rows = []
    for _, group in df.groupby("date", sort=True):
        group = group.copy()
        group["score_delta_vs_top1"] = _score_delta(
            group,
            target=target,
            score_col=score_col,
        ).to_numpy(dtype=float)
        top1 = group[group["candidate_rank"] == 1].iloc[0]
        best = group.sort_values(
            [score_col, "anchor_predicted_spread", "candidate_rank"],
            ascending=[False, False, True],
        ).iloc[0]
        if int(best["candidate_rank"]) != 1 and float(best["score_delta_vs_top1"]) >= threshold:
            selected = best
        else:
            selected = top1
        selected_rows.append(selected)
    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    out["method"] = method
    out["threshold"] = threshold
    out["lift_vs_anchor"] = out["true_profit"] - out["top1_true_profit"]
    out["changed_from_anchor"] = (out["candidate_rank"] != 1).astype(int)
    out["miss_vs_oracle_top10"] = (
        out.groupby("date")["true_profit"].transform("max") - out["true_profit"]
    )
    return out


def summarize_selection(selected: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in selected.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        profit = group["true_profit"].astype(float)
        lift = group["lift_vs_anchor"].astype(float)
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "days": len(group),
                "mean_profit": float(profit.mean()),
                "worst_profit": float(profit.min()),
                "loss_days": int((profit < 0).sum()),
                "mean_lift_vs_anchor": float(lift.mean()),
                "positive_lift_days": int((lift > 0).sum()),
                "negative_lift_days": int((lift < 0).sum()),
                "changed_days": int(group["changed_from_anchor"].sum()),
                "mean_selected_rank": float(group["candidate_rank"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _best_threshold_from_oof(
    oof: pd.DataFrame,
    *,
    target: str,
    score_col: str,
) -> float:
    rows = []
    for threshold in THRESHOLD_GRID:
        selected = select_daily(
            oof,
            score_col=score_col,
            target=target,
            threshold=threshold,
            method="inner",
        )
        profit = selected["true_profit"].astype(float)
        rows.append(
            {
                "threshold": threshold,
                "mean_profit": float(profit.mean()),
                "loss_days": int((profit < 0).sum()),
                "changed_days": int(selected["changed_from_anchor"].sum()),
            }
        )
    scored = pd.DataFrame(rows)
    best = scored.sort_values(
        ["mean_profit", "loss_days", "changed_days", "threshold"],
        ascending=[False, True, True, False],
    ).iloc[0]
    return float(best["threshold"])


def _inner_oof_predictions(
    train: pd.DataFrame,
    *,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    frames = []
    for inner_fold in sorted(train["fold"].unique()):
        fit = train[train["fold"] != inner_fold].copy()
        valid = train[train["fold"] == inner_fold].copy()
        if fit.empty or valid.empty:
            continue
        model = fit_model(fit, features, target=target)
        valid = valid.copy()
        valid["rerank_score"] = model.predict(valid[features])
        frames.append(valid)
    if not frames:
        raise ValueError("inner OOF prediction produced no frames")
    return pd.concat(frames, ignore_index=True)


def crossfold_evaluate(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = feature_columns(dataset)
    selected_frames = []
    param_rows = []
    for heldout_fold in sorted(dataset["fold"].unique()):
        train = dataset[dataset["fold"] != heldout_fold].copy()
        valid = dataset[dataset["fold"] == heldout_fold].copy()

        anchor = select_daily(
            valid.assign(anchor_score=-valid["candidate_rank"]),
            score_col="anchor_score",
            target="lift",
            threshold=0.0,
            method="anchor_top1",
        )
        oracle = select_daily(
            valid.assign(oracle_score=valid["true_profit"]),
            score_col="oracle_score",
            target="true_profit",
            threshold=-1_000_000.0,
            method="oracle_top10",
        )
        selected_frames.extend([anchor, oracle])

        for target in MODEL_TARGETS:
            oof = _inner_oof_predictions(train, features=features, target=target)
            threshold = _best_threshold_from_oof(
                oof,
                target=target,
                score_col="rerank_score",
            )
            model = fit_model(train, features, target=target)
            scored_valid = valid.copy()
            scored_valid["rerank_score"] = model.predict(scored_valid[features])
            method = f"gbr_{target}"
            selected = select_daily(
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

    selected_all = pd.concat(selected_frames, ignore_index=True)
    selected_all["date"] = pd.to_datetime(selected_all["date"])
    by_fold = summarize_selection(selected_all, group_cols=["fold", "method"])
    return selected_all, by_fold, pd.DataFrame(param_rows)


def _regime_mask(df: pd.DataFrame, regime: str) -> pd.Series:
    folds = df["fold"].astype(str)
    dates = pd.to_datetime(df["date"])
    if regime == "standard_09_12":
        return folds.isin(["valid_2025_09", "valid_2025_10", "valid_2025_11", "valid_2025_12"])
    if regime == "jan_feb_like":
        return folds == "valid_2025_jan_feb"
    if regime == "all_5fold":
        return pd.Series(True, index=df.index)
    if regime == "winter_11_12_jan_feb":
        return (folds == "valid_2025_jan_feb") | dates.dt.month.isin([11, 12])
    raise ValueError(f"unknown regime={regime}")


def summarize_regimes(selected: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for regime in ["standard_09_12", "jan_feb_like", "winter_11_12_jan_feb", "all_5fold"]:
        part = selected[_regime_mask(selected, regime)].copy()
        summary = summarize_selection(part, group_cols=["method"])
        summary.insert(0, "regime", regime)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def summarize_overall(by_regime: pd.DataFrame) -> pd.DataFrame:
    all_5fold = by_regime[by_regime["regime"] == "all_5fold"].copy()
    anchor_profit = float(
        all_5fold.loc[all_5fold["method"] == "anchor_top1", "mean_profit"].iloc[0]
    )
    anchor_loss = int(all_5fold.loc[all_5fold["method"] == "anchor_top1", "loss_days"].iloc[0])
    all_5fold["profit_delta_vs_anchor"] = all_5fold["mean_profit"] - anchor_profit
    all_5fold["loss_day_delta_vs_anchor"] = all_5fold["loss_days"] - anchor_loss
    return all_5fold.sort_values("mean_profit", ascending=False)


def write_report(
    *,
    dataset: pd.DataFrame,
    selected: pd.DataFrame,
    by_fold: pd.DataFrame,
    by_regime: pd.DataFrame,
    overall: pd.DataFrame,
    params: pd.DataFrame,
) -> None:
    best = overall.iloc[0]
    anchor = overall[overall["method"] == "anchor_top1"].iloc[0]
    rank_counts = (
        selected.groupby(["method", "candidate_rank"], as_index=False)
        .size()
        .rename(columns={"size": "days"})
    )
    lines = [
        "# Current Prior Top-K Pair Reranker",
        "",
        "## Scope",
        "",
        "- Anchor: `ens_champion_segmented6_prior` current champion with dispatch prior.",
        "- Candidate set: anchor top10 legal charge/discharge pairs.",
        "- Added features: bid_space window means/ranks, selected weather month-hour residuals, "
        "candidate timing, top1 gap/confidence, and support from two existing experts.",
        "- Validation: outer held-out fold; threshold selected by inner out-of-fold predictions.",
        "",
        "## Overall",
        "",
        markdown_table(overall, floatfmt=".4f"),
        "",
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
    if best["method"] != "anchor_top1" and float(best["profit_delta_vs_anchor"]) > 100:
        lines += [
            "- A learned top-K reranker beats anchor top1 by more than 100/day on all_5fold.",
            "- Next step: inspect fold/regime stability, then implement submit-time rerank.",
        ]
    elif best["method"] != "anchor_top1" and float(best["mean_profit"]) > float(
        anchor["mean_profit"]
    ):
        lines += [
            "- A learned top-K reranker is positive but small.",
            "- Do not submit yet unless Jan-Feb-like and loss-day behavior are also favorable.",
        ]
    else:
        lines += [
            "- Learned top-K rerankers do not beat anchor top1.",
            "- Keep this as diagnostic evidence; avoid changing submit strategy from this result.",
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
    dataset = build_dataset()
    selected, by_fold, params = crossfold_evaluate(dataset)
    by_regime = summarize_regimes(selected)
    overall = summarize_overall(by_regime)

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
