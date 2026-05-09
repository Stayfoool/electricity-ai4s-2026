from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity.data import load_train_frame
from electricity.dispatch import rank_day_pairs
from electricity.eval.backtest import (
    _add_target_variants,
    _fold_train_frame,
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
from electricity.features import build_feature_frame, feature_columns
from electricity.models import predict_model, train_model

REPORTS_DIR = Path("reports")
EXPERT_CONFIGS = {
    "ens_champion_segmented6": Path("configs/ensemble_champion_and_segmented_6.yaml"),
    "lgb_segmented_6_last_180d": Path("configs/lgb_segmented_6_last_180d.yaml"),
    "lgb_segmented_6_margin_core_last_180d": Path(
        "configs/lgb_segmented_6_margin_core_last_180d.yaml"
    ),
}
TOP_K_VALUES = [1, 3, 5, 10]
BLOCK_SIZE = 8
POWER = 1000.0


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pair_profit(true_prices: np.ndarray, charge_start: int, discharge_start: int) -> float:
    charge_sum = true_prices[charge_start : charge_start + BLOCK_SIZE].sum()
    discharge_sum = true_prices[discharge_start : discharge_start + BLOCK_SIZE].sum()
    return float((discharge_sum - charge_sum) * POWER)


def predict_single_config(cfg: dict) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    target_mode = cfg["model"].get("target_mode", "absolute")
    train_target_col = _target_col_for_mode(target_mode)
    train_window_days = cfg["model"].get("train_window_days")
    boundaries = cfg["model"].get("segment_boundaries")
    build_spec = segmented_feature_build_spec(cfg)
    feature_cols = feature_columns(build_spec)

    df = _add_target_variants(
        build_feature_frame(load_train_frame(cfg), build_spec, time_col=time_col),
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
        train_df = _fold_train_frame(
            df,
            time_col=time_col,
            train_end=train_end,
            train_window_days=train_window_days,
        )
        valid_df = df[(df[time_col] >= valid_start) & (df[time_col] <= valid_end)].copy()
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
                feature_cols=feature_cols,
                target_col=train_target_col,
                cfg=cfg,
            )
            preds = predict_model(model, valid_df, feature_cols, cfg)

        out = valid_df[[time_col, target_col]].copy()
        out["pred"] = preds
        out["fold"] = fold["name"]
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def predict_ensemble_config(cfg: dict) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    specs = cfg["ensemble"]["members"]
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
        for spec, frame, feature_cols, train_target_col in zip(
            specs,
            member_frames,
            member_feature_cols,
            member_target_cols,
            strict=True,
        ):
            pred, valid_df, _, _ = _predict_member_fold(
                spec=spec,
                frame=frame,
                feature_cols=feature_cols,
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


def predict_config(cfg: dict) -> pd.DataFrame:
    if "ensemble" in cfg:
        return predict_ensemble_config(cfg)
    return predict_single_config(cfg)


def analyze_day(
    *,
    model: str,
    fold: str,
    date: pd.Timestamp,
    true_prices: np.ndarray,
    pred_prices: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    predicted_top10 = rank_day_pairs(pred_prices, top_k=10)
    true_ranked = rank_day_pairs(true_prices)
    oracle_spread, oracle_charge, oracle_discharge = true_ranked[0]
    oracle_profit = oracle_spread * POWER

    candidate_rows: list[dict[str, object]] = []
    for rank, (pred_spread, charge_start, discharge_start) in enumerate(predicted_top10, start=1):
        true_profit = pair_profit(true_prices, charge_start, discharge_start)
        candidate_rows.append(
            {
                "model": model,
                "fold": fold,
                "date": date.date().isoformat(),
                "candidate_rank": rank,
                "charge_start": charge_start,
                "discharge_start": discharge_start,
                "predicted_spread": pred_spread,
                "true_profit": true_profit,
                "regret_vs_oracle": oracle_profit - true_profit,
                "is_oracle_pair": (
                    charge_start == oracle_charge and discharge_start == oracle_discharge
                ),
            }
        )

    top1 = candidate_rows[0]
    day_rows: list[dict[str, object]] = []
    for top_k in TOP_K_VALUES:
        subset = candidate_rows[:top_k]
        best = max(subset, key=lambda row: float(row["true_profit"]))
        oracle_in_topk = any(bool(row["is_oracle_pair"]) for row in subset)
        day_rows.append(
            {
                "model": model,
                "fold": fold,
                "date": date.date().isoformat(),
                "top_k": top_k,
                "top1_profit": top1["true_profit"],
                "topk_best_profit": best["true_profit"],
                "topk_lift_vs_top1": float(best["true_profit"]) - float(top1["true_profit"]),
                "topk_best_rank": best["candidate_rank"],
                "topk_best_charge_start": best["charge_start"],
                "topk_best_discharge_start": best["discharge_start"],
                "oracle_profit": oracle_profit,
                "oracle_charge_start": oracle_charge,
                "oracle_discharge_start": oracle_discharge,
                "oracle_in_topk": oracle_in_topk,
                "top1_is_oracle": bool(top1["is_oracle_pair"]),
                "top1_regret": oracle_profit - float(top1["true_profit"]),
                "topk_best_regret": oracle_profit - float(best["true_profit"]),
            }
        )
    return candidate_rows, day_rows


def analyze_model(model: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    pred_df = predict_config(cfg)
    pred_df["date"] = pred_df[time_col].dt.normalize()
    pred_df["slot"] = pred_df[time_col].dt.hour * 4 + pred_df[time_col].dt.minute // 15

    candidate_rows: list[dict[str, object]] = []
    day_rows: list[dict[str, object]] = []
    for date, group in pred_df.groupby("date", sort=True):
        if len(group) != 96:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            continue
        candidates, days = analyze_day(
            model=model,
            fold=str(group["fold"].iloc[0]),
            date=date,
            true_prices=group[target_col].to_numpy(dtype=float),
            pred_prices=group["pred"].to_numpy(dtype=float),
        )
        candidate_rows.extend(candidates)
        day_rows.extend(days)
    return pd.DataFrame(candidate_rows), pd.DataFrame(day_rows)


def summarize(day_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, top_k), group in day_df.groupby(["model", "top_k"], sort=True):
        rows.append(
            {
                "model": model,
                "top_k": int(top_k),
                "days": len(group),
                "top1_mean_profit": float(group["top1_profit"].mean()),
                "topk_best_mean_profit": float(group["topk_best_profit"].mean()),
                "topk_lift_mean": float(group["topk_lift_vs_top1"].mean()),
                "topk_lift_positive_days": int((group["topk_lift_vs_top1"] > 0).sum()),
                "topk_lift_positive_rate": float((group["topk_lift_vs_top1"] > 0).mean()),
                "top1_oracle_hit_rate": float(group["top1_is_oracle"].mean()),
                "oracle_in_topk_rate": float(group["oracle_in_topk"].mean()),
                "top1_mean_regret": float(group["top1_regret"].mean()),
                "topk_best_mean_regret": float(group["topk_best_regret"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "top_k"])


def write_markdown(summary: pd.DataFrame) -> None:
    top10 = summary[summary["top_k"] == 10].copy()
    lines = [
        "# Top-K Pair Diagnostics",
        "",
        "## Scope",
        "",
        "For each expert and validation day, rank legal charge/discharge pairs by predicted "
        "spread, then score the predicted top-K candidates using hidden validation prices.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top-10 Interpretation",
        "",
        top10[
            [
                "model",
                "top1_mean_profit",
                "topk_best_mean_profit",
                "topk_lift_mean",
                "topk_lift_positive_rate",
                "oracle_in_topk_rate",
                "top1_mean_regret",
                "topk_best_mean_regret",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Decision",
        "",
    ]
    if float(top10["topk_lift_mean"].max()) > 100:
        lines += [
            "- Predicted top-K contains materially better pairs than top1 on average.",
            "- Next step is a conservative reranker diagnostic using only ex-ante features.",
        ]
    else:
        lines += [
            "- Predicted top-K does not contain enough extra value to justify reranking.",
            "- Focus should remain on improving price curve prediction.",
        ]
    (REPORTS_DIR / "topk_pair_diagnostics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    all_candidates: list[pd.DataFrame] = []
    all_days: list[pd.DataFrame] = []
    for model, config_path in EXPERT_CONFIGS.items():
        print(f"running_model={model}", flush=True)
        candidates, days = analyze_model(model, load_config(config_path))
        all_candidates.append(candidates)
        all_days.append(days)

    candidate_df = pd.concat(all_candidates, ignore_index=True)
    day_df = pd.concat(all_days, ignore_index=True)
    summary = summarize(day_df)

    candidate_path = REPORTS_DIR / "topk_pair_candidates.csv"
    day_path = REPORTS_DIR / "topk_pair_daily.csv"
    summary_path = REPORTS_DIR / "topk_pair_summary.csv"
    candidate_df.to_csv(candidate_path, index=False)
    day_df.to_csv(day_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown(summary)

    print(summary.to_string(index=False))
    print(f"candidate_path={candidate_path}")
    print(f"day_path={day_path}")
    print(f"summary_path={summary_path}")
    print(f"markdown_path={REPORTS_DIR / 'topk_pair_diagnostics.md'}")


if __name__ == "__main__":
    main()
