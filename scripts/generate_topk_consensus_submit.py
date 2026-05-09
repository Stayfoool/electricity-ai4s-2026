from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity.data import load_test_frame, load_train_frame
from electricity.dispatch import rank_day_pairs
from electricity.eval.segmented_utils import segmented_feature_build_spec
from electricity.features import build_feature_frame, feature_columns
from electricity.submit import (
    _format_times,
    _member_train_frame,
    _predict_full_model,
    validate_submission,
)

POWER = 1000.0
TOP_K = 10
RULE = "unanimous_then_rank"
CHAMPION_CONFIG = Path("configs/ensemble_champion_and_segmented_6.yaml")
SEGMENTED_CONFIG = Path("configs/lgb_segmented_6_last_180d.yaml")
MARGIN_CONFIG = Path("configs/lgb_segmented_6_margin_core_last_180d.yaml")
OUTPUT_PATH = Path("outputs/output_topk_consensus_unanimous_then_rank.csv")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict_full_config(cfg: dict) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    raw_train = load_train_frame(cfg)
    raw_test = load_test_frame(cfg)

    if "ensemble" in cfg:
        specs = cfg["ensemble"]["members"]
        weights = np.asarray([float(spec.get("weight", 1.0)) for spec in specs], dtype=float)
        weights = weights / weights.sum()
        preds: list[np.ndarray] = []
        for spec in specs:
            build_spec = segmented_feature_build_spec(spec)
            feature_cols = feature_columns(build_spec)
            train_df = build_feature_frame(raw_train, build_spec, time_col=time_col)
            train_df = _member_train_frame(train_df, spec, time_col=time_col)
            test_df = build_feature_frame(raw_test, build_spec, time_col=time_col)
            print(f"training_member={spec['model']['name']}", flush=True)
            preds.append(
                _predict_full_model(
                    train_df,
                    test_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    spec=spec,
                    time_col=time_col,
                )
            )
        out = raw_test.sort_values(time_col).reset_index(drop=True)
        out["pred"] = np.average(np.vstack(preds), axis=0, weights=weights)
        return out[[time_col, "pred"]].copy()

    build_spec = segmented_feature_build_spec(cfg)
    feature_cols = feature_columns(build_spec)
    train_df = build_feature_frame(raw_train, build_spec, time_col=time_col)
    train_df = _member_train_frame(train_df, cfg, time_col=time_col)
    test_df = build_feature_frame(raw_test, build_spec, time_col=time_col)
    print(f"training_model={cfg['model']['name']}", flush=True)
    test_df["pred"] = _predict_full_model(
        train_df,
        test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        spec=cfg,
        time_col=time_col,
    )
    return test_df[[time_col, "pred"]].copy()


def candidate_rows(model: str, prices: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ranked = rank_day_pairs(prices, top_k=TOP_K)
    top1_spread = ranked[0][0]
    for rank, (spread, charge_start, discharge_start) in enumerate(ranked, start=1):
        rows.append(
            {
                "model": model,
                "candidate_rank": rank,
                "charge_start": charge_start,
                "discharge_start": discharge_start,
                "predicted_spread": spread,
                "spread_delta_from_model_top1": top1_spread - spread,
            }
        )
    return rows


def choose_unanimous_then_rank(candidates: pd.DataFrame) -> tuple[int, int]:
    index_cols = ["charge_start", "discharge_start"]
    agg = (
        candidates.groupby(index_cols, as_index=False)
        .agg(
            support_count=("model", "nunique"),
            mean_rank=("candidate_rank", "mean"),
            mean_predicted_spread=("predicted_spread", "mean"),
            mean_spread_delta=("spread_delta_from_model_top1", "mean"),
        )
        .sort_values(["charge_start", "discharge_start"])
        .reset_index(drop=True)
    )
    champion_top1 = candidates[
        (candidates["model"] == "champion") & (candidates["candidate_rank"] == 1)
    ].iloc[0]
    unanimous = agg[agg["support_count"] == 3]
    if unanimous.empty:
        return int(champion_top1["charge_start"]), int(champion_top1["discharge_start"])

    selected = unanimous.sort_values(
        ["mean_rank", "mean_spread_delta", "mean_predicted_spread"],
        ascending=[True, True, False],
    ).iloc[0]
    return int(selected["charge_start"]), int(selected["discharge_start"])


def build_consensus_power(predictions: dict[str, pd.DataFrame], cfg: dict) -> np.ndarray:
    time_col = cfg["data"]["time_col"]
    points_per_day = int(cfg["dispatch"]["points_per_day"])
    block_size = int(cfg["dispatch"]["block_size"])
    charge_power = float(cfg["dispatch"]["charge_power"])
    discharge_power = float(cfg["dispatch"]["discharge_power"])

    base = predictions["champion"][[time_col]].copy()
    base["date"] = base[time_col].dt.normalize()
    base["slot"] = base[time_col].dt.hour * 4 + base[time_col].dt.minute // 15
    power = np.zeros(len(base), dtype=float)

    pred_by_model = {}
    for model, frame in predictions.items():
        work = frame.copy()
        work["date"] = work[time_col].dt.normalize()
        work["slot"] = work[time_col].dt.hour * 4 + work[time_col].dt.minute // 15
        pred_by_model[model] = work

    for date, group in base.groupby("date", sort=True):
        if len(group) != points_per_day:
            raise ValueError(f"{date.date()} does not contain {points_per_day} rows")
        if not np.array_equal(group.sort_values("slot")["slot"].to_numpy(), np.arange(96)):
            raise ValueError(f"{date.date()} has missing or duplicated slots")

        rows: list[dict[str, object]] = []
        for model, frame in pred_by_model.items():
            prices = (
                frame[frame["date"] == date]
                .sort_values("slot")["pred"]
                .to_numpy(dtype=float)
            )
            if len(prices) != points_per_day:
                raise ValueError(f"{model} missing predictions for {date.date()}")
            rows.extend(candidate_rows(model, prices))

        charge_start, discharge_start = choose_unanimous_then_rank(pd.DataFrame(rows))
        day_index = group.sort_values("slot").index.to_numpy()
        power[day_index[charge_start : charge_start + block_size]] = charge_power
        power[day_index[discharge_start : discharge_start + block_size]] = discharge_power

    return power


def main() -> None:
    cfg = load_config(CHAMPION_CONFIG)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    time_col = cfg["data"]["time_col"]

    predictions = {
        "champion": predict_full_config(cfg),
        "segmented": predict_full_config(load_config(SEGMENTED_CONFIG)),
        "margin": predict_full_config(load_config(MARGIN_CONFIG)),
    }
    base = predictions["champion"].copy()
    base["power"] = build_consensus_power(predictions, cfg)
    submission = pd.DataFrame(
        {
            "times": _format_times(base[time_col]),
            "实时价格": base["pred"],
            "power": base["power"],
        }
    )
    validate_submission(submission, cfg)
    submission.to_csv(OUTPUT_PATH, index=False)
    trade_days = int((submission["power"] != 0).groupby(base[time_col].dt.date).any().sum())
    print(f"rule={RULE}")
    print(f"output_path={OUTPUT_PATH}")
    print(f"shape={submission.shape}")
    print(f"trade_days={trade_days}")


if __name__ == "__main__":
    main()
