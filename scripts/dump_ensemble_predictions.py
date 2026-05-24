"""Dump per-fold 96-point predictions for the champion ensemble to a CSV cache.

This script reproduces the prediction logic of ``run_ensemble_backtest`` for the
``configs/ensemble_champion_and_segmented_6.yaml`` config but does **not** rely
on mlflow and reads training CSVs directly (faster than opening the 4.7GB zip).

Output: ``reports/cache_predictions_<run_name>.csv`` with columns
``[fold, times, target, pred]``. One row per (valid timestamp, fold).

Used by ``scripts/tune_dispatch_prior.py`` to grid-search dispatch priors
without retraining models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from electricity.eval.backtest import _fold_bounds, _fold_train_frame
from electricity.eval.segmented_utils import (
    add_slot_and_segment,
    predict_segmented_fold,
    segment_boundaries,
    segmented_feature_build_spec,
    segmented_feature_columns,
)
from electricity.features import build_feature_frame, feature_columns
from electricity.models import predict_model, train_model


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_frame_fast(cfg: dict) -> pd.DataFrame:
    """Load training frame directly from the unzipped CSVs."""
    root = Path(cfg["paths"].get("material_dir", "eletricmaterial"))
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]

    feat_path = root / cfg["data"]["train_feature"]
    label_path = root / cfg["data"]["train_label"]
    features = pd.read_csv(feat_path, parse_dates=[time_col])
    labels = pd.read_csv(label_path, parse_dates=[time_col])
    train = features.merge(labels[[time_col, target_col]], on=time_col, how="inner")
    return train.sort_values(time_col).reset_index(drop=True)


def add_target_absolute(df: pd.DataFrame, *, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out["target_absolute"] = out[target_col]
    return out


def member_predict_fold(
    *,
    spec: dict,
    frame: pd.DataFrame,
    feature_cols: list[str],
    train_target_col: str,
    time_col: str,
    fold: dict,
) -> tuple[np.ndarray, pd.DataFrame]:
    train_window_days = spec["model"].get("train_window_days")
    bounds = _fold_bounds(fold)

    train_df = _fold_train_frame(
        frame,
        time_col=time_col,
        train_end=bounds.train_end,
        train_window_days=train_window_days,
        train_start=bounds.train_start,
        valid_start=bounds.valid_start,
        valid_end=bounds.valid_end,
    )
    valid_df = frame[
        (frame[time_col] >= bounds.valid_start) & (frame[time_col] <= bounds.valid_end)
    ].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(
            f"empty train/valid split for fold={fold['name']} member={spec['model']['name']}"
        )

    boundaries = spec["model"].get("segment_boundaries")
    if boundaries is not None:
        valid_df = valid_df.reset_index(drop=True)
        boundary_values = segment_boundaries(spec)
        feature_cols_by_segment = segmented_feature_columns(spec, boundary_values)
        train_df = add_slot_and_segment(train_df, time_col=time_col, boundaries=boundary_values)
        valid_df = add_slot_and_segment(valid_df, time_col=time_col, boundaries=boundary_values)
        preds, _, _ = predict_segmented_fold(
            train_df,
            valid_df,
            feature_cols_by_segment=feature_cols_by_segment,
            target_col=train_target_col,
            cfg=spec,
            boundaries=boundary_values,
        )
        return preds, valid_df

    model = train_model(
        train_df,
        valid_df,
        feature_cols=feature_cols,
        target_col=train_target_col,
        cfg=spec,
    )
    preds = predict_model(model, valid_df, feature_cols, spec)
    return preds, valid_df


def model_spec_frame(
    df: pd.DataFrame,
    spec: dict,
    *,
    time_col: str,
    target_col: str,
) -> pd.DataFrame:
    build_spec = segmented_feature_build_spec(spec)
    return add_target_absolute(
        build_feature_frame(df, build_spec, time_col=time_col),
        target_col=target_col,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ensemble_champion_and_segmented_6.yaml")
    parser.add_argument("--out", default="reports/cache_predictions_ens_champion_segmented6.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]

    raw = load_train_frame_fast(cfg)
    print(
        f"loaded train frame: {len(raw)} rows from "
        f"{raw[time_col].min()} to {raw[time_col].max()}"
    )

    specs = cfg["ensemble"]["members"]
    weights = np.array([float(s.get("weight", 1.0)) for s in specs], dtype=float)
    weights = weights / weights.sum()
    print(f"ensemble members: {[s['model']['name'] for s in specs]}")
    print(f"weights         : {weights.tolist()}")

    member_frames = [
        model_spec_frame(raw, s, time_col=time_col, target_col=target_col) for s in specs
    ]
    member_feat_cols = [feature_columns(s) for s in specs]

    all_rows: list[pd.DataFrame] = []
    for fold in cfg["folds"]:
        print(
            f"\n[fold] {fold['name']} train<={fold['train_end']} "
            f"valid {fold['valid_start']}..{fold['valid_end']}"
        )
        member_preds: list[np.ndarray] = []
        valid_ref: pd.DataFrame | None = None
        for spec, frame, fcols in zip(specs, member_frames, member_feat_cols, strict=True):
            print(f"  training member: {spec['model']['name']}", flush=True)
            preds, valid_df = member_predict_fold(
                spec=spec,
                frame=frame,
                feature_cols=fcols,
                train_target_col="target_absolute",
                time_col=time_col,
                fold=fold,
            )
            member_preds.append(preds)
            if valid_ref is None:
                valid_ref = valid_df[[time_col, target_col]].copy()

        assert valid_ref is not None
        ens_pred = np.average(np.vstack(member_preds), axis=0, weights=weights)
        out_df = valid_ref.rename(columns={target_col: "target"})
        out_df["pred"] = ens_pred
        out_df["fold"] = fold["name"]
        all_rows.append(out_df)

    cache = pd.concat(all_rows, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(out_path, index=False)
    print(f"\nsaved predictions cache -> {out_path}  rows={len(cache)}")


if __name__ == "__main__":
    main()
