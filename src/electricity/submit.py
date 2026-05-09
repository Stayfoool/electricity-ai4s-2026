from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from electricity.data import load_test_frame, load_train_frame
from electricity.dispatch import optimize_day
from electricity.eval.segmented_utils import (
    add_slot_and_segment,
    segment_boundaries,
    segmented_feature_build_spec,
    segmented_feature_columns,
)
from electricity.features import build_feature_frame, feature_columns
from electricity.models import predict_model, train_model_full

SUBMIT_COLUMNS = ["times", "实时价格", "power"]


def _feature_cols(cfg: dict) -> list[str]:
    return feature_columns(cfg)


def _format_times(series: pd.Series) -> pd.Series:
    return series.dt.strftime("%Y-%m-%d %H:%M:%S")


def build_power_schedule(
    df: pd.DataFrame,
    *,
    time_col: str,
    price_col: str,
    cfg: dict,
) -> np.ndarray:
    power = np.zeros(len(df), dtype=float)
    work = df[[time_col, price_col]].copy()
    work["date"] = work[time_col].dt.normalize()
    work["slot"] = work[time_col].dt.hour * 4 + work[time_col].dt.minute // 15

    for _, group in work.groupby("date", sort=True):
        if len(group) != cfg["dispatch"]["points_per_day"]:
            raise ValueError("each submit day must contain exactly 96 rows")
        group = group.sort_values("slot")
        expected_slots = np.arange(cfg["dispatch"]["points_per_day"])
        if not np.array_equal(group["slot"].to_numpy(), expected_slots):
            raise ValueError("submit day has missing or duplicated 15-minute slots")

        result = optimize_day(
            group[price_col].to_numpy(dtype=float),
            tau=float(cfg["dispatch"]["tau"]),
            block_size=int(cfg["dispatch"]["block_size"]),
            charge_power=float(cfg["dispatch"]["charge_power"]),
            discharge_power=float(cfg["dispatch"]["discharge_power"]),
        )
        power[group.index.to_numpy()] = result.power

    return power


def _member_train_frame(df: pd.DataFrame, spec: dict, *, time_col: str) -> pd.DataFrame:
    train_window_days = spec["model"].get("train_window_days")
    if train_window_days is None:
        return df.copy()
    train_end = df[time_col].max()
    train_start = train_end - pd.Timedelta(days=train_window_days) + pd.Timedelta(seconds=1)
    return df[df[time_col] >= train_start].copy()


def _segment_boundaries(spec: dict) -> list[int] | None:
    boundaries = spec["model"].get("segment_boundaries")
    if boundaries is None:
        return None
    return segment_boundaries(spec)


def _predict_segmented_full(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    spec: dict,
    time_col: str,
) -> np.ndarray:
    boundaries = _segment_boundaries(spec)
    if boundaries is None:
        raise ValueError("segmented full prediction requires segment_boundaries")

    train_seg = add_slot_and_segment(train_df, time_col=time_col, boundaries=boundaries)
    test_seg = add_slot_and_segment(test_df, time_col=time_col, boundaries=boundaries)
    feature_cols_by_segment = segmented_feature_columns(spec, boundaries)
    preds = np.full(len(test_seg), np.nan, dtype=float)

    for segment_id in sorted(test_seg["segment"].unique()):
        train_part = train_seg[train_seg["segment"] == segment_id].copy()
        test_part = test_seg[test_seg["segment"] == segment_id].copy()
        if train_part.empty or test_part.empty:
            raise ValueError(f"empty segmented full prediction split segment={segment_id}")
        feature_cols = feature_cols_by_segment[segment_id]
        model = train_model_full(
            train_part,
            feature_cols=feature_cols,
            target_col=target_col,
            cfg=spec,
        )
        preds[test_part.index.to_numpy()] = predict_model(model, test_part, feature_cols, spec)

    if np.isnan(preds).any():
        missing = int(np.isnan(preds).sum())
        raise ValueError(f"segmented full prediction left {missing} rows without predictions")
    return preds


def _predict_full_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    spec: dict,
    time_col: str,
) -> np.ndarray:
    if _segment_boundaries(spec) is not None:
        return _predict_segmented_full(
            train_df,
            test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            spec=spec,
            time_col=time_col,
        )
    model = train_model_full(
        train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        cfg=spec,
    )
    return predict_model(model, test_df, feature_cols, spec)


def validate_submission(df: pd.DataFrame, cfg: dict) -> None:
    if list(df.columns) != SUBMIT_COLUMNS:
        raise ValueError(f"submission columns must be {SUBMIT_COLUMNS}, got {list(df.columns)}")

    time_col = "times"
    times = pd.to_datetime(df[time_col])
    if times.isna().any():
        raise ValueError("submission contains invalid times")
    if not times.is_monotonic_increasing:
        raise ValueError("submission times must be increasing")
    if times.duplicated().any():
        raise ValueError("submission times must be unique")

    expected_rows = 59 * int(cfg["dispatch"]["points_per_day"])
    if len(df) != expected_rows:
        raise ValueError(f"submission must contain {expected_rows} rows, got {len(df)}")

    valid_power = {
        float(cfg["dispatch"]["charge_power"]),
        0.0,
        float(cfg["dispatch"]["discharge_power"]),
    }
    observed_power = set(df["power"].astype(float).unique())
    if not observed_power <= valid_power:
        raise ValueError(f"invalid power values: {sorted(observed_power - valid_power)}")

    work = pd.DataFrame({"times": times, "power": df["power"].astype(float)})
    work["date"] = work["times"].dt.normalize()
    work["slot"] = work["times"].dt.hour * 4 + work["times"].dt.minute // 15
    block_size = int(cfg["dispatch"]["block_size"])
    charge_power = float(cfg["dispatch"]["charge_power"])
    discharge_power = float(cfg["dispatch"]["discharge_power"])

    for date, group in work.groupby("date", sort=True):
        if len(group) != int(cfg["dispatch"]["points_per_day"]):
            raise ValueError(f"{date.date()} does not contain 96 rows")
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            raise ValueError(f"{date.date()} has missing or duplicated slots")

        values = group["power"].to_numpy(dtype=float)
        charge_idx = np.flatnonzero(values == charge_power)
        discharge_idx = np.flatnonzero(values == discharge_power)
        if len(charge_idx) not in {0, block_size}:
            raise ValueError(f"{date.date()} has invalid charge block length")
        if len(discharge_idx) not in {0, block_size}:
            raise ValueError(f"{date.date()} has invalid discharge block length")
        if len(charge_idx) != len(discharge_idx):
            raise ValueError(f"{date.date()} must charge and discharge together or not trade")
        if len(charge_idx) == 0:
            continue
        if not np.array_equal(charge_idx, np.arange(charge_idx[0], charge_idx[0] + block_size)):
            raise ValueError(f"{date.date()} charge block must be consecutive")
        if not np.array_equal(
            discharge_idx,
            np.arange(discharge_idx[0], discharge_idx[0] + block_size),
        ):
            raise ValueError(f"{date.date()} discharge block must be consecutive")
        if discharge_idx[0] < charge_idx[0] + block_size:
            raise ValueError(f"{date.date()} discharge must start after charge block")


def run_submit(cfg: dict, *, config_path: str) -> Path:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    build_spec = segmented_feature_build_spec(cfg)
    feature_cols = _feature_cols(build_spec)
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("loading_train", flush=True)
    train_df = build_feature_frame(load_train_frame(cfg), build_spec, time_col=time_col)
    print("predicting_test", flush=True)
    test_df = build_feature_frame(load_test_frame(cfg), build_spec, time_col=time_col)
    print("training_full_model", flush=True)
    test_df["实时价格"] = _predict_full_model(
        train_df,
        test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        spec=cfg,
        time_col=time_col,
    )
    test_df["power"] = build_power_schedule(
        test_df,
        time_col=time_col,
        price_col="实时价格",
        cfg=cfg,
    )

    submission = pd.DataFrame(
        {
            "times": _format_times(test_df[time_col]),
            "实时价格": test_df["实时价格"],
            "power": test_df["power"],
        }
    )
    validate_submission(submission, cfg)

    output_path = outputs_dir / f"output_{cfg['model']['name']}.csv"
    submission.to_csv(output_path, index=False)
    print(f"config={config_path}")
    print(f"output_path={output_path}")
    print(f"shape={submission.shape}")
    trade_days = int((submission["power"] != 0).groupby(test_df[time_col].dt.date).any().sum())
    print(f"trade_days={trade_days}")
    return output_path


def run_ensemble_submit(cfg: dict, *, config_path: str) -> Path:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    specs = cfg["ensemble"]["members"]
    weights = np.array([float(spec.get("weight", 1.0)) for spec in specs], dtype=float)
    weights = weights / weights.sum()

    raw_train = load_train_frame(cfg)
    raw_test = load_test_frame(cfg)
    preds: list[np.ndarray] = []

    for spec in specs:
        build_spec = segmented_feature_build_spec(spec)
        feature_cols = feature_columns(build_spec)
        train_df = build_feature_frame(raw_train, build_spec, time_col=time_col)
        train_df = _member_train_frame(train_df, spec, time_col=time_col)
        test_df = build_feature_frame(raw_test, build_spec, time_col=time_col)

        actual_train_start = train_df[time_col].min().strftime("%Y-%m-%d")
        actual_train_end = train_df[time_col].max().strftime("%Y-%m-%d")
        print(
            f"training_member={spec['model']['name']} "
            f"train_range={actual_train_start}..{actual_train_end}",
            flush=True,
        )
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

    base_test = raw_test.sort_values(time_col).reset_index(drop=True)
    base_test["实时价格"] = np.average(np.vstack(preds), axis=0, weights=weights)
    base_test["power"] = build_power_schedule(
        base_test,
        time_col=time_col,
        price_col="实时价格",
        cfg=cfg,
    )

    submission = pd.DataFrame(
        {
            "times": _format_times(base_test[time_col]),
            "实时价格": base_test["实时价格"],
            "power": base_test["power"],
        }
    )
    validate_submission(submission, cfg)

    output_path = outputs_dir / f"output_{cfg['model']['name']}.csv"
    submission.to_csv(output_path, index=False)
    print(f"config={config_path}")
    print(f"output_path={output_path}")
    print(f"shape={submission.shape}")
    trade_days = int((submission["power"] != 0).groupby(base_test[time_col].dt.date).any().sum())
    print(f"trade_days={trade_days}")
    return output_path
