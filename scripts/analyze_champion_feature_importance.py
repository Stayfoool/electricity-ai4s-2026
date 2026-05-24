from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

from electricity.eval.backtest import (
    _add_sample_weight,
    _daily_curve_z_rmse,
    _evaluate_dispatch,
    _fold_bounds,
    _fold_train_frame,
    _maybe_build_prior,
    _model_spec_frame,
    _target_col_for_mode,
)
from electricity.eval.segmented_utils import (
    add_slot_and_segment,
    segment_boundaries,
    segment_labels,
    segmented_feature_columns,
)
from electricity.features import (
    TIME_FEATURES,
    apply_fold_bias_correction,
    apply_fold_weather_correction,
)
from electricity.features import feature_columns as configured_feature_columns
from electricity.models import best_iteration, model_backend, predict_model, train_model

RAW_BOUNDARY_FEATURES = [
    "系统负荷预测值",
    "风光总加预测值",
    "联络线预测值",
    "风电预测值",
    "光伏预测值",
    "水电预测值",
    "非市场化机组预测值",
]

TIME_DROP_GROUPS = {
    "drop_time_clock": [
        "hour",
        "minute",
        "quarter",
        "hour_sin",
        "hour_cos",
        "quarter_sin",
        "quarter_cos",
    ],
    "drop_time_calendar": ["dayofweek", "month", "is_weekend"],
    "drop_time_all": list(TIME_FEATURES),
}


@dataclass(frozen=True)
class SegmentModel:
    segment_id: int
    segment_label: str
    model: object
    feature_cols: list[str]
    valid_rows: int
    train_rows: int


@dataclass(frozen=True)
class TrainedMember:
    member_idx: int
    member_name: str
    weight: float
    spec: dict
    valid_df: pd.DataFrame
    pred: np.ndarray
    feature_cols: list[str]
    train_range: str
    models: list[SegmentModel]


@dataclass(frozen=True)
class FoldBundle:
    fold: str
    valid_df: pd.DataFrame
    members: list[TrainedMember]
    pred: np.ndarray
    metrics: dict
    mae: float
    rmse: float
    curve_z_rmse: float


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_from_zip(zip_path: Path, member: str, *, parse_dates: list[str]) -> pd.DataFrame:
    with ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            return pd.read_csv(f, parse_dates=parse_dates)


def load_train_frame_fast(cfg: dict) -> pd.DataFrame:
    """Prefer unzipped competition CSVs; fall back to the official zip."""
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    material_dir = Path(cfg["paths"].get("material_dir", "eletricmaterial"))
    feature_path = material_dir / cfg["data"]["train_feature"]
    label_path = material_dir / cfg["data"]["train_label"]

    if feature_path.exists() and label_path.exists():
        features = pd.read_csv(feature_path, parse_dates=[time_col])
        labels = pd.read_csv(label_path, parse_dates=[time_col])
    else:
        zip_path = Path(cfg["paths"]["data_zip"])
        if not zip_path.is_absolute():
            zip_path = Path.cwd() / zip_path
        features = read_csv_from_zip(
            zip_path,
            cfg["data"]["train_feature"],
            parse_dates=[time_col],
        )
        labels = read_csv_from_zip(
            zip_path,
            cfg["data"]["train_label"],
            parse_dates=[time_col],
        )

    train = features.merge(labels[[time_col, target_col]], on=time_col, how="inner")
    return train.sort_values(time_col).reset_index(drop=True)


def prepare_specs(cfg: dict) -> tuple[list[dict], np.ndarray]:
    specs: list[dict] = []
    for spec in cfg["ensemble"]["members"]:
        merged_spec = deepcopy(spec)
        merged_spec.setdefault("paths", cfg["paths"])
        if "sample_weighting" in cfg:
            merged_spec.setdefault("sample_weighting", cfg["sample_weighting"])
        if "weekly_relative_features" in cfg:
            merged_spec.setdefault("weekly_relative_features", cfg["weekly_relative_features"])
        specs.append(merged_spec)
    weights = np.asarray([float(spec.get("weight", 1.0)) for spec in specs], dtype=float)
    weights = weights / weights.sum()
    return specs, weights


def drop_feature_columns(feature_cols: list[str], drop_cols: set[str]) -> list[str]:
    kept = [col for col in feature_cols if col not in drop_cols]
    if not kept:
        raise ValueError(f"feature drop removed all model columns: {sorted(drop_cols)}")
    return kept


def apply_fold_transforms(
    frame: pd.DataFrame,
    spec: dict,
    *,
    time_col: str,
    train_end: pd.Timestamp,
    train_start: pd.Timestamp | None = None,
    valid_start: pd.Timestamp | None = None,
    valid_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    out = apply_fold_bias_correction(
        frame,
        spec,
        time_col=time_col,
        train_end=train_end,
        train_start=train_start,
        valid_start=valid_start,
        valid_end=valid_end,
    )
    return apply_fold_weather_correction(
        out,
        spec,
        time_col=time_col,
        train_end=train_end,
        train_start=train_start,
        valid_start=valid_start,
        valid_end=valid_end,
    )


def train_member_fold(
    *,
    member_idx: int,
    spec: dict,
    frame: pd.DataFrame,
    feature_cols: list[str],
    weight: float,
    train_target_col: str,
    time_col: str,
    target_col: str,
    fold: dict,
    drop_cols: set[str],
) -> TrainedMember:
    train_window_days = spec["model"].get("train_window_days")
    train_end = pd.Timestamp(fold["train_end"])
    valid_start = pd.Timestamp(fold["valid_start"])
    valid_end = pd.Timestamp(fold["valid_end"])
    fold_train_start = pd.Timestamp(fold["train_start"]) if fold.get("train_start") else None

    train_df = _fold_train_frame(
        frame,
        time_col=time_col,
        train_end=train_end,
        train_window_days=train_window_days,
        train_start=fold_train_start,
        valid_start=valid_start,
        valid_end=valid_end,
    )
    valid_df = frame[(frame[time_col] >= valid_start) & (frame[time_col] <= valid_end)].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(f"empty train/valid split for fold={fold['name']}")
    train_df = _add_sample_weight(train_df, spec, time_col=time_col)

    train_range = (
        f"{spec['model']['name']}:{train_df[time_col].min():%Y-%m-%d}.."
        f"{train_df[time_col].max():%Y-%m-%d}"
    )

    boundaries = spec["model"].get("segment_boundaries")
    if boundaries is None:
        cols = drop_feature_columns(feature_cols, drop_cols)
        model = train_model(
            train_df,
            valid_df,
            feature_cols=cols,
            target_col=train_target_col,
            cfg=spec,
        )
        pred = predict_model(model, valid_df, cols, spec)
        return TrainedMember(
            member_idx=member_idx,
            member_name=spec["model"]["name"],
            weight=weight,
            spec=spec,
            valid_df=valid_df,
            pred=pred,
            feature_cols=cols,
            train_range=train_range,
            models=[
                SegmentModel(
                    segment_id=-1,
                    segment_label="all",
                    model=model,
                    feature_cols=cols,
                    valid_rows=len(valid_df),
                    train_rows=len(train_df),
                )
            ],
        )

    boundary_values = segment_boundaries(spec)
    labels = segment_labels(boundary_values)
    feature_cols_by_segment = {
        segment_id: drop_feature_columns(cols, drop_cols)
        for segment_id, cols in segmented_feature_columns(spec, boundary_values).items()
    }
    train_seg_df = add_slot_and_segment(train_df, time_col=time_col, boundaries=boundary_values)
    valid_seg_df = add_slot_and_segment(
        valid_df.reset_index(drop=True),
        time_col=time_col,
        boundaries=boundary_values,
    )
    preds = np.full(len(valid_seg_df), np.nan, dtype=float)
    models: list[SegmentModel] = []
    for segment_id, label in enumerate(labels):
        segment_train = train_seg_df[train_seg_df["segment"] == segment_id].copy()
        segment_valid = valid_seg_df[valid_seg_df["segment"] == segment_id].copy()
        if segment_train.empty or segment_valid.empty:
            raise ValueError(f"empty train/valid segment={label} fold={fold['name']}")
        cols = feature_cols_by_segment[segment_id]
        model = train_model(
            segment_train,
            segment_valid,
            feature_cols=cols,
            target_col=train_target_col,
            cfg=spec,
        )
        preds[segment_valid.index.to_numpy()] = predict_model(model, segment_valid, cols, spec)
        models.append(
            SegmentModel(
                segment_id=segment_id,
                segment_label=label,
                model=model,
                feature_cols=cols,
                valid_rows=len(segment_valid),
                train_rows=len(segment_train),
            )
        )
    if np.isnan(preds).any():
        raise ValueError(f"segmented prediction left {int(np.isnan(preds).sum())} missing rows")

    return TrainedMember(
        member_idx=member_idx,
        member_name=spec["model"]["name"],
        weight=weight,
        spec=spec,
        valid_df=valid_seg_df,
        pred=preds,
        feature_cols=sorted(set().union(*(set(m.feature_cols) for m in models))),
        train_range=train_range,
        models=models,
    )


def predict_member(member: TrainedMember, valid_df: pd.DataFrame) -> np.ndarray:
    if len(member.models) == 1 and member.models[0].segment_id == -1:
        segment_model = member.models[0]
        return predict_model(segment_model.model, valid_df, segment_model.feature_cols, member.spec)

    preds = np.full(len(valid_df), np.nan, dtype=float)
    for segment_model in member.models:
        part = valid_df[valid_df["segment"] == segment_model.segment_id].copy()
        if part.empty:
            continue
        preds[part.index.to_numpy()] = predict_model(
            segment_model.model,
            part,
            segment_model.feature_cols,
            member.spec,
        )
    if np.isnan(preds).any():
        raise ValueError(f"member prediction has {int(np.isnan(preds).sum())} missing rows")
    return preds


def evaluate_prediction(
    valid_df: pd.DataFrame,
    *,
    pred: np.ndarray,
    cfg: dict,
    raw_df: pd.DataFrame,
    fold: dict,
) -> tuple[pd.DataFrame, dict, float, float, float]:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    eval_df = valid_df[[time_col, target_col]].copy()
    eval_df["pred"] = pred

    y_true = eval_df[target_col].to_numpy()
    mae = float(mean_absolute_error(y_true, pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    curve_z_rmse = _daily_curve_z_rmse(eval_df, time_col=time_col, target_col=target_col)

    log_pc, log_pd, lc_w, ld_w = _maybe_build_prior(
        cfg,
        raw_df,
        time_col=time_col,
        target_col=target_col,
        train_end=pd.Timestamp(fold["train_end"]),
        train_start=_fold_bounds(fold).train_start,
        valid_start=pd.Timestamp(fold["valid_start"]),
        valid_end=pd.Timestamp(fold["valid_end"]),
    )
    metrics = _evaluate_dispatch(
        eval_df,
        time_col=time_col,
        target_col=target_col,
        tau=float(cfg["dispatch"]["tau"]),
        log_prior_charge=log_pc,
        log_prior_discharge=log_pd,
        lambda_charge=lc_w,
        lambda_discharge=ld_w,
    )
    return eval_df, metrics, mae, rmse, curve_z_rmse


def train_ensemble_fold(
    *,
    cfg: dict,
    raw_df: pd.DataFrame,
    specs: list[dict],
    weights: np.ndarray,
    member_frames: list[pd.DataFrame],
    member_feature_cols: list[list[str]],
    member_target_cols: list[str],
    fold: dict,
    drop_cols: set[str],
) -> FoldBundle:
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    bounds = _fold_bounds(fold)
    train_end = bounds.train_end
    members: list[TrainedMember] = []
    valid_df_for_eval: pd.DataFrame | None = None

    for idx, (spec, frame, cols, train_target_col, weight) in enumerate(
        zip(specs, member_frames, member_feature_cols, member_target_cols, weights, strict=True)
    ):
        fold_frame = apply_fold_transforms(
            frame,
            spec,
            time_col=time_col,
            train_end=train_end,
            train_start=bounds.train_start,
            valid_start=bounds.valid_start,
            valid_end=bounds.valid_end,
        )
        member = train_member_fold(
            member_idx=idx,
            spec=spec,
            frame=fold_frame,
            feature_cols=cols,
            weight=float(weight),
            train_target_col=train_target_col,
            time_col=time_col,
            target_col=target_col,
            fold=fold,
            drop_cols=drop_cols,
        )
        members.append(member)
        if valid_df_for_eval is None:
            valid_df_for_eval = member.valid_df[[time_col, target_col]].copy()

    if valid_df_for_eval is None:
        raise ValueError(f"no validation data for fold={fold['name']}")

    pred = np.average(np.vstack([m.pred for m in members]), axis=0, weights=weights)
    eval_df, metrics, mae, rmse, curve_z_rmse = evaluate_prediction(
        valid_df_for_eval,
        pred=pred,
        cfg=cfg,
        raw_df=raw_df,
        fold=fold,
    )
    return FoldBundle(
        fold=fold["name"],
        valid_df=eval_df,
        members=members,
        pred=pred,
        metrics=metrics,
        mae=mae,
        rmse=rmse,
        curve_z_rmse=curve_z_rmse,
    )


def summarize_fold(
    *,
    run_name: str,
    drop_cols: list[str],
    bundle: FoldBundle,
) -> dict:
    return {
        "run_name": run_name,
        "drop_cols": ",".join(drop_cols),
        "fold": bundle.fold,
        "valid_rows": len(bundle.valid_df),
        "valid_days": bundle.metrics["valid_days"],
        "mae": bundle.mae,
        "rmse": bundle.rmse,
        "curve_z_rmse": bundle.curve_z_rmse,
        "avg_profit": bundle.metrics["avg_profit"],
        "oracle_avg_profit": bundle.metrics["oracle_avg_profit"],
        "oracle_ratio": bundle.metrics["oracle_ratio"],
        "avg_regret": bundle.metrics["avg_regret"],
        "loss_days": bundle.metrics["loss_days"],
        "trade_days": bundle.metrics["trade_days"],
        "trade_days_ratio": bundle.metrics["trade_days_ratio"],
        "mean_abs_charge_gap": bundle.metrics["mean_abs_charge_gap"],
        "mean_abs_discharge_gap": bundle.metrics["mean_abs_discharge_gap"],
        "best_iteration_mean": float(
            np.mean(
                [
                    best_iteration(segment_model.model, member.spec)
                    for member in bundle.members
                    for segment_model in member.models
                ]
            )
        ),
    }


def summarize_backtest(fold_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_name, group in fold_rows.groupby("run_name", sort=False):
        total_profit = float((group["avg_profit"] * group["valid_days"]).sum())
        total_oracle = float((group["oracle_avg_profit"] * group["valid_days"]).sum())
        rows.append(
            {
                "run_name": run_name,
                "drop_cols": group["drop_cols"].iloc[0],
                "folds": int(group["fold"].nunique()),
                "valid_days": int(group["valid_days"].sum()),
                "avg_profit_fold_mean": float(group["avg_profit"].mean()),
                "avg_profit_fold_min": float(group["avg_profit"].min()),
                "avg_profit_day_weighted": (
                    total_profit / group["valid_days"].sum()
                    if group["valid_days"].sum() > 0
                    else 0.0
                ),
                "oracle_ratio_fold_mean": float(group["oracle_ratio"].mean()),
                "oracle_ratio_day_weighted": (
                    total_profit / total_oracle if total_oracle > 0 else 0.0
                ),
                "avg_regret_fold_mean": float(group["avg_regret"].mean()),
                "loss_days_total": int(group["loss_days"].sum()),
                "rmse_mean": float(group["rmse"].mean()),
                "curve_z_rmse_mean": float(group["curve_z_rmse"].mean()),
                "mean_abs_charge_gap": float(group["mean_abs_charge_gap"].mean()),
                "mean_abs_discharge_gap": float(group["mean_abs_discharge_gap"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        base = out[out["run_name"] == "baseline"].iloc[0]
        for col in [
            "avg_profit_fold_mean",
            "avg_profit_day_weighted",
            "oracle_ratio_fold_mean",
            "oracle_ratio_day_weighted",
            "avg_regret_fold_mean",
            "rmse_mean",
            "curve_z_rmse_mean",
        ]:
            out[f"delta_{col}"] = out[col] - float(base[col])
        out["delta_loss_days_total"] = out["loss_days_total"] - int(base["loss_days_total"])
        out = out.sort_values(
            ["delta_avg_profit_fold_mean", "delta_avg_profit_day_weighted"],
            ascending=False,
        )
    return out


def booster_importance_rows(bundle: FoldBundle) -> list[dict]:
    rows: list[dict] = []
    for member in bundle.members:
        member_valid_rows = max(len(member.valid_df), 1)
        for segment_model in member.models:
            model = segment_model.model
            if not hasattr(model, "feature_importance"):
                continue
            gain = np.asarray(model.feature_importance(importance_type="gain"), dtype=float)
            split = np.asarray(model.feature_importance(importance_type="split"), dtype=float)
            gain_total = float(gain.sum())
            split_total = float(split.sum())
            scope_weight = member.weight * segment_model.valid_rows / member_valid_rows
            for feature, gain_value, split_value in zip(
                segment_model.feature_cols,
                gain,
                split,
                strict=True,
            ):
                gain_share = float(gain_value / gain_total) if gain_total > 0 else 0.0
                split_share = float(split_value / split_total) if split_total > 0 else 0.0
                rows.append(
                    {
                        "fold": bundle.fold,
                        "member_idx": member.member_idx,
                        "member_name": member.member_name,
                        "member_weight": member.weight,
                        "segment": segment_model.segment_label,
                        "scope_weight": scope_weight,
                        "feature": feature,
                        "gain": float(gain_value),
                        "split": float(split_value),
                        "gain_share": gain_share,
                        "split_share": split_share,
                        "weighted_gain_share": gain_share * scope_weight,
                        "weighted_split_share": split_share * scope_weight,
                        "best_iteration": best_iteration(model, member.spec),
                        "train_rows": segment_model.train_rows,
                        "valid_rows": segment_model.valid_rows,
                    }
                )
    return rows


def summarize_importance(raw: pd.DataFrame, *, n_folds: int) -> pd.DataFrame:
    if raw.empty:
        return raw
    summary = (
        raw.groupby("feature", as_index=False)
        .agg(
            weighted_gain_share=("weighted_gain_share", lambda s: float(s.sum() / n_folds)),
            weighted_split_share=("weighted_split_share", lambda s: float(s.sum() / n_folds)),
            gain_sum=("gain", "sum"),
            split_sum=("split", "sum"),
            mean_gain_share=("gain_share", "mean"),
            mean_split_share=("split_share", "mean"),
            appearances=("feature", "count"),
            used_models=("split", lambda s: int((s > 0).sum())),
        )
        .sort_values("weighted_gain_share", ascending=False)
    )
    return summary


def permute_same_slot(
    valid_df: pd.DataFrame,
    *,
    col: str,
    time_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = valid_df.copy()
    if col not in out.columns:
        return out
    slot = out[time_col].dt.hour * 4 + out[time_col].dt.minute // 15
    for _, idx in out.groupby(slot, sort=False).groups.items():
        idx_list = list(idx)
        if len(idx_list) <= 1:
            continue
        values = out.loc[idx_list, col].to_numpy(copy=True)
        rng.shuffle(values)
        out.loc[idx_list, col] = values
    return out


def permutation_rows(
    *,
    cfg: dict,
    raw_df: pd.DataFrame,
    bundles: list[FoldBundle],
    baseline_fold_rows: pd.DataFrame,
    features: list[str],
    seed: int,
) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    rows: list[dict] = []
    base_by_fold = baseline_fold_rows.set_index("fold").to_dict(orient="index")
    for feature in features:
        for fold_idx, bundle in enumerate(bundles):
            rng = np.random.default_rng(seed + 1009 * fold_idx + 9176 * features.index(feature))
            preds = []
            weights = []
            for member in bundle.members:
                member_uses_feature = any(feature in model.feature_cols for model in member.models)
                valid_df = member.valid_df
                if member_uses_feature:
                    valid_df = permute_same_slot(
                        valid_df,
                        col=feature,
                        time_col=time_col,
                        rng=rng,
                    )
                preds.append(predict_member(member, valid_df))
                weights.append(member.weight)
            pred = np.average(np.vstack(preds), axis=0, weights=np.asarray(weights))
            _, metrics, mae, rmse, curve_z_rmse = evaluate_prediction(
                bundle.valid_df,
                pred=pred,
                cfg=cfg,
                raw_df=raw_df,
                fold=next(f for f in cfg["folds"] if f["name"] == bundle.fold),
            )
            base = base_by_fold[bundle.fold]
            rows.append(
                {
                    "feature": feature,
                    "fold": bundle.fold,
                    "permutation_mode": "same_slot",
                    "baseline_avg_profit": base["avg_profit"],
                    "permuted_avg_profit": metrics["avg_profit"],
                    "delta_avg_profit": metrics["avg_profit"] - base["avg_profit"],
                    "baseline_oracle_ratio": base["oracle_ratio"],
                    "permuted_oracle_ratio": metrics["oracle_ratio"],
                    "delta_oracle_ratio": metrics["oracle_ratio"] - base["oracle_ratio"],
                    "baseline_loss_days": base["loss_days"],
                    "permuted_loss_days": metrics["loss_days"],
                    "delta_loss_days": metrics["loss_days"] - base["loss_days"],
                    "baseline_rmse": base["rmse"],
                    "permuted_rmse": rmse,
                    "delta_rmse": rmse - base["rmse"],
                    "permuted_mae": mae,
                    "permuted_curve_z_rmse": curve_z_rmse,
                }
            )
    return pd.DataFrame(rows)


def summarize_permutation(permutation: pd.DataFrame) -> pd.DataFrame:
    if permutation.empty:
        return permutation
    return (
        permutation.groupby("feature", as_index=False)
        .agg(
            delta_avg_profit_mean=("delta_avg_profit", "mean"),
            delta_avg_profit_min=("delta_avg_profit", "min"),
            delta_oracle_ratio_mean=("delta_oracle_ratio", "mean"),
            delta_loss_days_total=("delta_loss_days", "sum"),
            delta_rmse_mean=("delta_rmse", "mean"),
        )
        .sort_values("delta_avg_profit_mean")
    )


def markdown_table(df: pd.DataFrame, *, max_rows: int = 20, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "(empty)"
    show = df.head(max_rows).copy()
    cols = list(show.columns)

    def fmt(value: object) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return "nan"
            return format(value, floatfmt)
        if isinstance(value, np.floating):
            return format(float(value), floatfmt)
        return str(value)

    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_report(
    *,
    path: Path,
    config_path: Path,
    importance_summary: pd.DataFrame,
    member_importance: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    drop_summary: pd.DataFrame,
) -> None:
    raw_drop = drop_summary[drop_summary["run_name"].str.startswith("drop_raw__")].copy()
    time_drop = drop_summary[drop_summary["run_name"].str.startswith("drop_time_")].copy()
    lines = [
        "# Champion Feature Importance And Ablation",
        "",
        f"- Config: `{config_path}`.",
        "- Scope: current champion 5fold ensemble.",
        "- Importance: LightGBM gain/split, normalized by ensemble member weight "
        "and segment coverage.",
        "- Permutation: validation feature values shuffled within the same 15-minute "
        "slot across days.",
        "- Ablation: retrain the full ensemble after removing candidate features.",
        "",
        "## Top Weighted Gain Features",
        "",
        markdown_table(
            importance_summary[
                [
                    "feature",
                    "weighted_gain_share",
                    "weighted_split_share",
                    "gain_sum",
                    "split_sum",
                    "used_models",
                ]
            ],
            max_rows=25,
        ),
        "",
        "## Member/Segment Raw-Feature Importance",
        "",
        markdown_table(member_importance, max_rows=35),
        "",
        "## Same-Slot Permutation Summary",
        "",
        "Negative `delta_avg_profit_mean` means shuffling the feature hurt profit, "
        "so the feature is useful under this diagnostic.",
        "",
        markdown_table(permutation_summary, max_rows=20),
        "",
        "## Raw Feature Drop Backtest",
        "",
        "Positive delta means removing the feature improved validation profit; "
        "negative delta means removing it hurt.",
        "",
        markdown_table(
            raw_drop[
                [
                    "run_name",
                    "drop_cols",
                    "avg_profit_fold_mean",
                    "delta_avg_profit_fold_mean",
                    "avg_profit_day_weighted",
                    "delta_avg_profit_day_weighted",
                    "loss_days_total",
                    "delta_loss_days_total",
                    "rmse_mean",
                    "delta_rmse_mean",
                    "curve_z_rmse_mean",
                    "delta_curve_z_rmse_mean",
                ]
            ],
            max_rows=20,
        ),
        "",
        "## Time Feature Group Drop Backtest",
        "",
        markdown_table(
            time_drop[
                [
                    "run_name",
                    "drop_cols",
                    "avg_profit_fold_mean",
                    "delta_avg_profit_fold_mean",
                    "avg_profit_day_weighted",
                    "delta_avg_profit_day_weighted",
                    "loss_days_total",
                    "delta_loss_days_total",
                    "rmse_mean",
                    "delta_rmse_mean",
                ]
            ],
            max_rows=20,
        ),
        "",
        "## Interpretation Rules",
        "",
        "- Gain/split 高只能说明模型经常用它，不等价于删掉会掉分。",
        "- Permutation 可能破坏特征之间的物理相关性，所以只作为方向信号。",
        "- 删特征回测最接近最终目标，因为它完整重训并经过 dispatch/profit 评估。",
        "- 如果某特征 gain 高、permutation 后收益下降、删除后也下降，说明它是稳定正贡献。",
        "- 如果某特征 gain 低、permutation 后收益上升、删除后也上升，"
        "才可以考虑从 champion 中移除。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ensemble_champion_segmented6_prior_5fold.yaml")
    parser.add_argument("--material-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--no-time-drop-groups", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    if args.material_dir:
        cfg["paths"]["material_dir"] = args.material_dir
        cfg["paths"]["data_zip"] = str(Path(args.material_dir) / "to_sais_new.zip")
    if args.reports_dir:
        cfg["paths"]["reports_dir"] = args.reports_dir

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]

    specs, weights = prepare_specs(cfg)
    unsupported = sorted({model_backend(spec) for spec in specs} - {"lightgbm"})
    if unsupported:
        raise ValueError(f"importance script currently supports LightGBM only: {unsupported}")

    raw_df = load_train_frame_fast(cfg)
    print(
        f"loaded rows={len(raw_df)} range={raw_df[time_col].min()}..{raw_df[time_col].max()}",
        flush=True,
    )
    member_frames = [
        _model_spec_frame(raw_df, spec, time_col=time_col, target_col=target_col)
        for spec in specs
    ]
    member_feature_cols = [configured_feature_columns(spec) for spec in specs]
    member_target_cols = [
        _target_col_for_mode(spec["model"].get("target_mode", "absolute")) for spec in specs
    ]

    print("training baseline folds for importance/permutation", flush=True)
    baseline_bundles: list[FoldBundle] = []
    fold_rows: list[dict] = []
    importance_rows: list[dict] = []
    for fold in cfg["folds"]:
        print(f"baseline fold={fold['name']}", flush=True)
        bundle = train_ensemble_fold(
            cfg=cfg,
            raw_df=raw_df,
            specs=specs,
            weights=weights,
            member_frames=member_frames,
            member_feature_cols=member_feature_cols,
            member_target_cols=member_target_cols,
            fold=fold,
            drop_cols=set(),
        )
        baseline_bundles.append(bundle)
        fold_rows.append(summarize_fold(run_name="baseline", drop_cols=[], bundle=bundle))
        importance_rows.extend(booster_importance_rows(bundle))

    importance_raw = pd.DataFrame(importance_rows)
    importance_raw_path = reports_dir / "champion_feature_importance_raw.csv"
    importance_raw.to_csv(importance_raw_path, index=False)
    importance_summary = summarize_importance(importance_raw, n_folds=len(cfg["folds"]))
    importance_summary_path = reports_dir / "champion_feature_importance_summary.csv"
    importance_summary.to_csv(importance_summary_path, index=False)

    raw_member_importance = importance_raw[importance_raw["feature"].isin(RAW_BOUNDARY_FEATURES)]
    member_importance = (
        raw_member_importance.groupby(["member_name", "segment", "feature"], as_index=False)
        .agg(weighted_gain_share=("weighted_gain_share", "sum"), split_sum=("split", "sum"))
        .sort_values(
            ["member_name", "segment", "weighted_gain_share"],
            ascending=[True, True, False],
        )
    )
    member_importance["weighted_gain_share"] = (
        member_importance["weighted_gain_share"] / len(cfg["folds"])
    )
    member_importance_path = reports_dir / "champion_feature_importance_by_member.csv"
    member_importance.to_csv(member_importance_path, index=False)

    baseline_fold_rows = pd.DataFrame(fold_rows)
    permutation_summary = pd.DataFrame()
    if not args.skip_permutation:
        print("running same-slot permutation importance", flush=True)
        permutation = permutation_rows(
            cfg=cfg,
            raw_df=raw_df,
            bundles=baseline_bundles,
            baseline_fold_rows=baseline_fold_rows,
            features=RAW_BOUNDARY_FEATURES,
            seed=args.seed,
        )
        permutation_path = reports_dir / "champion_permutation_importance.csv"
        permutation.to_csv(permutation_path, index=False)
        permutation_summary = summarize_permutation(permutation)
        permutation_summary_path = reports_dir / "champion_permutation_importance_summary.csv"
        permutation_summary.to_csv(permutation_summary_path, index=False)

    if not args.skip_ablation:
        drop_candidates: list[tuple[str, list[str]]] = [
            (f"drop_raw__{feature}", [feature]) for feature in RAW_BOUNDARY_FEATURES
        ]
        if not args.no_time_drop_groups:
            drop_candidates.extend((name, cols) for name, cols in TIME_DROP_GROUPS.items())
        for run_name, drop_cols in drop_candidates:
            print(f"ablation run={run_name}", flush=True)
            for fold in cfg["folds"]:
                print(f"  fold={fold['name']}", flush=True)
                bundle = train_ensemble_fold(
                    cfg=cfg,
                    raw_df=raw_df,
                    specs=specs,
                    weights=weights,
                    member_frames=member_frames,
                    member_feature_cols=member_feature_cols,
                    member_target_cols=member_target_cols,
                    fold=fold,
                    drop_cols=set(drop_cols),
                )
                fold_rows.append(
                    summarize_fold(run_name=run_name, drop_cols=drop_cols, bundle=bundle)
                )

    fold_rows_df = pd.DataFrame(fold_rows)
    fold_rows_path = reports_dir / "champion_feature_drop_backtest_by_fold.csv"
    fold_rows_df.to_csv(fold_rows_path, index=False)
    drop_summary = summarize_backtest(fold_rows_df)
    drop_summary_path = reports_dir / "champion_feature_drop_backtest.csv"
    drop_summary.to_csv(drop_summary_path, index=False)

    report_path = reports_dir / "champion_feature_importance.md"
    write_report(
        path=report_path,
        config_path=config_path,
        importance_summary=importance_summary,
        member_importance=member_importance,
        permutation_summary=permutation_summary,
        drop_summary=drop_summary,
    )

    print(f"importance_raw={importance_raw_path}", flush=True)
    print(f"importance_summary={importance_summary_path}", flush=True)
    print(f"member_importance={member_importance_path}", flush=True)
    print(f"drop_summary={drop_summary_path}", flush=True)
    print(f"drop_by_fold={fold_rows_path}", flush=True)
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
