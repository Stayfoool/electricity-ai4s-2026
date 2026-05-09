from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


def train_lgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> lgb.Booster:
    params = dict(cfg["model"]["params"])
    train_set = lgb.Dataset(
        train_df[feature_cols],
        label=train_df[target_col].to_numpy(),
        feature_name=feature_cols,
    )
    valid_set = lgb.Dataset(
        valid_df[feature_cols],
        label=valid_df[target_col].to_numpy(),
        feature_name=feature_cols,
        reference=train_set,
    )
    callbacks = [
        lgb.early_stopping(cfg["model"]["early_stopping_rounds"], verbose=False),
        lgb.log_evaluation(period=0),
    ]
    return lgb.train(
        params,
        train_set,
        num_boost_round=cfg["model"]["num_boost_round"],
        valid_sets=[valid_set],
        callbacks=callbacks,
    )


def train_lgbm_full(
    train_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> lgb.Booster:
    params = dict(cfg["model"]["params"])
    train_set = lgb.Dataset(
        train_df[feature_cols],
        label=train_df[target_col].to_numpy(),
        feature_name=feature_cols,
    )
    return lgb.train(
        params,
        train_set,
        num_boost_round=cfg["model"]["num_boost_round"],
    )


def predict_lgbm(model: lgb.Booster, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    best_iteration = (
        model.best_iteration if model.best_iteration and model.best_iteration > 0 else None
    )
    return model.predict(df[feature_cols], num_iteration=best_iteration)
