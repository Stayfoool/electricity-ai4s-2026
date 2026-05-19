from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from electricity.models.lgbm import predict_lgbm, train_lgbm, train_lgbm_full
from electricity.models.linear import predict_linear, train_linear, train_linear_full


def model_backend(cfg: dict) -> str:
    backend = cfg["model"].get("backend")
    if backend:
        return str(backend)
    name = str(cfg["model"].get("name", ""))
    if name.startswith("linear"):
        return "linear"
    if name.startswith("xgb"):
        return "xgboost"
    if name.startswith("cat"):
        return "catboost"
    return "lightgbm"


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> Any:
    backend = model_backend(cfg)
    if backend == "lightgbm":
        return train_lgbm(
            train_df,
            valid_df,
            feature_cols=feature_cols,
            target_col=target_col,
            cfg=cfg,
        )
    if backend == "xgboost":
        return _train_xgboost(
            train_df,
            valid_df,
            feature_cols=feature_cols,
            target_col=target_col,
            cfg=cfg,
        )
    if backend == "catboost":
        return _train_catboost(
            train_df,
            valid_df,
            feature_cols=feature_cols,
            target_col=target_col,
            cfg=cfg,
        )
    if backend == "linear":
        return train_linear(
            train_df,
            valid_df,
            feature_cols=feature_cols,
            target_col=target_col,
            cfg=cfg,
        )
    raise ValueError(f"unsupported model backend={backend}")


def train_model_full(
    train_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> Any:
    backend = model_backend(cfg)
    if backend == "lightgbm":
        return train_lgbm_full(train_df, feature_cols=feature_cols, target_col=target_col, cfg=cfg)
    if backend == "xgboost":
        return _train_xgboost_full(
            train_df,
            feature_cols=feature_cols,
            target_col=target_col,
            cfg=cfg,
        )
    if backend == "catboost":
        return _train_catboost_full(
            train_df,
            feature_cols=feature_cols,
            target_col=target_col,
            cfg=cfg,
        )
    if backend == "linear":
        return train_linear_full(
            train_df,
            feature_cols=feature_cols,
            target_col=target_col,
            cfg=cfg,
        )
    raise ValueError(f"unsupported model backend={backend}")


def predict_model(model: Any, df: pd.DataFrame, feature_cols: list[str], cfg: dict) -> np.ndarray:
    backend = model_backend(cfg)
    if backend == "lightgbm":
        return predict_lgbm(model, df, feature_cols)
    if backend == "xgboost":
        return _predict_xgboost(model, df, feature_cols)
    if backend == "catboost":
        return np.asarray(model.predict(df[feature_cols]), dtype=float)
    if backend == "linear":
        return predict_linear(model, df, feature_cols)
    raise ValueError(f"unsupported model backend={backend}")


def best_iteration(model: Any, cfg: dict) -> int:
    backend = model_backend(cfg)
    if backend == "linear":
        return 0
    default = int(cfg["model"]["num_boost_round"])
    if backend == "lightgbm":
        return int(model.best_iteration or default)
    if backend == "xgboost":
        value = getattr(model, "best_iteration", None)
        return int(value + 1) if value is not None else default
    if backend == "catboost":
        value = model.get_best_iteration()
        return int(value + 1) if value is not None and value >= 0 else default
    return default


def _xgb_params(cfg: dict) -> dict:
    params = dict(cfg["model"]["params"])
    params.setdefault("seed", 2026)
    params.setdefault("verbosity", 0)
    return params


def _train_xgboost(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> Any:
    import xgboost as xgb

    dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df[target_col].to_numpy())
    dvalid = xgb.DMatrix(valid_df[feature_cols], label=valid_df[target_col].to_numpy())
    return xgb.train(
        _xgb_params(cfg),
        dtrain,
        num_boost_round=int(cfg["model"]["num_boost_round"]),
        evals=[(dvalid, "valid")],
        early_stopping_rounds=int(cfg["model"]["early_stopping_rounds"]),
        verbose_eval=False,
    )


def _train_xgboost_full(
    train_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> Any:
    import xgboost as xgb

    dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df[target_col].to_numpy())
    return xgb.train(
        _xgb_params(cfg),
        dtrain,
        num_boost_round=int(cfg["model"]["num_boost_round"]),
        verbose_eval=False,
    )


def _predict_xgboost(model: Any, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    import xgboost as xgb

    dtest = xgb.DMatrix(df[feature_cols])
    best = getattr(model, "best_iteration", None)
    if best is None:
        return np.asarray(model.predict(dtest), dtype=float)
    return np.asarray(model.predict(dtest, iteration_range=(0, int(best) + 1)), dtype=float)


def _cat_params(cfg: dict) -> dict:
    params = dict(cfg["model"]["params"])
    params.setdefault("random_seed", 2026)
    params.setdefault("allow_writing_files", False)
    params.setdefault("verbose", False)
    params["iterations"] = int(cfg["model"]["num_boost_round"])
    return params


def _train_catboost(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> Any:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(**_cat_params(cfg))
    model.fit(
        train_df[feature_cols],
        train_df[target_col].to_numpy(),
        eval_set=(valid_df[feature_cols], valid_df[target_col].to_numpy()),
        early_stopping_rounds=int(cfg["model"]["early_stopping_rounds"]),
        verbose=False,
    )
    return model


def _train_catboost_full(
    train_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> Any:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(**_cat_params(cfg))
    model.fit(train_df[feature_cols], train_df[target_col].to_numpy(), verbose=False)
    return model
