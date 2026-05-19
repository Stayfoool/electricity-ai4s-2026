"""Linear model adapters for price prediction.

Two kinds are supported via ``cfg["model"]["kind"]``:

- ``ridge``: classic L2-regularised linear regression over all
  ``feature_cols``.
- ``quarter_dummy_shared``: ``y = alpha_q + beta . x_lin`` where
  ``alpha_q`` is a per-quarter intercept (96 of them by default) and
  ``beta`` is a single shared slope vector over ``params.linear_cols``.
  Implemented with a closed-form solve so it is O(seconds) on full data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def train_linear(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,  # noqa: ARG001 - kept for API symmetry
    *,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> dict[str, Any]:
    return _train_full_or_split(train_df, feature_cols, target_col, cfg)


def train_linear_full(
    train_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> dict[str, Any]:
    return _train_full_or_split(train_df, feature_cols, target_col, cfg)


def predict_linear(
    model: dict[str, Any],
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    kind = model["kind"]
    if kind == "ridge":
        return _predict_ridge(model, df, feature_cols)
    if kind == "quarter_dummy_shared":
        return _predict_qd_shared(model, df)
    raise ValueError(f"unsupported linear kind={kind}")


def _train_full_or_split(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> dict[str, Any]:
    kind = str(cfg["model"].get("kind", "ridge"))
    params = dict(cfg["model"].get("params", {}))
    if kind == "ridge":
        return _train_ridge(train_df, feature_cols, target_col, params)
    if kind == "quarter_dummy_shared":
        return _train_qd_shared(train_df, target_col, params)
    raise ValueError(f"unsupported linear kind={kind}")


def _train_ridge(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    params: dict,
) -> dict[str, Any]:
    from sklearn.linear_model import Ridge

    alpha = float(params.get("alpha", 1.0))
    estimator = Ridge(alpha=alpha, fit_intercept=True)
    estimator.fit(
        train_df[feature_cols].to_numpy(dtype=float),
        train_df[target_col].to_numpy(dtype=float),
    )
    return {"kind": "ridge", "estimator": estimator, "feature_cols": list(feature_cols)}


def _predict_ridge(
    model: dict[str, Any],
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    cols = model.get("feature_cols", feature_cols)
    return np.asarray(
        model["estimator"].predict(df[cols].to_numpy(dtype=float)),
        dtype=float,
    )


def _train_qd_shared(
    train_df: pd.DataFrame,
    target_col: str,
    params: dict,
) -> dict[str, Any]:
    dummy_col = str(params.get("dummy_col", "quarter"))
    linear_cols = list(params.get("linear_cols", ["bid_space"]))
    n_cats = int(params.get("n_categories", 96))
    alpha = float(params.get("alpha", 0.0))

    if dummy_col not in train_df.columns:
        raise KeyError(f"dummy_col={dummy_col!r} not found in training frame")
    missing = [c for c in linear_cols if c not in train_df.columns]
    if missing:
        raise KeyError(f"linear_cols missing from training frame: {missing}")

    q = train_df[dummy_col].to_numpy()
    if not np.issubdtype(q.dtype, np.integer):
        q = q.astype(int)
    if q.min() < 0 or q.max() >= n_cats:
        raise ValueError(
            f"dummy_col values must be in [0, {n_cats}), got "
            f"[{q.min()}, {q.max()}]"
        )
    n = q.shape[0]
    p = len(linear_cols)

    # design matrix X = [Q (n x n_cats one-hot, no intercept) | X_lin]
    Q = np.zeros((n, n_cats), dtype=float)
    Q[np.arange(n), q] = 1.0
    X_lin = train_df[linear_cols].to_numpy(dtype=float).reshape(n, p)
    X = np.concatenate([Q, X_lin], axis=1)
    y = train_df[target_col].to_numpy(dtype=float)

    if alpha > 0.0:
        A = X.T @ X + alpha * np.eye(X.shape[1])
        beta = np.linalg.solve(A, X.T @ y)
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    return {
        "kind": "quarter_dummy_shared",
        "beta": beta,
        "dummy_col": dummy_col,
        "linear_cols": linear_cols,
        "n_cats": n_cats,
    }


def _predict_qd_shared(model: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    n_cats = int(model["n_cats"])
    dummy_col = str(model["dummy_col"])
    linear_cols = list(model["linear_cols"])
    q = df[dummy_col].to_numpy()
    if not np.issubdtype(q.dtype, np.integer):
        q = q.astype(int)
    n = q.shape[0]
    p = len(linear_cols)
    Q = np.zeros((n, n_cats), dtype=float)
    Q[np.arange(n), q] = 1.0
    X_lin = df[linear_cols].to_numpy(dtype=float).reshape(n, p)
    X = np.concatenate([Q, X_lin], axis=1)
    return np.asarray(X @ model["beta"], dtype=float)
