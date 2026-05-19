from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from electricity.models.linear import (
    predict_linear,
    train_linear,
    train_linear_full,
)


def _make_frame(n_per_q: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(96):
        for _ in range(n_per_q):
            x = float(rng.normal(0.0, 1.0))
            # ground truth: alpha_q + beta * x with beta=2.5
            y = float(q * 0.1) + 2.5 * x + float(rng.normal(0.0, 1e-6))
            rows.append({"quarter": q, "bid_space": x, "A": y})
    return pd.DataFrame(rows)


def test_ridge_fits_and_predicts_known_linear() -> None:
    rng = np.random.default_rng(1)
    n = 500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 3.0 * x2 + rng.normal(scale=1e-6, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "A": y})
    cfg = {
        "model": {
            "name": "linear_l1_test",
            "backend": "linear",
            "kind": "ridge",
            "params": {"alpha": 1e-6},
        }
    }

    model = train_linear(
        df.iloc[:400], df.iloc[400:], feature_cols=["x1", "x2"], target_col="A", cfg=cfg
    )
    preds = predict_linear(model, df.iloc[400:], feature_cols=["x1", "x2"])

    assert preds.shape == (100,)
    assert np.allclose(preds, df.iloc[400:]["A"].to_numpy(), atol=1e-3)
    assert model["kind"] == "ridge"


def test_quarter_dummy_shared_recovers_known_intercepts_and_slope() -> None:
    df = _make_frame(n_per_q=8, seed=42)
    cfg = {
        "model": {
            "name": "linear_l2_test",
            "backend": "linear",
            "kind": "quarter_dummy_shared",
            "params": {
                "alpha": 0.0,
                "dummy_col": "quarter",
                "linear_cols": ["bid_space"],
                "n_categories": 96,
            },
        }
    }

    model = train_linear_full(
        df, feature_cols=["quarter", "bid_space"], target_col="A", cfg=cfg
    )
    beta = model["beta"]
    assert beta.shape == (97,)
    # last entry is shared slope
    assert beta[-1] == pytest.approx(2.5, abs=1e-3)
    # first 96 entries are per-quarter intercepts
    expected_intercepts = np.arange(96) * 0.1
    np.testing.assert_allclose(beta[:96], expected_intercepts, atol=1e-3)

    preds = predict_linear(model, df, feature_cols=["quarter", "bid_space"])
    assert preds.shape == (len(df),)
    np.testing.assert_allclose(preds, df["A"].to_numpy(), atol=1e-3)


def test_quarter_dummy_shared_matches_hand_computed_design_matrix() -> None:
    df = pd.DataFrame(
        {
            "quarter": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "bid_space": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 0.5, 1.0, 1.5],
            "A": [10.0, 20.0, 30.0, 11.0, 21.0, 31.0, 9.5, 19.5, 29.5],
        }
    )
    cfg = {
        "model": {
            "name": "linear_l2_test",
            "backend": "linear",
            "kind": "quarter_dummy_shared",
            "params": {
                "alpha": 0.0,
                "dummy_col": "quarter",
                "linear_cols": ["bid_space"],
                "n_categories": 3,
            },
        }
    }

    model = train_linear_full(
        df, feature_cols=["quarter", "bid_space"], target_col="A", cfg=cfg
    )
    preds = predict_linear(model, df, feature_cols=["quarter", "bid_space"])

    # Hand-built equivalent design matrix solve
    n = len(df)
    Q = np.zeros((n, 3))
    Q[np.arange(n), df["quarter"].to_numpy()] = 1.0
    X = np.concatenate([Q, df[["bid_space"]].to_numpy()], axis=1)
    beta_ref, *_ = np.linalg.lstsq(X, df["A"].to_numpy(), rcond=None)
    preds_ref = X @ beta_ref

    np.testing.assert_allclose(preds, preds_ref, atol=1e-9)
