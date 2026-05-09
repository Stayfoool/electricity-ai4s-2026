from __future__ import annotations

from electricity.models import model_backend


def test_model_backend_defaults_to_lightgbm() -> None:
    assert model_backend({"model": {"name": "lgb_baseline"}}) == "lightgbm"


def test_model_backend_can_infer_xgboost_and_catboost() -> None:
    assert model_backend({"model": {"name": "xgb_baseline"}}) == "xgboost"
    assert model_backend({"model": {"name": "cat_baseline"}}) == "catboost"


def test_model_backend_explicit_backend_wins() -> None:
    assert model_backend({"model": {"name": "custom", "backend": "xgboost"}}) == "xgboost"
