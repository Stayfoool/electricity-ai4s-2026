from __future__ import annotations

import numpy as np
import pandas as pd

from electricity.features.bias_correction import (
    BIAS_CORRECTED_FORECAST_COLUMNS,
    CORRECTABLE_CHANNELS,
    NET_LOAD_FEATURES,
    apply_bias_correction,
    apply_fold_bias_correction,
    bias_correction_feature_columns,
    fit_bias_table,
)


def _make_synthetic_frame(
    n_days: int = 60,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01 00:00:00", periods=n_days * 96, freq="15min")
    hour = times.hour.to_numpy()
    n = len(times)

    # Base actuals.
    load = 3.0 + 0.1 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.02, n)
    wind = 0.8 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 0.05, n)
    solar = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * 1.0 + rng.normal(0, 0.02, n)
    solar = np.clip(solar, 0, None)
    wind_solar = wind + solar
    hydro = 0.03 + rng.normal(0, 0.01, n)
    tie = 0.18 + rng.normal(0, 0.01, n)
    non_mkt = 0.65 + rng.normal(0, 0.05, n)

    # Forecasts: solar over-predicted at midday by +0.4; wind by +0.1 in afternoon.
    midday = (hour >= 10) & (hour <= 15)
    solar_fct = solar + np.where(midday, 0.4, 0.0) + rng.normal(0, 0.02, n)
    wind_fct = wind + np.where((hour >= 12) & (hour <= 14), 0.1, 0.05) + rng.normal(0, 0.05, n)
    wind_solar_fct = wind_fct + solar_fct
    non_mkt_fct = non_mkt + 0.08 + rng.normal(0, 0.02, n)

    df = pd.DataFrame({
        "times": times,
        "系统负荷实际值": load,
        "系统负荷预测值": load + rng.normal(0, 0.01, n),
        "风光总加实际值": wind_solar,
        "风光总加预测值": wind_solar_fct,
        "联络线实际值": tie,
        "联络线预测值": tie + rng.normal(0, 0.005, n),
        "风电实际值": wind,
        "风电预测值": wind_fct,
        "光伏实际值": solar,
        "光伏预测值": solar_fct,
        "水电实际值": hydro,
        "水电预测值": hydro + rng.normal(0, 0.005, n),
        "非市场化机组实际值": non_mkt,
        "非市场化机组预测值": non_mkt_fct,
    })
    return df


def test_fit_bias_table_returns_long_form_with_expected_columns():
    df = _make_synthetic_frame(n_days=60)
    table = fit_bias_table(df)
    assert set(table.columns) == {"channel", "hour", "month", "bias", "n"}
    assert set(table["channel"].unique()) == set(CORRECTABLE_CHANNELS)
    n_months = table[table["channel"] == CORRECTABLE_CHANNELS[0]]["month"].nunique()
    for c in CORRECTABLE_CHANNELS:
        sub = table[table["channel"] == c]
        assert len(sub) == 24 * n_months


def test_fit_bias_table_recovers_solar_midday_shape():
    """remove_daily_mean=True centers W per-day, so midday rises and night drops
    by the daily mean (here ~0.1). What matters for dispatch is the *contrast*
    between midday and night, which should match the injected +0.4."""
    df = _make_synthetic_frame(n_days=120)
    table = fit_bias_table(df)
    solar = table[table["channel"] == "光伏"]
    midday = solar[(solar["hour"] >= 10) & (solar["hour"] <= 15)]["bias"].mean()
    night = solar[(solar["hour"] <= 4)]["bias"].mean()
    assert midday - night > 0.35, (
        f"midday-night solar bias contrast should ~+0.4, got {midday - night:.3f}"
    )

    # remove_daily_mean=False should also recover a similar contrast.
    table_raw = fit_bias_table(df, remove_daily_mean=False)
    solar_raw = table_raw[table_raw["channel"] == "光伏"]
    midday_raw = solar_raw[(solar_raw["hour"] >= 10) & (solar_raw["hour"] <= 15)]["bias"].mean()
    night_raw = solar_raw[(solar_raw["hour"] <= 4)]["bias"].mean()
    assert midday_raw - night_raw > 0.35


def test_fit_bias_table_excludes_rows_after_train_end():
    df = _make_synthetic_frame(n_days=60)
    cutoff = df["times"].iloc[30 * 96]  # halfway
    table = fit_bias_table(df, train_end=cutoff)
    # Recompute manually on truncated frame and verify match.
    table_ref = fit_bias_table(df[df["times"] <= cutoff])
    pd.testing.assert_frame_equal(
        table.sort_values(["channel", "hour", "month"]).reset_index(drop=True),
        table_ref.sort_values(["channel", "hour", "month"]).reset_index(drop=True),
    )


def test_remove_daily_mean_yields_cleaner_within_day_signal():
    """Inject random daily offsets and check that remove_daily_mean=True
    produces a bias profile whose hour-to-hour variation closely matches the
    injected structural shape, while remove_daily_mean=False has the same
    expected shape but more noise from the daily offsets contaminating the
    bucket means."""
    df = _make_synthetic_frame(n_days=60)
    rng = np.random.default_rng(1)
    daily_offsets = rng.normal(0, 0.5, 60)
    df = df.copy()
    df["__date_idx"] = (df["times"] - df["times"].min()).dt.days
    df["光伏预测值"] = df["光伏预测值"] + daily_offsets[df["__date_idx"].to_numpy()]
    df = df.drop(columns="__date_idx")

    table_strip = fit_bias_table(df, channels=["光伏"], remove_daily_mean=True)
    table_raw = fit_bias_table(df, channels=["光伏"], remove_daily_mean=False)

    # Restrict to month=1 (well-populated) to avoid sparse-bucket fallback noise.
    s = table_strip[table_strip["month"] == 1]
    r = table_raw[table_raw["month"] == 1]

    # Both should recover the midday-night contrast (~0.4).
    contrast_strip = (
        s[(s["hour"] >= 10) & (s["hour"] <= 15)]["bias"].mean()
        - s[s["hour"] <= 4]["bias"].mean()
    )
    contrast_raw = (
        r[(r["hour"] >= 10) & (r["hour"] <= 15)]["bias"].mean()
        - r[r["hour"] <= 4]["bias"].mean()
    )
    assert contrast_strip > 0.35
    assert contrast_raw > 0.35

    # Stripped version should have lower variance across hours within
    # a known-flat region (hours 0..4 where true bias is 0 in expectation).
    flat_strip_var = s[s["hour"] <= 4]["bias"].var()
    flat_raw_var = r[r["hour"] <= 4]["bias"].var()
    assert flat_strip_var < flat_raw_var


def test_apply_bias_correction_adds_expected_columns():
    df = _make_synthetic_frame(n_days=30)
    table = fit_bias_table(df)
    out = apply_bias_correction(df, table)
    for col in BIAS_CORRECTED_FORECAST_COLUMNS:
        assert col in out.columns
    for col in NET_LOAD_FEATURES:
        assert col in out.columns
    assert len(out) == len(df)


def test_apply_bias_correction_reduces_residual_on_train():
    df = _make_synthetic_frame(n_days=120)
    table = fit_bias_table(df)
    out = apply_bias_correction(df, table)
    raw_mae = (df["光伏预测值"] - df["光伏实际值"]).abs().mean()
    corrected_mae = (out["光伏预测值_debiased"] - df["光伏实际值"]).abs().mean()
    assert corrected_mae < raw_mae


def test_shrink_zero_disables_correction():
    df = _make_synthetic_frame(n_days=30)
    table = fit_bias_table(df)
    out = apply_bias_correction(df, table, shrink=0.0)
    np.testing.assert_allclose(
        out["光伏预测值_debiased"].to_numpy(),
        df["光伏预测值"].to_numpy(),
    )


def test_apply_fold_bias_correction_noop_when_disabled():
    df = _make_synthetic_frame(n_days=30)
    spec = {"feature_sets": {"bias_correction": False}}
    out = apply_fold_bias_correction(df, spec, train_end=df["times"].max())
    assert "光伏预测值_debiased" not in out.columns
    assert out is df  # unchanged ref


def test_apply_fold_bias_correction_idempotent_drops_old_columns():
    df = _make_synthetic_frame(n_days=30)
    spec = {"feature_sets": {"bias_correction": True}}
    out1 = apply_fold_bias_correction(df, spec, train_end=df["times"].max())
    out2 = apply_fold_bias_correction(out1, spec, train_end=df["times"].max())
    np.testing.assert_allclose(
        out1["光伏预测值_debiased"].to_numpy(),
        out2["光伏预测值_debiased"].to_numpy(),
    )


def test_bias_correction_feature_columns_respects_flag():
    cfg_off = {"feature_sets": {"bias_correction": False}}
    assert bias_correction_feature_columns(cfg_off) == []
    cfg_on = {"feature_sets": {"bias_correction": True}}
    cols = bias_correction_feature_columns(cfg_on)
    for c in BIAS_CORRECTED_FORECAST_COLUMNS:
        assert c in cols
    for c in NET_LOAD_FEATURES:
        assert c in cols


def test_net_load_explicit_uses_correct_definition():
    """net_load = 系统负荷 - 风光 - 水电 - 非市场化 - 联络线."""
    df = _make_synthetic_frame(n_days=10)
    table = fit_bias_table(df)
    out = apply_bias_correction(df, table)
    expected = (
        df["系统负荷预测值"]
        - df["风光总加预测值"]
        - df["水电预测值"]
        - df["非市场化机组预测值"]
        - df["联络线预测值"]
    ).to_numpy()
    np.testing.assert_allclose(out["net_load_fct_explicit"].to_numpy(), expected)


def test_min_samples_per_bucket_fallback():
    # Force a sparse case by limiting to a single day.
    df = _make_synthetic_frame(n_days=1)
    table = fit_bias_table(df, min_samples_per_bucket=1000)
    # All buckets should have fallen back to month mean (all finite).
    assert table["bias"].notna().all()
