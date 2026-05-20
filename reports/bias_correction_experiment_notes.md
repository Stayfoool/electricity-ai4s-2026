# Bias-correction experiment (2026-05-20)

## Motivation

EDA on 2025 boundary data shows forecast bias is strongly time-of-day-dependent
(solar over-predicts by +0.4 at noon, ~0 at night; non-market gen consistently
~+0.08). A constant per-day offset is dispatch-neutral (rank-invariant), but
the within-day component W distorts the intraday rank of net-load forecasts,
which we hypothesized was causing the model to pick wrong charge / discharge
slots (observed top-8 / bottom-8 dispatch slot agreement: only ~50%).

## Setup

- Base: `configs/ensemble_champion_segmented6_prior_5fold.yaml` (3-member
  ensemble, 5 rolling folds including a cross-year `valid_2025_jan_feb` fold).
- All three variants below fit a fold-aware `(channel, hour, month)` bias
  table on rows with `time <= train_end` (leak-free) and apply it to both
  train and valid.

| Variant | Config | shrink | mode | feature count |
|---|---|---|---|---|
| augment | `..._clean_features_5fold.yaml` | 1.0 | augment | 23 (17+6) |
| shrink05 | `..._clean_features_shrink05_5fold.yaml` | 0.5 | augment | 23 |
| replace | `..._replace_features_5fold.yaml` | 1.0 | replace | 18 (17+1) |

`augment` adds `_debiased` columns alongside raw forecasts. `replace`
overwrites raw forecast columns with debiased values (keeping feature count
constant). `shrink05` is a partial-correction probe.

## Results (avg_profit, delta vs base)

| Fold | base | augment | shrink05 | replace |
|---|---|---|---|---|
| 2025-09 | 11310 | −577 | −594 | **−773** |
| 2025-10 | 7899 | −340 | −640 | **−774** |
| 2025-11 | 7121 | −173 | −317 | −245 |
| 2025-12 | 5928 | +56 | −82 | +142 |
| **jan_feb (cross-year)** | 14153 | **+285** | +161 | +194 |
| **5-fold mean** | 9282 | **−150** | −294 | −291 |

Oracle ratio mean delta: augment −0.013, shrink05 −0.025, replace −0.023.

## Findings

1. **Cross-year fold (jan_feb) is consistently positive** across all three
   variants (+161 to +285 profit, +1.0% to +1.7% oracle ratio). This is a
   real signal — bias correction does help when train and test distributions
   are time-shifted.

2. **Close-time folds (09, 10, 11) consistently lose** (−173 to −774). This
   is true regardless of whether we augment or replace, and regardless of
   shrink. The feature-redundancy hypothesis (that adding `_debiased` columns
   alongside raw fragments LightGBM splits) is **falsified by the replace
   variant**, which has fewer features but the worst close-fold losses.

3. Root cause is more likely **the bias table itself overfits to training
   distribution's high-frequency time-of-day noise**. Each (hour, month)
   bucket has ~100 samples; the lookup table memorizes noise that doesn't
   generalize within the same regime.

## Decision

**Rejected for leaderboard.** The 5-fold mean is negative in all three
variants; the only positive signal (cross-year fold) is smaller in magnitude
than the close-fold losses. We cannot guarantee that the Jan-Feb 2026
leaderboard test distribution is more like the cross-year fold than the
adjacent-month folds.

Code is kept (feature flag `feature_sets.bias_correction` defaults off) for
potential future use with smarter bucket regularization or as a per-fold
prediction-time correction rather than a feature.

## Artifacts

- `reports/backtest_ens_champion_segmented6_prior_5fold.csv` (base)
- `reports/backtest_ens_champion_segmented6_prior_clean_features_5fold.csv`
- `reports/backtest_ens_champion_segmented6_prior_clean_features_shrink05_5fold.csv`
- `reports/backtest_ens_champion_segmented6_prior_replace_features_5fold.csv`
- `src/electricity/features/bias_correction.py`
- `tests/test_bias_correction.py` (12 unit tests)
