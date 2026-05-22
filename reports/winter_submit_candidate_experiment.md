# Winter Submit Candidate Experiment

## Purpose

The public test period is January-February 2026-like. This experiment checks whether winter-oriented components should replace or blend into the current champion submit path.

The comparison uses the 5-fold validation regime summary, especially:

- `jan_feb_like`: January-February 2025 pseudo-test fold.
- `winter_11_12_jan_feb`: November, December, and Jan-Feb-like days.
- `late_winter_12_jan_feb`: December plus Jan-Feb-like days.
- `standard_09_12`: the existing September-December rolling validation view.
- `all_5fold`: all five validation folds.

## Compared Candidates

| candidate | config | intent |
|---|---|---|
| champion | `configs/ensemble_champion_segmented6_prior_5fold.yaml` | current robust baseline |
| holiday_only | `configs/ensemble_champion_segmented6_prior_holiday_only_5fold.yaml` | test whether holiday/calendar-only variant is more winter-like |
| weekly_delta_percentile | `configs/ensemble_champion_segmented6_prior_weekly_delta_percentile_5fold.yaml` | safe weekly-relative forecast-input features |
| blend_holiday20 | `configs/ensemble_champion_segmented6_prior_blend_holiday20_5fold.yaml` | 80% champion + 20% holiday_only |
| blend_weekly20 | `configs/ensemble_champion_segmented6_prior_blend_weekly20_5fold.yaml` | 80% champion + 20% weekly_delta_percentile |
| blend_holiday15_weekly15 | `configs/ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold.yaml` | 70% champion + 15% holiday_only + 15% weekly_delta_percentile |

## Key Results

| candidate | standard_09_12 | jan_feb_like | winter_11_12_jan_feb | late_winter_12_jan_feb | all_5fold |
|---|---:|---:|---:|---:|---:|
| champion | 7991.351 | 14153.500 | 10170.888 | 11222.654 | 9952.035 |
| holiday_only | 7572.295 | 14370.663 | 10221.720 | 11356.475 | 9735.412 |
| weekly_delta_percentile | 7491.406 | 14164.877 | 10157.776 | 11389.943 | 9614.784 |
| blend_holiday20 | 7776.892 | 14210.010 | 10193.512 | 11269.951 | 9823.793 |
| blend_weekly20 | 7774.096 | 14149.653 | 10118.775 | 11177.704 | 9802.682 |
| blend_holiday15_weekly15 | 7790.477 | 14213.770 | 10141.630 | 11221.729 | 9834.252 |

## Interpretation

- `holiday_only` is the best Jan-Feb-like candidate: `14370.663`, about `+217.163` above champion.
- `holiday_only` is also best on `winter_11_12_jan_feb`: `10221.720`, about `+50.832` above champion.
- `weekly_delta_percentile` is best on `late_winter_12_jan_feb`: `11389.943`, about `+167.289` above champion, but it is much weaker on `standard_09_12` and `all_5fold`.
- The conservative blends do not dominate. They improve Jan-Feb-like relative to champion, but they dilute the strongest holiday-only signal and still lose the broader 5fold view.
- `blend_weekly20` is not useful: it is slightly below champion on Jan-Feb-like and weaker elsewhere.

## Decision

Do not replace the current champion globally based only on these blends.

For final submission candidates, keep two separate submit paths instead of forcing one blended model:

1. `champion`: safest broad-validation candidate.
2. `holiday_only`: winter-biased candidate, justified by Jan-Feb-like validation.

The next promotion step should generate final `output.csv` files for both candidates and, if submission quota allows, prefer submitting both or using the leaderboard to choose between the broad-safe and winter-biased variants.

## Artifacts

- `reports/validation_regime_summary.csv`
- `reports/validation_regime_summary.md`
- `reports/backtest_ensemble_champion_segmented6_prior_blend_holiday20_5fold.csv`
- `reports/backtest_ensemble_champion_segmented6_prior_blend_holiday20_5fold_daily.csv`
- `reports/backtest_ensemble_champion_segmented6_prior_blend_weekly20_5fold.csv`
- `reports/backtest_ensemble_champion_segmented6_prior_blend_weekly20_5fold_daily.csv`
- `reports/backtest_ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold.csv`
- `reports/backtest_ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold_daily.csv`
