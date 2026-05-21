# Lag and Winter-Weight Experiment

## Purpose

This experiment evaluates three ideas from external competition discussions:

1. Add safe lag features from official forecast inputs.
2. Increase training weight for winter-like months.
3. Combine lag features with winter weighting.

The implementation intentionally avoids historical true price lags and actual boundary-condition lags, because those values are not available in the official test setup.

## Compared Configs

| Name | Config |
| --- | --- |
| champion | `configs/ensemble_champion_segmented6_prior.yaml` |
| lag | `configs/ensemble_champion_segmented6_prior_lag.yaml` |
| winter_weight | `configs/ensemble_champion_segmented6_prior_winter_weight.yaml` |
| lag_winter_weight | `configs/ensemble_champion_segmented6_prior_lag_winter_weight.yaml` |

## Summary

| model | fold_mean_profit | day_weighted_profit | worst_fold_profit | loss_days | oracle_ratio_mean | mae_mean | rmse_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| champion | 8064.630 | 7991.351 | 5928.223 | 5 | 0.707 | 0.438 | 0.693 |
| lag | 7327.843 | 7271.896 | 5690.471 | 10 | 0.652 | 0.457 | 0.702 |
| winter_weight | 7710.043 | 7651.217 | 5975.297 | 5 | 0.685 | 0.440 | 0.701 |
| lag_winter_weight | 7283.063 | 7232.926 | 5787.223 | 6 | 0.651 | 0.455 | 0.712 |

## Fold Profit

| fold | champion | lag | winter_weight | lag_winter_weight |
| --- | ---: | ---: | ---: | ---: |
| valid_2025_09 | 11310.428 | 9670.912 | 10212.931 | 9345.064 |
| valid_2025_10 | 7899.103 | 6937.710 | 7391.525 | 6886.486 |
| valid_2025_11 | 7120.767 | 7012.279 | 7260.417 | 7113.478 |
| valid_2025_12 | 5928.223 | 5690.471 | 5975.297 | 5787.223 |

## Delta vs Champion

| fold | lag | winter_weight | lag_winter_weight |
| --- | ---: | ---: | ---: |
| valid_2025_09 | -1639.516 | -1097.497 | -1965.364 |
| valid_2025_10 | -961.394 | -507.578 | -1012.617 |
| valid_2025_11 | -108.488 | +139.651 | -7.289 |
| valid_2025_12 | -237.753 | +47.074 | -141.001 |

## Conclusions

- Do not promote naive lag features. The safe 1-day and 2-day lags reduce profit, increase loss days, and lower oracle ratio.
- Do not promote the current winter-weighted config as a new champion. It slightly improves November and December but loses more in September and October.
- Do not combine lag and winter weighting in the current form. The combination is worse than both champion and winter weighting alone.
- The external suggestion is still useful, but it should be narrowed: if we continue this direction, test winter-focused validation or month-gated use, not a global replacement.

## Next Candidate Directions

1. Try robust LightGBM objectives such as Huber/MAE-like losses under the same harness.
2. Add sinusoidal time encoding if not already equivalent to current slot/hour features.
3. Run a winter-only or Jan-Feb-like validation design before trusting winter weighting for final submission.
4. Keep weather features selective: wind-speed features looked more useful after de-seasonalization than broad weather feature injection.
