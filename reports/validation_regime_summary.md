# Validation Regime Summary

This report re-summarizes daily backtest outputs under multiple validation regimes.
It is intended to reduce overfitting to a single trailing validation split.

## Regimes

- `standard_09_12`: rolling validation on September to December 2025.
- `jan_feb_like`: pseudo-test fold using January-February 2025.
- `winter_11_12_jan_feb`: winter-focused view combining November, December, and Jan-Feb-like.
- `late_winter_12_jan_feb`: stricter late-winter view combining December and Jan-Feb-like.
- `all_5fold`: all available 5-fold validation days.

## standard_09_12

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold | 120 | 7991.3509 | -15925.8974 | 4 | 0.6908 | 3576.3927 | 11045.4592 | 12.6333 | 9.9750 |
| ens_champion_segmented6_prior_5fold_cap_only | 120 | 7720.3443 | -15925.8974 | 5 | 0.6674 | 3847.3993 | 13036.3968 | 13.1667 | 10.9250 |
| ens_champion_segmented6_prior_5fold_holiday_only | 120 | 7572.2950 | -10239.8016 | 5 | 0.6546 | 3995.4487 | 11926.1808 | 13.4500 | 10.5167 |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | 120 | 7491.4065 | -15989.5246 | 6 | 0.6476 | 4076.3371 | 11432.7013 | 13.7500 | 10.2833 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 120 | 7451.4888 | -15989.5246 | 4 | 0.6442 | 4116.2548 | 13003.7233 | 13.1667 | 10.5167 |
| ens_champion_segmented6_prior_weekly_percentile_5fold | 120 | 7423.4154 | -10116.7306 | 9 | 0.6417 | 4144.3282 | 11314.1466 | 14.2250 | 10.6000 |
| ens_champion_segmented6_prior_weekly_delta_5fold | 120 | 7390.4029 | -8042.4700 | 6 | 0.6389 | 4177.3407 | 13439.7401 | 14.4083 | 10.6583 |

## jan_feb_like

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold_holiday_only | 56 | 14370.6631 | 5.7614 | 0 | 0.8628 | 2284.3912 | 8119.4717 | 14.7321 | 9.5714 |
| ens_champion_segmented6_prior_weekly_delta_5fold | 56 | 14242.1040 | 8.7837 | 0 | 0.8551 | 2412.9503 | 8076.6728 | 12.0000 | 9.8929 |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | 56 | 14164.8772 | 10.5589 | 0 | 0.8505 | 2490.1771 | 8728.8520 | 12.9821 | 10.0179 |
| ens_champion_segmented6_prior_5fold | 56 | 14153.4998 | 8.7837 | 0 | 0.8498 | 2501.5545 | 8728.8520 | 13.7321 | 9.5000 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 56 | 14053.0723 | 8.7837 | 0 | 0.8438 | 2601.9820 | 9275.2604 | 13.6071 | 9.1071 |
| ens_champion_segmented6_prior_weekly_percentile_5fold | 56 | 13687.2947 | 13.7403 | 0 | 0.8218 | 2967.7596 | 11655.9101 | 13.1607 | 9.6786 |
| ens_champion_segmented6_prior_5fold_cap_only | 56 | 13644.8040 | 5.7614 | 0 | 0.8193 | 3010.2503 | 11627.8775 | 13.9464 | 10.3036 |

## winter_11_12_jan_feb

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold_holiday_only | 117 | 10221.7204 | -10239.8016 | 1 | 0.8042 | 2488.7217 | 6992.6333 | 13.0769 | 9.3846 |
| ens_champion_segmented6_prior_5fold | 117 | 10170.8882 | -12550.1225 | 1 | 0.8002 | 2539.5540 | 7189.1142 | 12.7265 | 9.4872 |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | 117 | 10157.7760 | -5577.1532 | 2 | 0.7992 | 2552.6661 | 7514.3767 | 12.3761 | 9.8547 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 117 | 10055.0381 | -10239.8016 | 1 | 0.7911 | 2655.4040 | 7189.1142 | 12.4103 | 9.2564 |
| ens_champion_segmented6_prior_weekly_delta_5fold | 117 | 9940.9096 | -8042.4700 | 2 | 0.7821 | 2769.5326 | 6400.9000 | 12.7949 | 10.2308 |
| ens_champion_segmented6_prior_weekly_percentile_5fold | 117 | 9880.4166 | -7083.8907 | 3 | 0.7773 | 2830.0255 | 10054.9257 | 12.3761 | 9.2051 |
| ens_champion_segmented6_prior_5fold_cap_only | 117 | 9872.5342 | -7083.8907 | 2 | 0.7767 | 2837.9080 | 8038.1561 | 13.0000 | 10.3846 |

## late_winter_12_jan_feb

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | 87 | 11389.9429 | -5577.1532 | 1 | 0.7947 | 2942.0580 | 8971.7471 | 13.4253 | 10.9885 |
| ens_champion_segmented6_prior_5fold_holiday_only | 87 | 11356.4751 | -10239.8016 | 1 | 0.7924 | 2975.5259 | 8604.0822 | 14.5862 | 10.9885 |
| ens_champion_segmented6_prior_weekly_delta_5fold | 87 | 11227.3318 | -8042.4700 | 1 | 0.7834 | 3104.6691 | 10186.0602 | 13.2184 | 10.7701 |
| ens_champion_segmented6_prior_5fold | 87 | 11222.6542 | -12550.1225 | 1 | 0.7830 | 3109.3467 | 9367.5085 | 14.1494 | 11.2759 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 87 | 11183.7611 | -10239.8016 | 1 | 0.7803 | 3148.2398 | 11279.0030 | 13.8966 | 10.8966 |
| ens_champion_segmented6_prior_5fold_cap_only | 87 | 10957.8537 | -7083.8907 | 1 | 0.7646 | 3374.1472 | 12261.3961 | 14.1379 | 11.7701 |
| ens_champion_segmented6_prior_weekly_percentile_5fold | 87 | 10876.7465 | -7083.8907 | 2 | 0.7589 | 3455.2544 | 12364.0065 | 13.1609 | 10.8391 |

## all_5fold

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold | 176 | 9952.0347 | -15925.8974 | 4 | 0.7547 | 3234.3987 | 9990.9094 | 12.9830 | 9.8239 |
| ens_champion_segmented6_prior_5fold_holiday_only | 176 | 9735.4121 | -10239.8016 | 5 | 0.7383 | 3451.0213 | 11377.1067 | 13.8580 | 10.2159 |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | 176 | 9614.7835 | -15989.5246 | 6 | 0.7291 | 3571.6498 | 11013.8031 | 13.5057 | 10.1989 |
| ens_champion_segmented6_prior_5fold_cap_only | 176 | 9605.3997 | -15925.8974 | 5 | 0.7284 | 3581.0337 | 12306.8008 | 13.4148 | 10.7273 |
| ens_champion_segmented6_prior_weekly_delta_5fold | 176 | 9570.4896 | -8042.4700 | 6 | 0.7258 | 3615.9438 | 12404.8217 | 13.6420 | 10.4148 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 176 | 9551.9927 | -15989.5246 | 4 | 0.7244 | 3634.4407 | 12762.6375 | 13.3068 | 10.0682 |
| ens_champion_segmented6_prior_weekly_percentile_5fold | 176 | 9416.4679 | -10116.7306 | 9 | 0.7141 | 3769.9655 | 11370.4274 | 13.8864 | 10.3068 |

## Artifacts

- `reports/validation_regime_summary.csv`
