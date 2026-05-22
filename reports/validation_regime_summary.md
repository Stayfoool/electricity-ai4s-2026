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
| ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold | 120 | 7790.4766 | -15992.7060 | 3 | 0.6735 | 3777.2670 | 10813.2252 | 13.2000 | 9.8167 |
| ensemble_champion_segmented6_prior_blend_holiday20_5fold | 120 | 7776.8918 | -15992.7060 | 4 | 0.6723 | 3790.8519 | 11200.5147 | 12.9083 | 9.8333 |
| ensemble_champion_segmented6_prior_blend_weekly20_5fold | 120 | 7774.0963 | -15992.7060 | 3 | 0.6720 | 3793.6473 | 10813.2252 | 13.2750 | 9.7833 |
| ens_champion_segmented6_prior_shallow_d4_l15_leaf200_5fold | 120 | 7757.3524 | -11244.2975 | 4 | 0.6706 | 3810.3913 | 10698.6246 | 13.5083 | 10.5083 |
| ens_champion_segmented6_prior_shallow_d5_l31_leaf150_5fold | 120 | 7736.8000 | -15992.7060 | 4 | 0.6688 | 3830.9436 | 10758.9426 | 13.3667 | 9.8750 |
| ens_champion_segmented6_prior_5fold_cap_only | 120 | 7720.3443 | -15925.8974 | 5 | 0.6674 | 3847.3993 | 13036.3968 | 13.1667 | 10.9250 |
| ens_champion_segmented6_prior_shallow_d3_l7_leaf200_5fold | 120 | 7592.4318 | -11244.2975 | 5 | 0.6563 | 3975.3118 | 10724.5324 | 13.8750 | 10.7083 |
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
| ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold | 56 | 14213.7697 | 8.7837 | 0 | 0.8534 | 2441.2846 | 8728.8520 | 14.0714 | 9.4643 |
| ensemble_champion_segmented6_prior_blend_holiday20_5fold | 56 | 14210.0096 | 8.7837 | 0 | 0.8532 | 2445.0447 | 8728.8520 | 14.0714 | 9.5714 |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | 56 | 14164.8772 | 10.5589 | 0 | 0.8505 | 2490.1771 | 8728.8520 | 12.9821 | 10.0179 |
| ens_champion_segmented6_prior_5fold | 56 | 14153.4998 | 8.7837 | 0 | 0.8498 | 2501.5545 | 8728.8520 | 13.7321 | 9.5000 |
| ensemble_champion_segmented6_prior_blend_weekly20_5fold | 56 | 14149.6527 | 8.7837 | 0 | 0.8496 | 2505.4016 | 8728.8520 | 13.9464 | 9.5000 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 56 | 14053.0723 | 8.7837 | 0 | 0.8438 | 2601.9820 | 9275.2604 | 13.6071 | 9.1071 |
| ens_champion_segmented6_prior_shallow_d3_l7_leaf200_5fold | 56 | 13966.2662 | -5405.1338 | 1 | 0.8386 | 2688.7881 | 8923.7200 | 13.0179 | 8.9464 |
| ens_champion_segmented6_prior_shallow_d5_l31_leaf150_5fold | 56 | 13939.2760 | 5.7614 | 0 | 0.8369 | 2715.7783 | 10434.8668 | 13.6250 | 9.5179 |
| ens_champion_segmented6_prior_shallow_d4_l15_leaf200_5fold | 56 | 13916.7024 | -5405.1338 | 1 | 0.8356 | 2738.3519 | 8923.7200 | 12.6250 | 9.0536 |
| ens_champion_segmented6_prior_weekly_percentile_5fold | 56 | 13687.2947 | 13.7403 | 0 | 0.8218 | 2967.7596 | 11655.9101 | 13.1607 | 9.6786 |
| ens_champion_segmented6_prior_5fold_cap_only | 56 | 13644.8040 | 5.7614 | 0 | 0.8193 | 3010.2503 | 11627.8775 | 13.9464 | 10.3036 |

## winter_11_12_jan_feb

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold_holiday_only | 117 | 10221.7204 | -10239.8016 | 1 | 0.8042 | 2488.7217 | 6992.6333 | 13.0769 | 9.3846 |
| ensemble_champion_segmented6_prior_blend_holiday20_5fold | 117 | 10193.5121 | -12550.1225 | 1 | 0.8020 | 2516.9300 | 7189.1142 | 12.8291 | 9.3761 |
| ens_champion_segmented6_prior_5fold | 117 | 10170.8882 | -12550.1225 | 1 | 0.8002 | 2539.5540 | 7189.1142 | 12.7265 | 9.4872 |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | 117 | 10157.7760 | -5577.1532 | 2 | 0.7992 | 2552.6661 | 7514.3767 | 12.3761 | 9.8547 |
| ens_champion_segmented6_prior_shallow_d3_l7_leaf200_5fold | 117 | 10147.1475 | -11244.2975 | 2 | 0.7983 | 2563.2946 | 7056.2606 | 12.4017 | 9.3504 |
| ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold | 117 | 10141.6298 | -12550.1225 | 1 | 0.7979 | 2568.8123 | 7189.1142 | 12.7949 | 9.3162 |
| ensemble_champion_segmented6_prior_blend_weekly20_5fold | 117 | 10118.7748 | -12550.1225 | 1 | 0.7961 | 2591.6674 | 7189.1142 | 12.7949 | 9.3162 |
| ens_champion_segmented6_prior_shallow_d4_l15_leaf200_5fold | 117 | 10089.7886 | -11244.2975 | 2 | 0.7938 | 2620.6535 | 7056.2606 | 12.2051 | 9.3419 |
| ens_champion_segmented6_prior_shallow_d5_l31_leaf150_5fold | 117 | 10083.9101 | -11244.2975 | 1 | 0.7934 | 2626.5320 | 8688.2915 | 12.7179 | 9.2650 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 117 | 10055.0381 | -10239.8016 | 1 | 0.7911 | 2655.4040 | 7189.1142 | 12.4103 | 9.2564 |
| ens_champion_segmented6_prior_weekly_delta_5fold | 117 | 9940.9096 | -8042.4700 | 2 | 0.7821 | 2769.5326 | 6400.9000 | 12.7949 | 10.2308 |
| ens_champion_segmented6_prior_weekly_percentile_5fold | 117 | 9880.4166 | -7083.8907 | 3 | 0.7773 | 2830.0255 | 10054.9257 | 12.3761 | 9.2051 |
| ens_champion_segmented6_prior_5fold_cap_only | 117 | 9872.5342 | -7083.8907 | 2 | 0.7767 | 2837.9080 | 8038.1561 | 13.0000 | 10.3846 |

## late_winter_12_jan_feb

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | 87 | 11389.9429 | -5577.1532 | 1 | 0.7947 | 2942.0580 | 8971.7471 | 13.4253 | 10.9885 |
| ens_champion_segmented6_prior_5fold_holiday_only | 87 | 11356.4751 | -10239.8016 | 1 | 0.7924 | 2975.5259 | 8604.0822 | 14.5862 | 10.9885 |
| ensemble_champion_segmented6_prior_blend_holiday20_5fold | 87 | 11269.9508 | -12550.1225 | 1 | 0.7863 | 3062.0501 | 9367.5085 | 14.3678 | 11.0920 |
| ens_champion_segmented6_prior_weekly_delta_5fold | 87 | 11227.3318 | -8042.4700 | 1 | 0.7834 | 3104.6691 | 10186.0602 | 13.2184 | 10.7701 |
| ens_champion_segmented6_prior_5fold | 87 | 11222.6542 | -12550.1225 | 1 | 0.7830 | 3109.3467 | 9367.5085 | 14.1494 | 11.2759 |
| ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold | 87 | 11221.7287 | -12550.1225 | 1 | 0.7830 | 3110.2723 | 9367.5085 | 14.3678 | 10.9655 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 87 | 11183.7611 | -10239.8016 | 1 | 0.7803 | 3148.2398 | 11279.0030 | 13.8966 | 10.8966 |
| ensemble_champion_segmented6_prior_blend_weekly20_5fold | 87 | 11177.7037 | -12550.1225 | 1 | 0.7799 | 3154.2972 | 9367.5085 | 14.3678 | 10.9885 |
| ens_champion_segmented6_prior_shallow_d3_l7_leaf200_5fold | 87 | 11156.1464 | -11244.2975 | 2 | 0.7784 | 3175.8546 | 11299.5839 | 13.7241 | 10.5862 |
| ens_champion_segmented6_prior_shallow_d5_l31_leaf150_5fold | 87 | 11094.0662 | -11244.2975 | 1 | 0.7741 | 3237.9348 | 11336.1339 | 14.1724 | 10.8851 |
| ens_champion_segmented6_prior_shallow_d4_l15_leaf200_5fold | 87 | 11079.0086 | -11244.2975 | 2 | 0.7730 | 3252.9924 | 11299.5839 | 13.4368 | 10.5747 |
| ens_champion_segmented6_prior_5fold_cap_only | 87 | 10957.8537 | -7083.8907 | 1 | 0.7646 | 3374.1472 | 12261.3961 | 14.1379 | 11.7701 |
| ens_champion_segmented6_prior_weekly_percentile_5fold | 87 | 10876.7465 | -7083.8907 | 2 | 0.7589 | 3455.2544 | 12364.0065 | 13.1609 | 10.8391 |

## all_5fold

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold | 176 | 9952.0347 | -15925.8974 | 4 | 0.7547 | 3234.3987 | 9990.9094 | 12.9830 | 9.8239 |
| ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold | 176 | 9834.2517 | -15992.7060 | 3 | 0.7458 | 3352.1817 | 10080.7415 | 13.4773 | 9.7045 |
| ensemble_champion_segmented6_prior_blend_holiday20_5fold | 176 | 9823.7929 | -15992.7060 | 4 | 0.7450 | 3362.6405 | 10359.3111 | 13.2784 | 9.7500 |
| ensemble_champion_segmented6_prior_blend_weekly20_5fold | 176 | 9802.6824 | -15992.7060 | 3 | 0.7434 | 3383.7510 | 10080.7415 | 13.4886 | 9.6932 |
| ens_champion_segmented6_prior_5fold_holiday_only | 176 | 9735.4121 | -10239.8016 | 5 | 0.7383 | 3451.0213 | 11377.1067 | 13.8580 | 10.2159 |
| ens_champion_segmented6_prior_shallow_d4_l15_leaf200_5fold | 176 | 9717.1456 | -11244.2975 | 5 | 0.7369 | 3469.2878 | 10418.7628 | 13.2273 | 10.0455 |
| ens_champion_segmented6_prior_shallow_d5_l31_leaf150_5fold | 176 | 9710.3151 | -15992.7060 | 4 | 0.7364 | 3476.1183 | 10772.8881 | 13.4489 | 9.7614 |
| ens_champion_segmented6_prior_shallow_d3_l7_leaf200_5fold | 176 | 9620.4700 | -11244.2975 | 6 | 0.7296 | 3565.9634 | 10618.0401 | 13.6023 | 10.1477 |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | 176 | 9614.7835 | -15989.5246 | 6 | 0.7291 | 3571.6498 | 11013.8031 | 13.5057 | 10.1989 |
| ens_champion_segmented6_prior_5fold_cap_only | 176 | 9605.3997 | -15925.8974 | 5 | 0.7284 | 3581.0337 | 12306.8008 | 13.4148 | 10.7273 |
| ens_champion_segmented6_prior_weekly_delta_5fold | 176 | 9570.4896 | -8042.4700 | 6 | 0.7258 | 3615.9438 | 12404.8217 | 13.6420 | 10.4148 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 176 | 9551.9927 | -15989.5246 | 4 | 0.7244 | 3634.4407 | 12762.6375 | 13.3068 | 10.0682 |
| ens_champion_segmented6_prior_weekly_percentile_5fold | 176 | 9416.4679 | -10116.7306 | 9 | 0.7141 | 3769.9655 | 11370.4274 | 13.8864 | 10.3068 |

## Artifacts

- `reports/validation_regime_summary.csv`
