# Online Risk Dashboard

Purpose: explain why local validation can diverge from online score and avoid submitting candidates that only win one validation regime.

## Key Readout

- `test_like_weighted_profit` uses only visible 2026 Jan-Feb test features. It reweights validation days by similarity to test-day feature signatures.
- `risk_class=baseline_champion` is the current online best reference.
- `winter_overfit_risk` means Jan-Feb-like improves but broad 9-12 validation weakens.
- This table is a promotion filter, not a substitute for online scoring.

## Candidate Dashboard

| model | risk_class | online_score | delta_vs_champion__online_score | all_5fold_mean_profit | delta_vs_champion__all_5fold_mean_profit | jan_feb_like_mean_profit | delta_vs_champion__jan_feb_like_mean_profit | standard_09_12_mean_profit | delta_vs_champion__standard_09_12_mean_profit | test_like_weighted_profit | delta_vs_champion__test_like_weighted_profit | all_5fold_loss_days | all_5fold_p90_regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold | baseline_champion | 5482.000 | 0.000 | 9952.035 | 0.000 | 14153.500 | 0.000 | 7991.351 | 0.000 | 10152.642 | 0.000 | 4 | 9990.909 |
| ens_champion_segmented6_prior_5fold_holiday_only | winter_overfit_risk | 5370.000 | -112.000 | 9735.412 | -216.623 | 14370.663 | 217.163 | 7572.295 | -419.056 | 9938.449 | -214.192 | 5 | 11377.107 |
| ensemble_champion_segmented6_prior_blend_holiday05_5fold | watch | nan | nan | 9913.540 | -38.495 | 14150.366 | -3.133 | 7936.354 | -54.997 | 10122.744 | -29.898 | 4 | 10338.632 |
| ensemble_champion_segmented6_prior_blend_weekly05_5fold | watch | nan | nan | 9874.232 | -77.803 | 14161.728 | 8.229 | 7873.400 | -117.951 | 10080.937 | -71.704 | 4 | 10338.632 |
| ensemble_champion_segmented6_prior_blend_holiday10_5fold | watch | nan | nan | 9864.428 | -87.606 | 14149.230 | -4.270 | 7864.854 | -126.497 | 10074.909 | -77.732 | 4 | 10359.311 |
| ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold | watch | nan | nan | 9834.252 | -117.783 | 14213.770 | 60.270 | 7790.477 | -200.874 | 10042.102 | -110.540 | 3 | 10080.741 |
| ensemble_champion_segmented6_prior_blend_holiday05_weekly05_5fold | watch | nan | nan | 9825.450 | -126.584 | 14161.728 | 8.229 | 7801.854 | -189.497 | 10051.375 | -101.267 | 4 | 10359.311 |
| ensemble_champion_segmented6_prior_blend_holiday20_5fold | watch | nan | nan | 9823.793 | -128.242 | 14210.010 | 56.510 | 7776.892 | -214.459 | 10051.747 | -100.895 | 4 | 10359.311 |
| ensemble_champion_segmented6_prior_blend_weekly10_5fold | watch | nan | nan | 9810.611 | -141.424 | 14162.333 | 8.833 | 7779.808 | -211.543 | 10027.241 | -125.401 | 4 | 10359.311 |
| ensemble_champion_segmented6_prior_blend_weekly20_5fold | watch | nan | nan | 9802.682 | -149.352 | 14149.653 | -3.847 | 7774.096 | -217.255 | 10006.226 | -146.416 | 3 | 10080.741 |
| ens_champion_segmented6_prior_shallow_d4_l15_leaf200_5fold | watch | nan | nan | 9717.146 | -234.889 | 13916.702 | -236.797 | 7757.352 | -233.999 | 9917.814 | -234.827 | 5 | 10418.763 |
| ens_champion_segmented6_prior_shallow_d5_l31_leaf150_5fold | watch | nan | nan | 9710.315 | -241.720 | 13939.276 | -214.224 | 7736.800 | -254.551 | 9927.569 | -225.073 | 4 | 10772.888 |
| ens_champion_segmented6_prior_feaug_ws025_w05_5fold | broad_validation_weak | nan | nan | 9650.494 | -301.540 | 13812.120 | -341.380 | 7708.402 | -282.949 | 9869.024 | -283.618 | 4 | 10774.646 |
| ens_champion_segmented6_prior_shallow_d3_l7_leaf200_5fold | loss_day_risk | nan | nan | 9620.470 | -331.565 | 13966.266 | -187.234 | 7592.432 | -398.919 | 9859.848 | -292.793 | 6 | 10618.040 |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | loss_day_risk | nan | nan | 9614.784 | -337.251 | 14164.877 | 11.377 | 7491.406 | -499.944 | 9913.591 | -239.050 | 6 | 11013.803 |
| ens_champion_segmented6_prior_5fold_cap_only | broad_validation_weak | nan | nan | 9605.400 | -346.635 | 13644.804 | -508.696 | 7720.344 | -271.007 | 9830.293 | -322.349 | 5 | 12306.801 |
| ens_champion_segmented6_prior_feaug_ws_late050_w05_5fold | broad_validation_weak | nan | nan | 9591.477 | -360.558 | 13750.738 | -402.762 | 7650.488 | -340.863 | 9808.787 | -343.855 | 5 | 11922.731 |
| ens_champion_segmented6_prior_weekly_delta_5fold | loss_day_risk | nan | nan | 9570.490 | -381.545 | 14242.104 | 88.604 | 7390.403 | -600.948 | 9811.428 | -341.213 | 6 | 12404.822 |
| ens_champion_segmented6_prior_5fold_cap_holiday | broad_validation_weak | nan | nan | 9551.993 | -400.042 | 14053.072 | -100.428 | 7451.489 | -539.862 | 9794.737 | -357.905 | 4 | 12762.637 |
| ens_champion_segmented6_prior_forecast_error_aug_5fold | broad_validation_weak | nan | nan | 9548.048 | -403.986 | 13788.616 | -364.884 | 7569.117 | -422.234 | 9769.244 | -383.397 | 3 | 11256.118 |
| ens_champion_segmented6_prior_weekly_percentile_5fold | loss_day_risk | nan | nan | 9416.468 | -535.567 | 13687.295 | -466.205 | 7423.415 | -567.936 | 9685.405 | -467.237 | 9 | 11370.427 |
| ens_champion_segmented6_prior_centered_5fold | loss_day_risk | nan | nan | 8755.415 | -1196.619 | 13095.783 | -1057.717 | 6729.910 | -1261.441 | 8990.449 | -1162.193 | 11 | 13389.113 |
| lgb_pair_spread_bidspace_weather_resid_5fold | loss_day_risk | nan | nan | 8679.118 | -1272.917 | 12716.167 | -1437.333 | 6795.162 | -1196.189 | 8956.054 | -1196.587 | 10 | 15890.902 |

## Feature Drift By Validation Regime

Lower distance means the validation regime is more similar to the 2026 Jan-Feb test feature distribution.

| regime | days | mean_test_like_distance | median_test_like_distance | mean_weight_raw |
| --- | --- | --- | --- | --- |
| winter_11_12_jan_feb | 120 | 0.773 | 0.729 | 0.428 |
| late_winter_12_jan_feb | 90 | 0.777 | 0.739 | 0.427 |
| standard_09_12 | 122 | 0.844 | 0.791 | 0.401 |
| jan_feb_like | 59 | 0.853 | 0.805 | 0.390 |
| all_5fold | 365 | 0.904 | 0.893 | 0.372 |

## Online Score Ledger

| submitted_at | model | config_path | output_path | online_score | notes | dashboard_model |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-05-22 | champion | configs/ensemble_champion_segmented6_prior.yaml | outputs/output_ens_champion_segmented6_prior.csv | 5482 | current best online score | ens_champion_segmented6_prior_5fold |
| 2026-05-22 | holiday_only | configs/ensemble_champion_segmented6_prior_holiday_only.yaml | outputs/output_ens_champion_segmented6_prior_holiday_only.csv | 5370 | underperformed champion despite better Jan-Feb-like validation | ens_champion_segmented6_prior_5fold_holiday_only |

## Promotion Rule

- Do not promote a candidate based only on `jan_feb_like_mean_profit`.
- Prefer candidates that preserve `all_5fold_mean_profit`, `standard_09_12_mean_profit`, and `test_like_weighted_profit`.
- Penalize extra loss days and higher p90 regret because online has only 59 days.
- If local lift is below 100 and risk metrics do not improve, skip submission.
