# Experiment Map

This map summarizes existing configs, backtest reports, online scores, and manual notes.
It is intended to expose matrix-style experiment coverage and avoid linear local search.

## Baseline

- Baseline model for local comparison: `ens_champion_segmented6_prior_5fold`.
- Baseline avg_profit_mean: `9282.404257945691`.
- Baseline Jan-Feb-like avg_profit: `14153.49983443873`.

## Decision Buckets

| decision_bucket | count |
| --- | --- |
| global_failed | 41 |
| rejected | 33 |
| candidate | 8 |
| promoted | 5 |
| near_champion | 2 |
| online_tested | 2 |

## Top Measured Experiments

| model | decision_bucket | model_family | feature_flags | avg_profit_mean | jan_feb_like_avg_profit | daily_loss_days | online_score | opportunity_tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior | online_tested | ensemble | base | 8064.630 |  | 4.000 | 5482.000 |  |
| ens_champion_segmented6_prior_holiday_only | online_tested | ensemble | holiday | 7627.724 |  | 5.000 | 5370.000 |  |
| ens_champion_segmented6_prior_5fold | near_champion | ensemble | base | 9282.404 | 14153.500 | 4.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_5fold | near_champion | ensemble | base | 9282.404 | 14153.500 | 4.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_5fold | rejected | ensemble | base | 9282.404 | 14153.500 | 4.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_5fold | rejected | ensemble | base | 9282.404 | 14153.500 | 4.000 |  | has_jan_feb_evidence |
| ensemble_champion_segmented6_prior_blend_holiday05_5fold | rejected | ensemble | holiday | 9233.784 | 14150.366 | 4.000 |  | has_jan_feb_evidence |
| ensemble_champion_segmented6_prior_blend_weekly05_5fold | rejected | ensemble | weekly_relative | 9186.280 | 14161.728 | 4.000 |  | has_jan_feb_evidence/temporal_relative_candidate |
| ensemble_champion_segmented6_prior_blend_holiday10_5fold | rejected | ensemble | holiday | 9177.351 | 14149.230 | 4.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_clean_features_5fold | global_failed | ensemble | bias_correction | 9132.478 | 14438.687 |  |  | has_jan_feb_evidence |
| ensemble_champion_segmented6_prior_blend_holiday05_weekly05_5fold | rejected | ensemble | holiday/weekly_relative | 9131.555 | 14161.728 | 4.000 |  | has_jan_feb_evidence/temporal_relative_candidate |
| ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold | global_failed | ensemble | holiday/weekly_relative | 9130.920 | 14213.770 | 3.000 |  | has_jan_feb_evidence/temporal_relative_candidate |
| ensemble_champion_segmented6_prior_blend_holiday20_5fold | global_failed | ensemble | holiday | 9117.507 | 14210.010 | 4.000 |  | has_jan_feb_evidence |
| ensemble_champion_segmented6_prior_blend_weekly10_5fold | rejected | ensemble | weekly_relative | 9109.885 | 14162.333 | 4.000 |  | has_jan_feb_evidence/temporal_relative_candidate |
| ensemble_champion_segmented6_prior_blend_weekly20_5fold | global_failed | ensemble | weekly_relative | 9102.952 | 14149.653 | 3.000 |  | has_jan_feb_evidence/temporal_relative_candidate |
| sequence_window_gru_5fold | rejected | lightgbm | base | 9098.298 | 14229.524 | 6.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_shallow_d4_l15_leaf200_5fold | global_failed | ensemble | base | 9039.647 | 13916.702 | 5.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_shallow_d5_l31_leaf150_5fold | rejected | ensemble | base | 9027.991 | 13939.276 | 4.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_replace_features_5fold | global_failed | ensemble | bias_correction | 8991.226 | 14347.536 |  |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_clean_features_shrink05_5fold | global_failed | ensemble | bias_correction | 8988.055 | 14314.424 |  |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_feaug_ws025_w05_5fold | global_failed | ensemble | base | 8976.858 | 13812.120 | 4.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_5fold_holiday_only | global_failed | ensemble | holiday | 8976.312 | 14370.663 | 5.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_5fold_cap_only | global_failed | ensemble | capacity | 8958.206 | 13644.804 | 5.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_feaug_ws_late050_w05_5fold | global_failed | ensemble | base | 8931.025 | 13750.738 | 5.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_shallow_d3_l7_leaf200_5fold | global_failed | ensemble | base | 8913.871 | 13966.266 | 6.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | global_failed | ensemble | weekly_relative | 8877.436 | 14164.877 | 6.000 |  | has_jan_feb_evidence/temporal_relative_candidate |
| ens_champion_segmented6_prior_forecast_error_aug_5fold | global_failed | ensemble | base | 8868.187 | 13788.616 | 3.000 |  | has_jan_feb_evidence |
| sequence_gru_centered_5fold | rejected | lightgbm | base | 8834.369 | 14160.324 | 5.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_weekly_delta_5fold | global_failed | ensemble | weekly_relative | 8810.781 | 14242.104 | 6.000 |  | has_jan_feb_evidence/temporal_relative_candidate |
| ens_champion_segmented6_prior_5fold_cap_holiday | global_failed | ensemble | capacity/holiday | 8810.118 | 14053.072 | 4.000 |  | has_jan_feb_evidence |

## Reusable Opportunity Tags

| model | decision_bucket | model_family | feature_flags | avg_profit_mean | jan_feb_like_avg_profit | daily_loss_days | online_score | opportunity_tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cat_baseline_last_180d | candidate | catboost | base | 7556.908 |  | 7.000 |  | model_family_diversity |
| cat_baseline | candidate | catboost | base | 7509.300 |  | 7.000 |  | model_family_diversity |
| ens_champion_segmented6_prior_clean_features_5fold | global_failed | ensemble | bias_correction | 9132.478 | 14438.687 |  |  | has_jan_feb_evidence |
| ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold | global_failed | ensemble | holiday/weekly_relative | 9130.920 | 14213.770 | 3.000 |  | has_jan_feb_evidence/temporal_relative_candidate |
| ensemble_champion_segmented6_prior_blend_holiday20_5fold | global_failed | ensemble | holiday | 9117.507 | 14210.010 | 4.000 |  | has_jan_feb_evidence |
| ensemble_champion_segmented6_prior_blend_weekly20_5fold | global_failed | ensemble | weekly_relative | 9102.952 | 14149.653 | 3.000 |  | has_jan_feb_evidence/temporal_relative_candidate |
| ens_champion_segmented6_prior_shallow_d4_l15_leaf200_5fold | global_failed | ensemble | base | 9039.647 | 13916.702 | 5.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_replace_features_5fold | global_failed | ensemble | bias_correction | 8991.226 | 14347.536 |  |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_clean_features_shrink05_5fold | global_failed | ensemble | bias_correction | 8988.055 | 14314.424 |  |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_feaug_ws025_w05_5fold | global_failed | ensemble | base | 8976.858 | 13812.120 | 4.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_5fold_holiday_only | global_failed | ensemble | holiday | 8976.312 | 14370.663 | 5.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_5fold_cap_only | global_failed | ensemble | capacity | 8958.206 | 13644.804 | 5.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_feaug_ws_late050_w05_5fold | global_failed | ensemble | base | 8931.025 | 13750.738 | 5.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_shallow_d3_l7_leaf200_5fold | global_failed | ensemble | base | 8913.871 | 13966.266 | 6.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | global_failed | ensemble | weekly_relative | 8877.436 | 14164.877 | 6.000 |  | has_jan_feb_evidence/temporal_relative_candidate |
| ens_champion_segmented6_prior_forecast_error_aug_5fold | global_failed | ensemble | base | 8868.187 | 13788.616 | 3.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_weekly_delta_5fold | global_failed | ensemble | weekly_relative | 8810.781 | 14242.104 | 6.000 |  | has_jan_feb_evidence/temporal_relative_candidate |
| ens_champion_segmented6_prior_5fold_cap_holiday | global_failed | ensemble | capacity/holiday | 8810.118 | 14053.072 | 4.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_weekly_percentile_5fold | global_failed | ensemble | weekly_relative | 8727.476 | 13687.295 | 9.000 |  | has_jan_feb_evidence/temporal_relative_candidate |
| ens_champion_segmented6_prior_weather_correction | global_failed | ensemble | weather_correction | 7676.675 |  | 4.000 |  | weather_as_global_failed_try_as_risk_or_rerank |
| ens_champion_segmented6_prior_bidspace | global_failed | ensemble | bid_space | 7457.757 |  | 10.000 |  | bidspace_as_global_failed_try_as_aux_signal |
| ens_champion_segmented6_prior_lag | global_failed | ensemble | lag | 7327.843 |  | 10.000 |  | temporal_relative_candidate |
| ens_champion_segmented6_prior_lag_winter_weight | global_failed | ensemble | lag | 7283.063 |  | 6.000 |  | temporal_relative_candidate |
| lgb_bid_space_only_last_180d | global_failed | segmented | bid_space | 7201.132 |  | 12.000 |  | bidspace_as_global_failed_try_as_aux_signal |
| linear_l2_bidspace | global_failed | linear | bid_space | 7162.123 |  | 10.000 |  | bidspace_as_global_failed_try_as_aux_signal |
| linear_l1_bidspace | global_failed | linear | bid_space | 6821.713 |  | 10.000 |  | bidspace_as_global_failed_try_as_aux_signal |
| ens_champion_segmented6_prior_5fold | near_champion | ensemble | base | 9282.404 | 14153.500 | 4.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_5fold | near_champion | ensemble | base | 9282.404 | 14153.500 | 4.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_5fold | rejected | ensemble | base | 9282.404 | 14153.500 | 4.000 |  | has_jan_feb_evidence |
| ens_champion_segmented6_prior_5fold | rejected | ensemble | base | 9282.404 | 14153.500 | 4.000 |  | has_jan_feb_evidence |

## Unmeasured Configs

| model | model_family | feature_flags | target_family | config_path |
| --- | --- | --- | --- | --- |

## Artifacts

- `reports/experiment_map.csv`
- `reports/experiment_map.md`
