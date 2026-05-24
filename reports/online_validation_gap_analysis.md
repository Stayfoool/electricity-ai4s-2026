# Online vs Local Validation Gap Analysis

## Scope

- Known online scores are currently limited to champion and holiday_only, so this is a directional diagnostic, not a statistical proof.
- Goal: prevent submitting candidates that only win Jan-Feb-like but lose on broader or test-like validation.
- The analysis uses existing backtest reports and the online score ledger; no model training is performed.

## Known Online Scores

| model | online_score | risk_class | delta_vs_champion__online_score | delta_vs_champion__standard_09_12_mean_profit | delta_vs_champion__jan_feb_like_mean_profit | delta_vs_champion__all_5fold_mean_profit | delta_vs_champion__test_like_weighted_profit | delta_vs_champion__all_5fold_loss_days | delta_vs_champion__all_5fold_p90_regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold | 5482.0000 | baseline_champion | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| ens_champion_segmented6_prior_5fold_holiday_only | 5370.0000 | winter_overfit_risk | -112.0000 | -419.0560 | 217.1633 | -216.6226 | -214.1924 | 1 | 1386.1973 |
| sequence_gru_centered_5fold | 4693.0000 | broad_validation_weak | -789.0000 | -541.6875 | 6.8242 | -367.1611 | -372.8056 | 1 | 2670.2411 |
| sequence_gru_direct_price_5fold | 4686.0000 | winter_overfit_risk | -796.0000 | -875.6543 | 386.0071 | -474.2166 | -387.2962 | 5 | 1454.0621 |
| sequence_window_gru_5fold | 4477.0000 | loss_day_risk | -1005.0000 | -228.4203 | 76.0238 | -131.5518 | -85.5401 | 2 | 969.7777 |

## Metric Direction Check

| metric | interpretation | known_models | direction_match_rate | mean_abs_local_delta |
| --- | --- | --- | --- | --- |
| all_5fold_p90_regret | lower_is_better | 4 | 1.0000 | 1620.0695 |
| standard_09_12_mean_profit | higher_is_better | 4 | 1.0000 | 516.2045 |
| all_5fold_mean_profit | higher_is_better | 4 | 1.0000 | 297.3880 |
| test_like_weighted_profit | higher_is_better | 4 | 1.0000 | 264.9586 |
| all_5fold_loss_days | lower_is_better | 4 | 1.0000 | 2.2500 |
| all_5fold_worst_day_profit | higher_is_better | 4 | 0.2500 | 2394.4035 |
| winter_11_12_jan_feb_mean_profit | higher_is_better | 4 | 0.2500 | 27.0784 |
| jan_feb_like_mean_profit | higher_is_better | 4 | 0.0000 | 171.5046 |
| late_winter_12_jan_feb_mean_profit | higher_is_better | 4 | 0.0000 | 97.4673 |

## Candidate Filter: Watch Only

| model | risk_class | safety_score | standard_delta | jan_feb_delta | all_5fold_delta | test_like_delta | loss_day_delta | p90_regret_delta | hard_reject_reasons | submit_filter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold | watch | -118.5524 | -200.8743 | 60.2699 | -117.7830 | -110.5395 | -1 | 89.8321 |  | watch_only |
| ensemble_champion_segmented6_prior_blend_holiday20_5fold | watch | -135.8587 | -214.4592 | 56.5098 | -128.2418 | -100.8948 | 0 | 368.4017 |  | watch_only |
| ensemble_champion_segmented6_prior_blend_weekly20_5fold | watch | -151.8461 | -217.2546 | -3.8472 | -149.3523 | -146.4160 | -1 | 89.8321 |  | watch_only |

## Candidate Filter: Rejected By Safety Rules

| model | risk_class | safety_score | standard_delta | jan_feb_delta | all_5fold_delta | test_like_delta | loss_day_delta | p90_regret_delta | hard_reject_reasons | submit_filter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_shallow_d5_l31_leaf150_5fold | watch | -274.8087 | -254.5509 | -214.2239 | -241.7196 | -225.0725 | 0 | 781.9787 | all_5fold<-150;test_like<-150 | reject |
| ens_champion_segmented6_prior_shallow_d4_l15_leaf200_5fold | watch | -506.2729 | -233.9986 | -236.7974 | -234.8891 | -234.8273 | 1 | 427.8534 | all_5fold<-150;test_like<-150;more_loss_days | reject |
| ens_champion_segmented6_prior_5fold_cap_holiday | broad_validation_weak | -521.8829 | -539.8621 | -100.4275 | -400.0420 | -357.9046 | 0 | 2771.7281 | all_5fold<-150;test_like<-150;p90_regret+>1000 | reject |
| ens_champion_segmented6_prior_5fold_cap_only | broad_validation_weak | -705.0098 | -271.0066 | -508.6958 | -346.6350 | -322.3486 | 1 | 2315.8914 | all_5fold<-150;test_like<-150;more_loss_days;p90_regret+>1000 | reject |
| ens_champion_segmented6_prior_shallow_d3_l7_leaf200_5fold | loss_day_risk | -848.3890 | -398.9191 | -187.2337 | -331.5647 | -292.7933 | 2 | 627.1307 | all_5fold<-150;test_like<-150;more_loss_days | reject |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | loss_day_risk | -851.7014 | -499.9445 | 11.3774 | -337.2511 | -239.0505 | 2 | 1022.8937 | all_5fold<-150;test_like<-150;more_loss_days;p90_regret+>1000 | reject |
| ens_champion_segmented6_prior_weekly_delta_5fold | loss_day_risk | -984.9902 | -600.9481 | 88.6042 | -381.5451 | -341.2131 | 2 | 2413.9124 | all_5fold<-150;test_like<-150;more_loss_days;p90_regret+>1000 | reject |
| sequence_tcn_direct_price_5fold | winter_overfit_risk | -1655.6662 | -678.3002 | 310.5334 | -363.6714 | -288.7753 | 5 | 1454.0621 | all_5fold<-150;test_like<-150;more_loss_days;p90_regret+>1000;jan_feb_only_win | reject |
| ens_champion_segmented6_prior_weekly_percentile_5fold | loss_day_risk | -1830.1647 | -567.9355 | -466.2051 | -535.5668 | -467.2367 | 5 | 1379.5180 | all_5fold<-150;test_like<-150;more_loss_days;p90_regret+>1000 | reject |
| sequence_tcn_centered_5fold | loss_day_risk | -1938.7191 | -594.7200 | -569.4467 | -586.6785 | -531.8902 | 5 | 2426.6283 | all_5fold<-150;test_like<-150;more_loss_days;p90_regret+>1000 | reject |
| sequence_tftlike_centered_5fold | loss_day_risk | -2124.9755 | -977.2018 | -267.1351 | -751.2715 | -653.7941 | 5 | 3220.9746 | all_5fold<-150;test_like<-150;more_loss_days;p90_regret+>1000 | reject |
| sequence_tftlike_direct_price_5fold | loss_day_risk | -2244.6495 | -865.1923 | -87.4030 | -617.7139 | -528.1378 | 6 | 3236.4535 | all_5fold<-150;test_like<-150;more_loss_days;p90_regret+>1000 | reject |
| sequence_gru_zscore_5fold | winter_overfit_risk | -2298.9604 | -1140.4719 | 349.5136 | -666.3856 | -572.6136 | 6 | 3443.3533 | all_5fold<-150;test_like<-150;more_loss_days;p90_regret+>1000;jan_feb_only_win | reject |
| sequence_window_tcn_5fold | loss_day_risk | -2347.8795 | -933.7724 | -969.8081 | -945.2383 | -980.7913 | 5 | 2800.6769 | all_5fold<-150;test_like<-150;more_loss_days;p90_regret+>1000 | reject |
| sequence_window_tftlike_5fold | loss_day_risk | -2523.5920 | -787.6925 | -1057.0660 | -873.4022 | -800.7964 | 6 | 3487.5472 | all_5fold<-150;test_like<-150;more_loss_days;p90_regret+>1000 | reject |
| lgb_pair_spread_bidspace_weather_resid_5fold | loss_day_risk | -3042.2972 | -1196.1893 | -1437.3327 | -1272.9168 | -1196.5872 | 6 | 5899.9930 | all_5fold<-150;test_like<-150;more_loss_days;p90_regret+>1000 | reject |
| sequence_tftlike_residual_5fold | loss_day_risk | -4005.3267 | -6.4079 | 2.2655 | -3.6482 | -4.2165 | 16 | 30.3804 | more_loss_days | reject |

## Conclusion

- `holiday_only` improved Jan-Feb-like by +217/day but online dropped by -112 points.
- The same candidate lost on standard_09_12, all_5fold, and test_like_weighted; those metrics correctly warned against submission.
- Therefore Jan-Feb-like cannot be used as the primary promotion metric. It is only an auxiliary signal.
- Current unsubmitted watch candidates still have negative all_5fold/test_like deltas, so none should be promoted directly.

## Next Action

1. Do not submit another winter/holiday-only variant unless it preserves all_5fold and test_like metrics.
2. If exploring winter effects, run constrained low-weight blends and require `all_5fold_delta >= -50`, `test_like_delta >= -50`, and no extra loss days.
3. Otherwise shift to a new axis: online-distribution-aware feature selection or model-family fallback, not more global holiday weighting.

## Artifacts

- `reports/online_validation_gap_analysis.md`
- `reports/online_validation_gap_known_scores.csv`
- `reports/online_validation_gap_candidate_filter.csv`
