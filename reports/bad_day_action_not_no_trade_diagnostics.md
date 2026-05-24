# Bad-Day Action Not No-Trade

## Scope

- Anchor: current online champion-prior top1 pair.
- Candidate actions: rank2/rank3, conservative switch, consensus rules, fold-safe reranker selections, auxiliary bid_space/support selections, and prior-disabled champion where available.
- Objective: identify high-risk days but switch to another pair instead of no-trade.
- Validation: each held-out fold learns its rule only from the other folds.
- Rule inputs use prediction-stage features only; true prices/profits are used only for scoring the held-out fold.

## Overall

| regime | method | days | mean_profit | worst_profit | p10_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | profit_delta_vs_anchor | loss_day_delta_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_5fold | oracle_top10 | 176 | 10689.6177 | -9394.2116 | 2303.5048 | 2 | 737.5830 | 131 | 0 | 131 | 737.5830 | -2 |
| all_5fold | bad_day_action_not_no_trade | 176 | 9952.8981 | -15925.8974 | 1751.1154 | 4 | 0.8634 | 8 | 4 | 16 | 0.8634 | 0 |
| all_5fold | anchor_top1 | 176 | 9952.0347 | -15925.8974 | 1502.6558 | 4 | 0.0000 | 0 | 0 | 0 | 0.0000 | 0 |

## Regime Summary

| regime | method | days | mean_profit | worst_profit | p10_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard_09_12 | anchor_top1 | 120 | 7991.3509 | -15925.8974 | 947.2304 | 4 | 0.0000 | 0 | 0 | 0 |
| standard_09_12 | bad_day_action_not_no_trade | 120 | 7992.3686 | -15925.8974 | 1485.0225 | 4 | 1.0176 | 6 | 3 | 13 |
| standard_09_12 | oracle_top10 | 120 | 8709.5839 | -9394.2116 | 2130.5835 | 2 | 718.2329 | 93 | 0 | 93 |
| jan_feb_like | anchor_top1 | 56 | 14153.4998 | 8.7837 | 2395.6052 | 0 | 0.0000 | 0 | 0 | 0 |
| jan_feb_like | bad_day_action_not_no_trade | 56 | 14154.0327 | 5.7614 | 2395.6052 | 0 | 0.5328 | 2 | 1 | 3 |
| jan_feb_like | oracle_top10 | 56 | 14932.5473 | 8.7837 | 3035.0271 | 0 | 779.0475 | 38 | 0 | 38 |
| winter_11_12_jan_feb | anchor_top1 | 117 | 10170.8882 | -12550.1225 | 2248.4777 | 1 | 0.0000 | 0 | 0 | 0 |
| winter_11_12_jan_feb | bad_day_action_not_no_trade | 117 | 10172.1800 | -12359.0149 | 2248.4777 | 1 | 1.2918 | 3 | 3 | 9 |
| winter_11_12_jan_feb | oracle_top10 | 117 | 10758.6165 | -9394.2116 | 2903.6340 | 1 | 587.7283 | 81 | 0 | 81 |
| all_5fold | anchor_top1 | 176 | 9952.0347 | -15925.8974 | 1502.6558 | 4 | 0.0000 | 0 | 0 | 0 |
| all_5fold | bad_day_action_not_no_trade | 176 | 9952.8981 | -15925.8974 | 1751.1154 | 4 | 0.8634 | 8 | 4 | 16 |
| all_5fold | oracle_top10 | 176 | 10689.6177 | -9394.2116 | 2303.5048 | 2 | 737.5830 | 131 | 0 | 131 |

## Fold Summary

| fold | method | days | mean_profit | worst_profit | p10_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | anchor_top1 | 28 | 11310.4282 | 0.0000 | 2741.0967 | 0 | 0.0000 | 0 | 0 | 0 |
| valid_2025_09 | bad_day_action_not_no_trade | 28 | 11315.6086 | 0.0000 | 2741.0967 | 0 | 5.1804 | 2 | 0 | 2 |
| valid_2025_09 | oracle_top10 | 28 | 11887.8514 | 127.2545 | 4035.9109 | 0 | 577.4232 | 23 | 0 | 23 |
| valid_2025_10 | anchor_top1 | 31 | 7899.1033 | -15925.8974 | 407.0743 | 3 | 0.0000 | 0 | 0 | 0 |
| valid_2025_10 | bad_day_action_not_no_trade | 31 | 7894.4505 | -15925.8974 | 407.0743 | 3 | -4.6528 | 3 | 1 | 5 |
| valid_2025_10 | oracle_top10 | 31 | 9346.9273 | -9224.9472 | 1484.5316 | 1 | 1447.8240 | 27 | 0 | 27 |
| valid_2025_11 | anchor_top1 | 30 | 7120.7667 | 613.3252 | 2689.0321 | 0 | 0.0000 | 0 | 0 | 0 |
| valid_2025_11 | bad_day_action_not_no_trade | 30 | 7120.5608 | 613.3252 | 2689.0321 | 0 | -0.2059 | 0 | 1 | 3 |
| valid_2025_11 | oracle_top10 | 30 | 7353.7912 | 676.9524 | 3341.1785 | 0 | 233.0245 | 22 | 0 | 22 |
| valid_2025_12 | anchor_top1 | 31 | 5928.2233 | -12550.1225 | 1958.4750 | 1 | 0.0000 | 0 | 0 | 0 |
| valid_2025_12 | bad_day_action_not_no_trade | 31 | 5932.3356 | -12359.0149 | 1958.4750 | 1 | 4.1123 | 1 | 1 | 3 |
| valid_2025_12 | oracle_top10 | 31 | 6513.6048 | -9394.2116 | 3054.1074 | 1 | 585.3815 | 21 | 0 | 21 |
| valid_2025_jan_feb | anchor_top1 | 56 | 14153.4998 | 8.7837 | 2395.6052 | 0 | 0.0000 | 0 | 0 | 0 |
| valid_2025_jan_feb | bad_day_action_not_no_trade | 56 | 14154.0327 | 5.7614 | 2395.6052 | 0 | 0.5328 | 2 | 1 | 3 |
| valid_2025_jan_feb | oracle_top10 | 56 | 14932.5473 | 8.7837 | 3035.0271 | 0 | 779.0475 | 38 | 0 | 38 |

## Selected Rules

| heldout_fold | selected_from_train | action | feature | op | quantile | threshold | train_mean_lift_vs_anchor | train_loss_days | train_anchor_loss_days | train_changed_days | train_negative_lift_days | valid_mean_lift_vs_anchor | valid_loss_days | valid_anchor_loss_days | valid_changed_days | valid_negative_lift_days | train_rows | valid_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | True | aux_bidspace_rank_top3 | top10_support_count_mean | <= | 0.2000 | 1.6000 | 99.8148 | 3.0000 | 4.0000 | 20.0000 | 2.0000 | 5.1804 | 0.0000 | 0.0000 | 2.0000 | 0.0000 | 148 | 28 |
| valid_2025_10 | True | aux_margin_combo_top3 | top10_support_count_mean | <= | 0.2000 | 1.7000 | 70.8751 | 1.0000 | 1.0000 | 19.0000 | 3.0000 | -4.6528 | 3.0000 | 3.0000 | 5.0000 | 1.0000 | 145 | 31 |
| valid_2025_11 | True | aux_bidspace_rank_top3 | top10_support_count_mean | <= | 0.2000 | 1.6000 | 102.2179 | 3.0000 | 4.0000 | 19.0000 | 1.0000 | -0.2059 | 0.0000 | 0.0000 | 3.0000 | 1.0000 | 146 | 30 |
| valid_2025_12 | True | aux_bidspace_rank_top3 | top10_support_count_mean | <= | 0.2000 | 1.6800 | 102.0011 | 2.0000 | 3.0000 | 19.0000 | 1.0000 | 4.1123 | 1.0000 | 1.0000 | 3.0000 | 1.0000 | 145 | 31 |
| valid_2025_jan_feb | True | anchor_rank2 | top10_charge_start_range | <= | 0.0500 | 2.0000 | 83.8561 | 4.0000 | 4.0000 | 8.0000 | 1.0000 | 0.5328 | 0.0000 | 0.0000 | 3.0000 | 1.0000 | 120 | 56 |

## Changed-Day Audit

| date | fold | selected_action | rule_feature | rule_op | rule_threshold | feature_value | anchor_profit | selected_profit | selected_lift_vs_anchor | oracle_top10_lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-10-25 | valid_2025_10 | aux_margin_combo_top3 | top10_support_count_mean | <= | 1.7000 | 1.4000 | 9253.1150 | 7026.1617 | -2226.9533 | 1940.0072 |
| 2025-12-16 | valid_2025_12 | aux_bidspace_rank_top3 | top10_support_count_mean | <= | 1.6800 | 1.5000 | 9480.4583 | 9416.8310 | -63.6272 | 0.0000 |
| 2025-11-09 | valid_2025_11 | aux_bidspace_rank_top3 | top10_support_count_mean | <= | 1.6000 | 1.2000 | 12537.5052 | 12531.3270 | -6.1782 | 22.9313 |
| 2025-01-25 | valid_2025_jan_feb | anchor_rank2 | top10_charge_start_range | <= | 2.0000 | 2.0000 | 8.7837 | 5.7614 | -3.0223 | 0.0000 |
| 2025-10-24 | valid_2025_10 | aux_margin_combo_top3 | top10_support_count_mean | <= | 1.7000 | 1.0000 | 15286.0173 | 15286.0173 | 0.0000 | 0.0000 |
| 2025-11-19 | valid_2025_11 | aux_bidspace_rank_top3 | top10_support_count_mean | <= | 1.6000 | 1.2000 | 8716.9314 | 8716.9314 | 0.0000 | 0.0000 |
| 2025-11-24 | valid_2025_11 | aux_bidspace_rank_top3 | top10_support_count_mean | <= | 1.6000 | 1.6000 | 8438.9695 | 8438.9695 | 0.0000 | 0.0000 |
| 2025-12-08 | valid_2025_12 | aux_bidspace_rank_top3 | top10_support_count_mean | <= | 1.6800 | 1.4000 | 6468.2017 | 6468.2017 | 0.0000 | 0.0000 |
| 2025-01-13 | valid_2025_jan_feb | anchor_rank2 | top10_charge_start_range | <= | 2.0000 | 1.0000 | 7345.7104 | 7346.7571 | 1.0467 | 1.0467 |
| 2025-09-05 | valid_2025_09 | aux_bidspace_rank_top3 | top10_support_count_mean | <= | 1.6000 | 1.5000 | 10679.3736 | 10688.8318 | 9.4582 | 10.0022 |
| 2025-02-03 | valid_2025_jan_feb | anchor_rank2 | top10_charge_start_range | <= | 2.0000 | 2.0000 | 8939.6268 | 8971.4404 | 31.8136 | 63.6272 |
| 2025-10-11 | valid_2025_10 | aux_margin_combo_top3 | top10_support_count_mean | <= | 1.7000 | 1.6000 | 20681.1998 | 20812.1383 | 130.9385 | 2758.7434 |
| 2025-09-11 | valid_2025_09 | aux_bidspace_rank_top3 | top10_support_count_mean | <= | 1.6000 | 1.2000 | 7755.0753 | 7890.6681 | 135.5928 | 616.3252 |
| 2025-12-12 | valid_2025_12 | aux_bidspace_rank_top3 | top10_support_count_mean | <= | 1.6800 | 1.4000 | -12550.1225 | -12359.0149 | 191.1076 | 3155.9109 |
| 2025-10-22 | valid_2025_10 | aux_margin_combo_top3 | top10_support_count_mean | <= | 1.7000 | 1.5000 | 680.0034 | 1543.7558 | 863.7525 | 1994.9716 |
| 2025-10-23 | valid_2025_10 | aux_margin_combo_top3 | top10_support_count_mean | <= | 1.7000 | 1.6000 | 1464.8612 | 2552.8870 | 1088.0257 | 2176.4237 |

## Action Pool If Always Used

| action | days_available | mean_profit_if_always_used | mean_lift_vs_anchor | loss_days | changed_days | positive_lift_days | negative_lift_days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conservative_switch | 176 | 9977.6876 | 25.6529 | 4 | 16 | 9 | 4 |
| aux_margin_combo_top3 | 176 | 9955.4340 | 3.3993 | 4 | 98 | 35 | 27 |
| reranker_gbr_lift | 176 | 9947.3503 | -4.6844 | 4 | 12 | 2 | 6 |
| aux_bidspace_rank_top3 | 176 | 9920.0459 | -31.9888 | 4 | 120 | 47 | 31 |
| anchor_rank3 | 176 | 9917.3728 | -34.6619 | 5 | 176 | 56 | 60 |
| aux_support_score_top3 | 176 | 9908.5370 | -43.4977 | 4 | 61 | 18 | 22 |
| consensus_support2_then_anchor_rank | 176 | 9876.1295 | -75.9052 | 4 | 176 | 6 | 12 |
| anchor_rank2 | 176 | 9874.6142 | -77.4204 | 6 | 176 | 55 | 61 |
| consensus_anchor_rank_le5_support2 | 176 | 9855.9466 | -96.0880 | 4 | 176 | 13 | 17 |
| consensus_unanimous_then_anchor_rank | 176 | 9841.8313 | -110.2034 | 4 | 176 | 17 | 23 |
| reranker_gbr_true_profit | 176 | 9840.3288 | -111.7059 | 4 | 41 | 10 | 22 |
| prior_disabled_champion | 120 | 7850.8142 | -140.5368 | 5 | 101 | 49 | 59 |
| consensus_support_score | 176 | 9808.2304 | -143.8043 | 3 | 176 | 29 | 45 |

## Top Passing Train Rules

| action | feature | op | quantile | threshold | days | mean_profit | anchor_mean_profit | mean_lift_vs_anchor | total_lift_vs_anchor | worst_profit | anchor_worst_profit | p10_profit | anchor_p10_profit | loss_days | anchor_loss_days | triggered_days | changed_days | positive_lift_days | negative_lift_days | changed_rate | passes_constraints | heldout_fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aux_bidspace_rank_top3 | top10_support_count_mean | <= | 0.2000 | 1.6000 | 146.0000 | 10636.0200 | 10533.8021 | 102.2179 | 14923.8129 | -15925.8974 | -15925.8974 | 1209.6349 | 1164.6823 | 3.0000 | 4.0000 | 31.0000 | 19.0000 | 14.0000 | 1.0000 | 0.1301 | True | valid_2025_11 |
| aux_bidspace_rank_top3 | top10_support_count_mean | <= | 0.2000 | 1.6800 | 145.0000 | 10914.2989 | 10812.2978 | 102.0011 | 14790.1544 | -15925.8974 | -15925.8974 | 1495.0969 | 1495.0969 | 2.0000 | 3.0000 | 29.0000 | 19.0000 | 13.0000 | 1.0000 | 0.1310 | True | valid_2025_12 |
| aux_bidspace_rank_top3 | top10_support_count_mean | <= | 0.2000 | 1.6000 | 148.0000 | 9794.8561 | 9695.0413 | 99.8148 | 14772.5837 | -15925.8974 | -15925.8974 | 1312.3296 | 1312.3296 | 3.0000 | 4.0000 | 31.0000 | 20.0000 | 12.0000 | 2.0000 | 0.1351 | True | valid_2025_09 |
| prior_disabled_champion | top10_charge_start_std | <= | 0.2000 | 1.1005 | 146.0000 | 10622.8501 | 10533.8021 | 89.0480 | 13001.0074 | -15925.8974 | -15925.8974 | 1202.4768 | 1164.6823 | 4.0000 | 4.0000 | 18.0000 | 14.0000 | 12.0000 | 4.0000 | 0.0959 | True | valid_2025_11 |
| prior_disabled_champion | top10_charge_start_range | <= | 0.1000 | 3.0000 | 148.0000 | 9782.0564 | 9695.0413 | 87.0150 | 12878.2259 | -15925.8974 | -15925.8974 | 1365.2420 | 1312.3296 | 4.0000 | 4.0000 | 26.0000 | 21.0000 | 16.0000 | 8.0000 | 0.1419 | True | valid_2025_09 |
| prior_disabled_champion | top10_charge_start_range | <= | 0.2000 | 3.0000 | 148.0000 | 9782.0564 | 9695.0413 | 87.0150 | 12878.2259 | -15925.8974 | -15925.8974 | 1365.2420 | 1312.3296 | 4.0000 | 4.0000 | 26.0000 | 21.0000 | 16.0000 | 8.0000 | 0.1419 | True | valid_2025_09 |
| prior_disabled_champion | top10_charge_start_range | <= | 0.1000 | 3.0000 | 146.0000 | 10619.1218 | 10533.8021 | 85.3198 | 12456.6859 | -15925.8974 | -15925.8974 | 1202.4768 | 1164.6823 | 4.0000 | 4.0000 | 24.0000 | 18.0000 | 14.0000 | 7.0000 | 0.1233 | True | valid_2025_11 |
| prior_disabled_champion | top10_charge_start_range | <= | 0.2000 | 3.0000 | 146.0000 | 10619.1218 | 10533.8021 | 85.3198 | 12456.6859 | -15925.8974 | -15925.8974 | 1202.4768 | 1164.6823 | 4.0000 | 4.0000 | 24.0000 | 18.0000 | 14.0000 | 7.0000 | 0.1233 | True | valid_2025_11 |
| prior_disabled_champion | top10_charge_start_std | <= | 0.2000 | 1.0770 | 148.0000 | 9780.0833 | 9695.0413 | 85.0420 | 12586.2087 | -15925.8974 | -15925.8974 | 1365.2420 | 1312.3296 | 4.0000 | 4.0000 | 19.0000 | 15.0000 | 12.0000 | 5.0000 | 0.1014 | True | valid_2025_09 |
| anchor_rank2 | top10_charge_start_range | <= | 0.0500 | 2.0000 | 120.0000 | 8075.2071 | 7991.3509 | 83.8561 | 10062.7334 | -15925.8974 | -15925.8974 | 947.2304 | 947.2304 | 4.0000 | 4.0000 | 8.0000 | 8.0000 | 6.0000 | 1.0000 | 0.0667 | True | valid_2025_jan_feb |
| anchor_rank3 | top10_predicted_spread_range | > | 0.9000 | 0.6326 | 148.0000 | 9776.9032 | 9695.0413 | 81.8618 | 12115.5517 | -15925.8974 | -15925.8974 | 1475.5484 | 1312.3296 | 3.0000 | 4.0000 | 15.0000 | 15.0000 | 10.0000 | 3.0000 | 0.1014 | True | valid_2025_09 |
| prior_disabled_champion | top10_charge_start_std | <= | 0.1000 | 0.9803 | 146.0000 | 10609.2085 | 10533.8021 | 75.4065 | 11009.3444 | -15925.8974 | -15925.8974 | 1202.4768 | 1164.6823 | 4.0000 | 4.0000 | 8.0000 | 7.0000 | 7.0000 | 1.0000 | 0.0479 | True | valid_2025_11 |
| prior_disabled_champion | top10_charge_start_std | <= | 0.0500 | 0.8232 | 146.0000 | 10608.0537 | 10533.8021 | 74.2516 | 10840.7322 | -15925.8974 | -15925.8974 | 1202.4768 | 1164.6823 | 4.0000 | 4.0000 | 6.0000 | 6.0000 | 6.0000 | 0.0000 | 0.0411 | True | valid_2025_11 |
| prior_disabled_champion | top10_charge_start_range | <= | 0.0500 | 2.0000 | 146.0000 | 10608.0537 | 10533.8021 | 74.2516 | 10840.7322 | -15925.8974 | -15925.8974 | 1202.4768 | 1164.6823 | 4.0000 | 4.0000 | 6.0000 | 6.0000 | 6.0000 | 0.0000 | 0.0411 | True | valid_2025_11 |
| consensus_support_score | top10_charge_start_std | <= | 0.1000 | 1.0290 | 120.0000 | 8064.3657 | 7991.3509 | 73.0148 | 8761.7759 | -15925.8974 | -15925.8974 | 947.2304 | 947.2304 | 4.0000 | 4.0000 | 12.0000 | 12.0000 | 5.0000 | 3.0000 | 0.1000 | True | valid_2025_jan_feb |
| prior_disabled_champion | top10_charge_start_std | <= | 0.1000 | 0.9398 | 148.0000 | 9767.5919 | 9695.0413 | 72.5506 | 10737.4875 | -15925.8974 | -15925.8974 | 1365.2420 | 1312.3296 | 4.0000 | 4.0000 | 10.0000 | 9.0000 | 7.0000 | 3.0000 | 0.0608 | True | valid_2025_09 |
| prior_disabled_champion | top10_charge_start_range | <= | 0.0500 | 2.0000 | 148.0000 | 9767.1620 | 9695.0413 | 72.1207 | 10673.8603 | -15925.8974 | -15925.8974 | 1365.2420 | 1312.3296 | 4.0000 | 4.0000 | 8.0000 | 8.0000 | 6.0000 | 2.0000 | 0.0541 | True | valid_2025_09 |
| aux_margin_combo_top3 | top10_support_count_mean | <= | 0.2000 | 1.7000 | 145.0000 | 10461.8124 | 10390.9373 | 70.8751 | 10276.8899 | -12359.0149 | -12550.1225 | 2219.7990 | 2219.7990 | 1.0000 | 1.0000 | 35.0000 | 19.0000 | 10.0000 | 3.0000 | 0.1310 | True | valid_2025_10 |
| anchor_rank2 | top10_charge_start_std | <= | 0.0500 | 0.8232 | 146.0000 | 10604.6538 | 10533.8021 | 70.8517 | 10344.3475 | -15925.8974 | -15925.8974 | 1202.4768 | 1164.6823 | 4.0000 | 4.0000 | 8.0000 | 8.0000 | 8.0000 | 0.0000 | 0.0548 | True | valid_2025_11 |
| anchor_rank2 | top10_charge_start_range | <= | 0.0500 | 2.0000 | 146.0000 | 10604.6331 | 10533.8021 | 70.8310 | 10341.3252 | -15925.8974 | -15925.8974 | 1202.4768 | 1164.6823 | 4.0000 | 4.0000 | 9.0000 | 9.0000 | 8.0000 | 1.0000 | 0.0616 | True | valid_2025_11 |
| aux_margin_combo_top3 | top10_support_count_mean | <= | 0.2000 | 1.6000 | 146.0000 | 10603.6601 | 10533.8021 | 69.8581 | 10199.2774 | -15925.8974 | -15925.8974 | 1542.1031 | 1164.6823 | 4.0000 | 4.0000 | 31.0000 | 17.0000 | 11.0000 | 2.0000 | 0.1164 | True | valid_2025_11 |
| aux_margin_combo_top3 | top10_support_count_mean | <= | 0.2000 | 1.6800 | 145.0000 | 10881.7159 | 10812.2978 | 69.4181 | 10065.6189 | -15925.8974 | -15925.8974 | 1710.1411 | 1495.0969 | 3.0000 | 3.0000 | 29.0000 | 17.0000 | 10.0000 | 2.0000 | 0.1172 | True | valid_2025_12 |
| aux_margin_combo_top3 | top10_support_count_mean | <= | 0.2000 | 1.6000 | 148.0000 | 9763.8497 | 9695.0413 | 68.8084 | 10183.6410 | -15925.8974 | -15925.8974 | 1542.7642 | 1312.3296 | 4.0000 | 4.0000 | 31.0000 | 17.0000 | 10.0000 | 3.0000 | 0.1149 | True | valid_2025_09 |
| prior_disabled_champion | top10_charge_start_range | <= | 0.1000 | 3.0000 | 145.0000 | 10880.6061 | 10812.2978 | 68.3083 | 9904.7056 | -15925.8974 | -15925.8974 | 1708.1578 | 1495.0969 | 3.0000 | 3.0000 | 22.0000 | 19.0000 | 13.0000 | 7.0000 | 0.1310 | True | valid_2025_12 |
| prior_disabled_champion | top10_charge_start_range | <= | 0.2000 | 3.0000 | 145.0000 | 10880.6061 | 10812.2978 | 68.3083 | 9904.7056 | -15925.8974 | -15925.8974 | 1708.1578 | 1495.0969 | 3.0000 | 3.0000 | 22.0000 | 19.0000 | 13.0000 | 7.0000 | 0.1310 | True | valid_2025_12 |
| anchor_rank2 | top10_charge_start_range | <= | 0.0500 | 2.0000 | 148.0000 | 9763.2344 | 9695.0413 | 68.1930 | 10092.5714 | -15925.8974 | -15925.8974 | 1365.2420 | 1312.3296 | 4.0000 | 4.0000 | 11.0000 | 11.0000 | 8.0000 | 2.0000 | 0.0743 | True | valid_2025_09 |
| aux_margin_combo_top3 | top10_charge_start_std | <= | 0.2000 | 1.1353 | 145.0000 | 10457.7565 | 10390.9373 | 66.8193 | 9688.7929 | -12359.0149 | -12550.1225 | 2219.7990 | 2219.7990 | 1.0000 | 1.0000 | 30.0000 | 15.0000 | 7.0000 | 4.0000 | 0.1034 | True | valid_2025_10 |
| aux_bidspace_rank_top3 | top10_charge_start_std | <= | 0.2000 | 1.1353 | 145.0000 | 10457.4362 | 10390.9373 | 66.4989 | 9642.3419 | -12359.0149 | -12550.1225 | 2219.7990 | 2219.7990 | 1.0000 | 1.0000 | 30.0000 | 19.0000 | 11.0000 | 5.0000 | 0.1310 | True | valid_2025_10 |
| anchor_rank2 | top10_charge_start_std | <= | 0.0500 | 0.7888 | 120.0000 | 8057.4676 | 7991.3509 | 66.1167 | 7933.9983 | -15925.8974 | -15925.8974 | 947.2304 | 947.2304 | 4.0000 | 4.0000 | 7.0000 | 7.0000 | 5.0000 | 2.0000 | 0.0583 | True | valid_2025_jan_feb |
| prior_disabled_champion | top10_charge_start_std | <= | 0.1000 | 0.9308 | 145.0000 | 10878.2647 | 10812.2978 | 65.9669 | 9565.2034 | -15925.8974 | -15925.8974 | 1708.1578 | 1495.0969 | 3.0000 | 3.0000 | 10.0000 | 9.0000 | 7.0000 | 3.0000 | 0.0621 | True | valid_2025_12 |

## Risk Features Used

`anchor_predicted_spread`, `top1_top2_gap`, `top1_top5_mean_gap`, `top5_spread_std`, `support_count`, `rank_score`, `mean_rank`, `support_present_cols_sum`, `lgb_segmented_6_last_180d_rank`, `lgb_segmented_6_last_180d_predicted_spread`, `lgb_segmented_6_margin_core_last_180d_rank`, `lgb_segmented_6_margin_core_last_180d_predicted_spread`, `charge_start`, `discharge_start`, `gap_slots`, `charge_hour`, `discharge_hour`, `month`, `dayofweek`, `is_weekend`, `diff_bid_space_mean`, `diff_bid_space_rank_pct`, `day_bid_space_std`, `day_bid_space_range`, `diff_net_load_mean`, `diff_net_load_rank_pct`, `day_net_load_std`, `day_net_load_range`, `diff_renewable_ratio_mean`, `diff_renewable_ratio_rank_pct`, `day_renewable_ratio_std`, `day_renewable_ratio_range`, `diff_wind_ratio_rank_pct`, `diff_solar_ratio_rank_pct`, `diff_tie_line_ratio_rank_pct`, `diff_non_market_ratio_rank_pct`, `diff_nwp_wind_speed100_mean_mh_resid_mean`, `diff_nwp_wind_speed100_mean_mh_resid_rank_pct`, `day_nwp_wind_speed100_mean_mh_resid_std`, `diff_nwp_ghi_mean_mh_resid_mean`, `diff_nwp_ghi_mean_mh_resid_rank_pct`, `day_nwp_ghi_mean_mh_resid_std`, `diff_nwp_tcc_mean_mh_resid_mean`, `diff_nwp_tcc_mean_mh_resid_rank_pct`, `day_nwp_tcc_mean_mh_resid_std`, `diff_nwp_sp_mean_mh_resid_mean`, `diff_nwp_sp_mean_mh_resid_rank_pct`, `day_nwp_sp_mean_mh_resid_std`, `top10_predicted_spread_std`, `top10_predicted_spread_range`, `top10_charge_start_std`, `top10_discharge_start_std`, `top10_charge_start_range`, `top10_discharge_start_range`, `top10_support_count_max`, `top10_support_count_mean`, `top10_mean_rank_min`, `top10_bidspace_score_max_delta`, `top10_margin_score_max_delta`, `top10_support_score_max_delta`, `rank2_spread_delta_from_top1`, `rank2_bidspace_rank_score_delta`, `rank2_margin_score_delta`, `rank2_support_score_delta`

## Decision

- The result is directionally positive but below the +100/day promotion threshold.
- Keep it as diagnostic evidence; do not generate a submit from this rule yet.

## Artifacts

- `reports/bad_day_action_not_no_trade_selected_daily.csv`
- `reports/bad_day_action_not_no_trade_rules.csv`
- `reports/bad_day_action_not_no_trade_rule_search.csv`
- `reports/bad_day_action_not_no_trade_overall.csv`
- `reports/bad_day_action_not_no_trade_by_regime.csv`
- `reports/bad_day_action_not_no_trade_by_fold.csv`
- `reports/bad_day_action_not_no_trade_action_pool.csv`
