# Model-Family Consensus Fallback

## Scope

- Anchor: current online champion `ens_champion_segmented6_prior_5fold`.
- Expert families: CatBoost 180d, XGBoost 180d, segmented_6, segmented_6 margin-core.
- This is a diagnostic-only experiment: no retraining and no submit file generation.
- Coverage is limited to the common 09-12 validation folds because Cat/XGB old daily reports do not include `valid_2025_jan_feb`.

## Plan Executed

- Build one candidate per day from each expert's top1 legal charge/discharge pair.
- Keep champion top1 by default.
- Search conservative fallback rules using only non-held-out folds: low champion confidence gate plus near-pair support among model families.
- Apply the selected rule to the held-out fold only.
- Reject any rule that changes more than 10% of training days, increases loss days, or has more negative than positive changed days.

## Data Coverage

| fold | days | start | end |
| --- | --- | --- | --- |
| valid_2025_09 | 28 | 2025-09-01 00:00:00 | 2025-09-28 00:00:00 |
| valid_2025_10 | 31 | 2025-10-01 00:00:00 | 2025-10-31 00:00:00 |
| valid_2025_11 | 30 | 2025-11-01 00:00:00 | 2025-11-30 00:00:00 |
| valid_2025_12 | 31 | 2025-12-01 00:00:00 | 2025-12-31 00:00:00 |

## Expert Top1 Standalone Comparison

| candidate_model | days | mean_profit | anchor_mean_profit | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | same_pair_rate | support_near2_ge3_days | support_near2_ge4_days | support_near2_ge3_mean_lift | support_near2_ge4_mean_lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seg6 | 120 | 7739.550 | 7991.351 | -251.801 | 44 | 59 | 0.108 | 62 | 27 | 106.443 | 24.259 |
| margin6 | 120 | 7725.897 | 7991.351 | -265.454 | 51 | 52 | 0.058 | 58 | 26 | 210.019 | 144.151 |
| cat180 | 120 | 7500.769 | 7991.351 | -490.582 | 56 | 50 | 0.067 | 40 | 23 | -230.543 | -259.244 |
| xgb180 | 120 | 7371.792 | 7991.351 | -619.559 | 45 | 65 | 0.033 | 36 | 22 | -25.571 | -110.682 |

## Overall

| method | days | mean_profit | anchor_mean_profit | mean_lift_vs_anchor | total_lift_vs_anchor | worst_profit | anchor_worst_profit | p10_profit | loss_days | anchor_loss_days | changed_days | changed_rate | positive_lift_days | negative_lift_days | anchor_group_mean_profit | anchor_group_loss_days | profit_delta_vs_anchor | loss_day_delta_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_family_top1 | 120.000 | 8973.753 | 7991.351 | 982.402 | 117888.296 | -6342.633 | -15925.897 | 2438.345 | 3.000 | 4.000 | 90.000 | 0.750 | 90.000 | 0.000 | 7991.351 | 4.000 | 982.402 | -1.000 |
| anchor_top1 | 120.000 | 7991.351 | 7991.351 | 0.000 | 0.000 | -15925.897 | -15925.897 | 947.230 | 4.000 | 4.000 | 0.000 | 0.000 | 0.000 | 0.000 | 7991.351 | 4.000 | 0.000 | 0.000 |
| model_family_consensus_fallback | 120.000 | 7934.579 | 7991.351 | -56.772 | -6812.629 | -15925.897 | -15925.897 | 947.230 | 4.000 | 4.000 | 8.000 | 0.067 | 3.000 | 5.000 | 7991.351 | 4.000 | -56.772 | 0.000 |

## Fold Summary

| fold | method | days | mean_profit | anchor_mean_profit | mean_lift_vs_anchor | total_lift_vs_anchor | worst_profit | anchor_worst_profit | p10_profit | loss_days | anchor_loss_days | changed_days | changed_rate | positive_lift_days | negative_lift_days | anchor_group_mean_profit | anchor_group_loss_days | profit_delta_vs_anchor | loss_day_delta_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | oracle_family_top1 | 28.000 | 12060.258 | 11310.428 | 749.830 | 20995.229 | 200.521 | -0.000 | 4109.017 | 0.000 | 0.000 | 20.000 | 0.714 | 20.000 | 0.000 | 11310.428 | 0.000 | 749.830 | 0.000 |
| valid_2025_09 | anchor_top1 | 28.000 | 11310.428 | 11310.428 | 0.000 | 0.000 | -0.000 | -0.000 | 2741.097 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 11310.428 | 0.000 | 0.000 | 0.000 |
| valid_2025_09 | model_family_consensus_fallback | 28.000 | 11268.048 | 11310.428 | -42.380 | -1186.648 | -0.000 | -0.000 | 2741.097 | 0.000 | 0.000 | 1.000 | 0.036 | 0.000 | 1.000 | 11310.428 | 0.000 | -42.380 | 0.000 |
| valid_2025_10 | oracle_family_top1 | 31.000 | 9539.561 | 7899.103 | 1640.457 | 50854.180 | -1520.691 | -15925.897 | 608.416 | 2.000 | 3.000 | 25.000 | 0.806 | 25.000 | 0.000 | 7899.103 | 3.000 | 1640.457 | -1.000 |
| valid_2025_10 | anchor_top1 | 31.000 | 7899.103 | 7899.103 | 0.000 | 0.000 | -15925.897 | -15925.897 | 407.074 | 3.000 | 3.000 | 0.000 | 0.000 | 0.000 | 0.000 | 7899.103 | 3.000 | 0.000 | 0.000 |
| valid_2025_10 | model_family_consensus_fallback | 31.000 | 7717.620 | 7899.103 | -181.483 | -5625.981 | -15925.897 | -15925.897 | 407.074 | 3.000 | 3.000 | 7.000 | 0.226 | 3.000 | 4.000 | 7899.103 | 3.000 | -181.483 | 0.000 |
| valid_2025_11 | oracle_family_top1 | 30.000 | 7692.289 | 7120.767 | 571.522 | 17145.660 | 763.527 | 613.325 | 3608.713 | 0.000 | 0.000 | 23.000 | 0.767 | 23.000 | 0.000 | 7120.767 | 0.000 | 571.522 | 0.000 |
| valid_2025_11 | anchor_top1 | 30.000 | 7120.767 | 7120.767 | 0.000 | 0.000 | 613.325 | 613.325 | 2689.032 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 7120.767 | 0.000 | 0.000 | 0.000 |
| valid_2025_11 | model_family_consensus_fallback | 30.000 | 7120.767 | 7120.767 | 0.000 | 0.000 | 613.325 | 613.325 | 2689.032 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 7120.767 | 0.000 | 0.000 | 0.000 |
| valid_2025_12 | oracle_family_top1 | 31.000 | 6860.263 | 5928.223 | 932.040 | 28893.227 | -6342.633 | -12550.122 | 2382.341 | 1.000 | 1.000 | 22.000 | 0.710 | 22.000 | 0.000 | 5928.223 | 1.000 | 932.040 | 0.000 |
| valid_2025_12 | anchor_top1 | 31.000 | 5928.223 | 5928.223 | 0.000 | 0.000 | -12550.122 | -12550.122 | 1958.475 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 5928.223 | 1.000 | 0.000 | 0.000 |
| valid_2025_12 | model_family_consensus_fallback | 31.000 | 5928.223 | 5928.223 | 0.000 | 0.000 | -12550.122 | -12550.122 | 1958.475 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 5928.223 | 1.000 | 0.000 | 0.000 |

## Month Summary

| month | method | days | mean_profit | anchor_mean_profit | mean_lift_vs_anchor | total_lift_vs_anchor | worst_profit | anchor_worst_profit | p10_profit | loss_days | anchor_loss_days | changed_days | changed_rate | positive_lift_days | negative_lift_days | anchor_group_mean_profit | anchor_group_loss_days | profit_delta_vs_anchor | loss_day_delta_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | oracle_family_top1 | 28.000 | 12060.258 | 11310.428 | 749.830 | 20995.229 | 200.521 | -0.000 | 4109.017 | 0.000 | 0.000 | 20.000 | 0.714 | 20.000 | 0.000 | 11310.428 | 0.000 | 749.830 | 0.000 |
| 9 | anchor_top1 | 28.000 | 11310.428 | 11310.428 | 0.000 | 0.000 | -0.000 | -0.000 | 2741.097 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 11310.428 | 0.000 | 0.000 | 0.000 |
| 9 | model_family_consensus_fallback | 28.000 | 11268.048 | 11310.428 | -42.380 | -1186.648 | -0.000 | -0.000 | 2741.097 | 0.000 | 0.000 | 1.000 | 0.036 | 0.000 | 1.000 | 11310.428 | 0.000 | -42.380 | 0.000 |
| 10 | oracle_family_top1 | 31.000 | 9539.561 | 7899.103 | 1640.457 | 50854.180 | -1520.691 | -15925.897 | 608.416 | 2.000 | 3.000 | 25.000 | 0.806 | 25.000 | 0.000 | 7899.103 | 3.000 | 1640.457 | -1.000 |
| 10 | anchor_top1 | 31.000 | 7899.103 | 7899.103 | 0.000 | 0.000 | -15925.897 | -15925.897 | 407.074 | 3.000 | 3.000 | 0.000 | 0.000 | 0.000 | 0.000 | 7899.103 | 3.000 | 0.000 | 0.000 |
| 10 | model_family_consensus_fallback | 31.000 | 7717.620 | 7899.103 | -181.483 | -5625.981 | -15925.897 | -15925.897 | 407.074 | 3.000 | 3.000 | 7.000 | 0.226 | 3.000 | 4.000 | 7899.103 | 3.000 | -181.483 | 0.000 |
| 11 | oracle_family_top1 | 30.000 | 7692.289 | 7120.767 | 571.522 | 17145.660 | 763.527 | 613.325 | 3608.713 | 0.000 | 0.000 | 23.000 | 0.767 | 23.000 | 0.000 | 7120.767 | 0.000 | 571.522 | 0.000 |
| 11 | anchor_top1 | 30.000 | 7120.767 | 7120.767 | 0.000 | 0.000 | 613.325 | 613.325 | 2689.032 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 7120.767 | 0.000 | 0.000 | 0.000 |
| 11 | model_family_consensus_fallback | 30.000 | 7120.767 | 7120.767 | 0.000 | 0.000 | 613.325 | 613.325 | 2689.032 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 7120.767 | 0.000 | 0.000 | 0.000 |
| 12 | oracle_family_top1 | 31.000 | 6860.263 | 5928.223 | 932.040 | 28893.227 | -6342.633 | -12550.122 | 2382.341 | 1.000 | 1.000 | 22.000 | 0.710 | 22.000 | 0.000 | 5928.223 | 1.000 | 932.040 | 0.000 |
| 12 | anchor_top1 | 31.000 | 5928.223 | 5928.223 | 0.000 | 0.000 | -12550.122 | -12550.122 | 1958.475 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 5928.223 | 1.000 | 0.000 | 0.000 |
| 12 | model_family_consensus_fallback | 31.000 | 5928.223 | 5928.223 | 0.000 | 0.000 | -12550.122 | -12550.122 | 1958.475 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 5928.223 | 1.000 | 0.000 | 0.000 |

## Selected Rules By Held-Out Fold

| heldout_fold | rule_id | selected_from_train | scope | gate_feature | gate_quantile | gate_threshold | support_col | min_support | train_mean_lift_vs_anchor | train_changed_days | train_changed_rate | train_positive_lift_days | train_negative_lift_days | train_loss_days | train_anchor_loss_days | train_rows | valid_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | scope_cat_xgb__gate_anchor_top5_spread_std_q0.50__support_near2_segmented_ge3 | True | cat_xgb | anchor_top5_spread_std | 0.500 | 0.069 | support_near2_segmented | 3 | 54.922 | 9.000 | 0.098 | 8.000 | 0.000 | 4.000 | 4.000 | 460 | 140 |
| valid_2025_10 | scope_margin6__gate_anchor_top1_top2_gap_q0.20__support_near2_all_ge3 | True | margin6 | anchor_top1_top2_gap | 0.200 | -0.026 | support_near2_all | 3 | 53.417 | 7.000 | 0.079 | 4.000 | 2.000 | 1.000 | 1.000 | 445 | 155 |
| valid_2025_11 | scope_margin6__gate_anchor_top1_top2_gap_q0.20__support_near4_all_ge4 | True | margin6 | anchor_top1_top2_gap | 0.200 | -0.053 | support_near4_all | 4 | 62.605 | 7.000 | 0.078 | 4.000 | 3.000 | 4.000 | 4.000 | 450 | 150 |
| valid_2025_12 | scope_margin6__gate_anchor_top1_top2_gap_q0.10__support_near4_all_ge4 | True | margin6 | anchor_top1_top2_gap | 0.100 | -0.075 | support_near4_all | 4 | 41.025 | 3.000 | 0.034 | 2.000 | 1.000 | 3.000 | 3.000 | 445 | 155 |

## Changed-Day Audit

| fold | date | candidate_model | true_profit | anchor_profit | lift_vs_anchor | charge_start | discharge_start | anchor_charge_start | anchor_discharge_start | support_near2_all | support_near4_all | support_near2_segmented | pair_distance_from_anchor | rule_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_10 | 2025-10-05 00:00:00 | margin6 | 7751.515 | 12372.507 | -4620.992 | 32 | 60 | 32 | 62 | 3 | 3 | 3 | 2 | scope_margin6__gate_anchor_top1_top2_gap_q0.20__support_near2_all_ge3 |
| valid_2025_10 | 2025-10-17 00:00:00 | margin6 | 3362.063 | 5525.730 | -2163.666 | 51 | 65 | 49 | 66 | 3 | 4 | 3 | 3 | scope_margin6__gate_anchor_top1_top2_gap_q0.20__support_near2_all_ge3 |
| valid_2025_10 | 2025-10-27 00:00:00 | margin6 | 13837.333 | 15574.312 | -1736.979 | 44 | 69 | 46 | 69 | 3 | 4 | 3 | 2 | scope_margin6__gate_anchor_top1_top2_gap_q0.20__support_near2_all_ge3 |
| valid_2025_09 | 2025-09-19 00:00:00 | xgb180 | 4298.131 | 5484.779 | -1186.648 | 46 | 69 | 46 | 71 | 5 | 5 | 3 | 2 | scope_cat_xgb__gate_anchor_top5_spread_std_q0.50__support_near2_segmented_ge3 |
| valid_2025_10 | 2025-10-28 00:00:00 | margin6 | 9960.895 | 10234.594 | -273.699 | 56 | 69 | 55 | 71 | 3 | 4 | 3 | 3 | scope_margin6__gate_anchor_top1_top2_gap_q0.20__support_near2_all_ge3 |
| valid_2025_10 | 2025-10-02 00:00:00 | margin6 | 6853.112 | 6744.487 | 108.624 | 57 | 72 | 55 | 73 | 4 | 4 | 3 | 3 | scope_margin6__gate_anchor_top1_top2_gap_q0.20__support_near2_all_ge3 |
| valid_2025_10 | 2025-10-06 00:00:00 | margin6 | 14724.205 | 13839.525 | 884.679 | 52 | 65 | 52 | 67 | 3 | 3 | 3 | 2 | scope_margin6__gate_anchor_top1_top2_gap_q0.20__support_near2_all_ge3 |
| valid_2025_10 | 2025-10-23 00:00:00 | margin6 | 3640.913 | 1464.861 | 2176.051 | 48 | 64 | 48 | 66 | 3 | 3 | 3 | 2 | scope_margin6__gate_anchor_top1_top2_gap_q0.20__support_near2_all_ge3 |

## Top Passing Train Rules

| days | mean_profit | anchor_mean_profit | mean_lift_vs_anchor | total_lift_vs_anchor | worst_profit | anchor_worst_profit | p10_profit | loss_days | anchor_loss_days | changed_days | changed_rate | positive_lift_days | negative_lift_days | rule_id | scope | gate_feature | gate_quantile | gate_threshold | support_col | min_support | passes_constraints | heldout_fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 90.000 | 8344.150 | 8281.546 | 62.605 | 5634.421 | -15925.897 | -15925.897 | 855.039 | 4.000 | 4.000 | 7.000 | 0.078 | 4.000 | 3.000 | scope_margin6__gate_anchor_top1_top2_gap_q0.20__support_near4_all_ge4 | margin6 | anchor_top1_top2_gap | 0.200 | -0.053 | support_near4_all | 4 | True | valid_2025_11 |
| 92.000 | 7036.119 | 6981.197 | 54.922 | 5052.785 | -15925.897 | -15925.897 | 873.695 | 4.000 | 4.000 | 9.000 | 0.098 | 8.000 | 0.000 | scope_cat_xgb__gate_anchor_top5_spread_std_q0.50__support_near2_segmented_ge3 | cat_xgb | anchor_top5_spread_std | 0.500 | 0.069 | support_near2_segmented | 3 | True | valid_2025_09 |
| 92.000 | 7035.771 | 6981.197 | 54.574 | 5020.803 | -15925.897 | -15925.897 | 873.695 | 4.000 | 4.000 | 7.000 | 0.076 | 3.000 | 3.000 | scope_margin6__gate_anchor_predicted_spread_q0.10__support_near4_all_ge3 | margin6 | anchor_predicted_spread | 0.100 | 3.979 | support_near4_all | 3 | True | valid_2025_09 |
| 89.000 | 8076.899 | 8023.482 | 53.417 | 4754.122 | -12550.122 | -12550.122 | 2236.612 | 1.000 | 1.000 | 7.000 | 0.079 | 4.000 | 2.000 | scope_margin6__gate_anchor_top1_top2_gap_q0.20__support_near2_all_ge3 | margin6 | anchor_top1_top2_gap | 0.200 | -0.026 | support_near2_all | 3 | True | valid_2025_10 |
| 90.000 | 8329.615 | 8281.546 | 48.069 | 4326.210 | -15925.897 | -15925.897 | 858.650 | 4.000 | 4.000 | 7.000 | 0.078 | 4.000 | 3.000 | scope_segmented_family__gate_anchor_top1_top2_gap_q0.20__support_near4_all_ge4 | segmented_family | anchor_top1_top2_gap | 0.200 | -0.053 | support_near4_all | 4 | True | valid_2025_11 |
| 92.000 | 7028.429 | 6981.197 | 47.232 | 4345.365 | -15925.897 | -15925.897 | 1007.266 | 4.000 | 4.000 | 8.000 | 0.087 | 6.000 | 1.000 | scope_cat_xgb__gate_anchor_top1_top2_gap_q0.20__support_near4_all_ge4 | cat_xgb | anchor_top1_top2_gap | 0.200 | -0.038 | support_near4_all | 4 | True | valid_2025_09 |
| 90.000 | 8324.671 | 8281.546 | 43.125 | 3881.261 | -15925.897 | -15925.897 | 657.929 | 4.000 | 4.000 | 7.000 | 0.078 | 6.000 | 0.000 | scope_cat_xgb__gate_anchor_top5_spread_std_q0.33__support_near2_segmented_ge3 | cat_xgb | anchor_top5_spread_std | 0.330 | 0.060 | support_near2_segmented | 3 | True | valid_2025_11 |
| 90.000 | 8324.451 | 8281.546 | 42.905 | 3861.486 | -15925.897 | -15925.897 | 657.929 | 4.000 | 4.000 | 9.000 | 0.100 | 6.000 | 1.000 | scope_cat_xgb__gate_anchor_top5_spread_std_q0.33__support_near2_all_ge4 | cat_xgb | anchor_top5_spread_std | 0.330 | 0.060 | support_near2_all | 4 | True | valid_2025_11 |
| 92.000 | 7023.880 | 6981.197 | 42.682 | 3926.790 | -15925.897 | -15925.897 | 873.695 | 4.000 | 4.000 | 8.000 | 0.087 | 5.000 | 1.000 | scope_cat_xgb__gate_anchor_top5_spread_std_q0.20__support_near4_all_ge4 | cat_xgb | anchor_top5_spread_std | 0.200 | 0.039 | support_near4_all | 4 | True | valid_2025_09 |
| 92.000 | 7023.649 | 6981.197 | 42.452 | 3905.621 | -15925.897 | -15925.897 | 873.695 | 4.000 | 4.000 | 4.000 | 0.043 | 2.000 | 1.000 | scope_segmented_family__gate_anchor_predicted_spread_q0.10__support_near2_all_ge3 | segmented_family | anchor_predicted_spread | 0.100 | 3.979 | support_near2_all | 3 | True | valid_2025_09 |
| 92.000 | 7023.649 | 6981.197 | 42.452 | 3905.621 | -15925.897 | -15925.897 | 873.695 | 4.000 | 4.000 | 4.000 | 0.043 | 2.000 | 1.000 | scope_seg6__gate_anchor_predicted_spread_q0.10__support_near2_all_ge3 | seg6 | anchor_predicted_spread | 0.100 | 3.979 | support_near2_all | 3 | True | valid_2025_09 |
| 92.000 | 7023.649 | 6981.197 | 42.452 | 3905.621 | -15925.897 | -15925.897 | 873.695 | 4.000 | 4.000 | 4.000 | 0.043 | 2.000 | 1.000 | scope_seg6__gate_anchor_predicted_spread_q0.20__support_near2_all_ge3 | seg6 | anchor_predicted_spread | 0.200 | 4.682 | support_near2_all | 3 | True | valid_2025_09 |
| 92.000 | 7023.649 | 6981.197 | 42.452 | 3905.621 | -15925.897 | -15925.897 | 873.695 | 4.000 | 4.000 | 5.000 | 0.054 | 2.000 | 1.000 | scope_segmented_family__gate_anchor_predicted_spread_q0.20__support_near2_all_ge3 | segmented_family | anchor_predicted_spread | 0.200 | 4.682 | support_near2_all | 3 | True | valid_2025_09 |
| 92.000 | 7022.874 | 6981.197 | 41.677 | 3834.324 | -15925.897 | -15925.897 | 873.695 | 4.000 | 4.000 | 6.000 | 0.065 | 5.000 | 0.000 | scope_cat_xgb__gate_anchor_top5_spread_std_q0.33__support_near2_segmented_ge3 | cat_xgb | anchor_top5_spread_std | 0.330 | 0.047 | support_near2_segmented | 3 | True | valid_2025_09 |
| 92.000 | 7022.874 | 6981.197 | 41.677 | 3834.324 | -15925.897 | -15925.897 | 873.695 | 4.000 | 4.000 | 7.000 | 0.076 | 5.000 | 0.000 | scope_cat_xgb__gate_anchor_top5_spread_std_q0.33__support_near2_all_ge4 | cat_xgb | anchor_top5_spread_std | 0.330 | 0.047 | support_near2_all | 4 | True | valid_2025_09 |
| 92.000 | 7022.337 | 6981.197 | 41.140 | 3784.869 | -15925.897 | -15925.897 | 873.695 | 4.000 | 4.000 | 5.000 | 0.054 | 2.000 | 2.000 | scope_all_experts__gate_anchor_predicted_spread_q0.10__support_near2_all_ge3 | all_experts | anchor_predicted_spread | 0.100 | 3.979 | support_near2_all | 3 | True | valid_2025_09 |
| 92.000 | 7022.337 | 6981.197 | 41.140 | 3784.869 | -15925.897 | -15925.897 | 873.695 | 4.000 | 4.000 | 6.000 | 0.065 | 2.000 | 2.000 | scope_all_experts__gate_anchor_predicted_spread_q0.20__support_near2_all_ge3 | all_experts | anchor_predicted_spread | 0.200 | 4.682 | support_near2_all | 3 | True | valid_2025_09 |
| 89.000 | 8750.993 | 8709.968 | 41.025 | 3651.185 | -15925.897 | -15925.897 | 1363.173 | 3.000 | 3.000 | 3.000 | 0.034 | 2.000 | 1.000 | scope_margin6__gate_anchor_top1_top2_gap_q0.10__support_near4_all_ge4 | margin6 | anchor_top1_top2_gap | 0.100 | -0.075 | support_near4_all | 4 | True | valid_2025_12 |
| 90.000 | 8322.114 | 8281.546 | 40.569 | 3651.185 | -15925.897 | -15925.897 | 855.039 | 4.000 | 4.000 | 3.000 | 0.033 | 2.000 | 1.000 | scope_margin6__gate_anchor_top1_top2_gap_q0.10__support_near4_all_ge4 | margin6 | anchor_top1_top2_gap | 0.100 | -0.079 | support_near4_all | 4 | True | valid_2025_11 |
| 89.000 | 8063.575 | 8023.482 | 40.093 | 3568.301 | -8661.846 | -12550.122 | 2236.612 | 1.000 | 1.000 | 6.000 | 0.067 | 3.000 | 2.000 | scope_segmented_family__gate_anchor_predicted_spread_q0.20__support_near2_all_ge3 | segmented_family | anchor_predicted_spread | 0.200 | 4.541 | support_near2_all | 3 | True | valid_2025_10 |

## Decision

- The fold-safe fallback does not improve the anchor enough, or worsens risk.
- Reject this standalone rule family; no submit.
- Oracle among the five top1 model-family choices is still `982.402`/day above anchor, so the remaining problem is learning when to trust which family, not whether an alternative family sometimes has the better pair.

## Artifacts

- `reports/model_family_consensus_fallback_selected_daily.csv`
- `reports/model_family_consensus_fallback_rules.csv`
- `reports/model_family_consensus_fallback_rule_search.csv`
- `reports/model_family_consensus_fallback_overall.csv`
- `reports/model_family_consensus_fallback_by_fold.csv`
- `reports/model_family_consensus_fallback_by_month.csv`
- `reports/model_family_consensus_fallback_expert_summary.csv`
