# Current Prior Top-K Consensus Diagnostics

## Scope

Anchor model is the current online-best champion with dispatch prior:

- `ens_champion_segmented6_prior`

All rules are constrained to the anchor model's predicted top-10 legal pairs. Other experts only provide support/rank signals. Fold-specific dispatch priors are rebuilt from training labels exactly as in the champion-prior backtest.

## Top-K Upper Bound

| model                                 |   days |   top1_mean_profit |   top10_best_mean_profit |   top10_lift_mean |   positive_lift_days |   oracle_in_top10_rate |
|:--------------------------------------|-------:|-------------------:|-------------------------:|------------------:|---------------------:|-----------------------:|
| lgb_segmented_6_margin_core_last_180d |    176 |          9276.2341 |               10458.5159 |         1182.2818 |                  139 |                 0.0511 |
| lgb_segmented_6_last_180d             |    176 |          9318.7357 |               10281.9992 |          963.2635 |                  140 |                 0.0398 |
| ens_champion_segmented6_prior         |    176 |          9952.0347 |               10689.6177 |          737.5830 |                  131 |                 0.0455 |

## Rule Summary

| rule                       |   days |   mean_profit |   worst_profit |   loss_days |   mean_lift_vs_anchor |   positive_lift_days |   negative_lift_days |   changed_days |   mean_anchor_rank |   mean_support_count |   mean_miss_vs_oracle_top10 |
|:---------------------------|-------:|--------------:|---------------:|------------:|----------------------:|---------------------:|---------------------:|---------------:|-------------------:|---------------------:|----------------------------:|
| anchor_top1                |    176 |     9952.0347 |    -15925.8974 |           4 |                0.0000 |                    0 |                    0 |              0 |             1.0000 |               2.4545 |                    737.5830 |
| support2_then_anchor_rank  |    176 |     9876.1295 |    -15925.8974 |           4 |              -75.9052 |                    6 |                   12 |             23 |             1.3977 |               2.6818 |                    813.4882 |
| anchor_rank_le5_support2   |    176 |     9855.9466 |    -15925.8974 |           4 |              -96.0880 |                   13 |                   17 |             41 |             1.4545 |               2.7614 |                    833.6711 |
| support_then_anchor_rank   |    176 |     9848.2552 |    -15925.8974 |           4 |             -103.7795 |                   18 |                   24 |             52 |             1.8920 |               2.8636 |                    841.3625 |
| unanimous_then_anchor_rank |    176 |     9841.8313 |    -15925.8974 |           4 |             -110.2034 |                   17 |                   23 |             49 |             1.8239 |               2.8466 |                    847.7864 |
| support_score              |    176 |     9808.2304 |    -15925.8974 |           3 |             -143.8043 |                   29 |                   45 |             99 |             2.5114 |               2.8636 |                    881.3873 |

## Fold Summary

| fold               | rule                       |   days |   mean_profit |   worst_profit |   loss_days |   mean_lift_vs_anchor |   changed_days |
|:-------------------|:---------------------------|-------:|--------------:|---------------:|------------:|----------------------:|---------------:|
| valid_2025_09      | anchor_rank_le5_support2   |     28 |    11002.5476 |        95.4409 |           0 |             -307.8806 |             10 |
| valid_2025_09      | anchor_top1                |     28 |    11310.4282 |         0.0000 |           0 |                0.0000 |              0 |
| valid_2025_09      | support2_then_anchor_rank  |     28 |    11226.3153 |         0.0000 |           0 |              -84.1128 |              4 |
| valid_2025_09      | support_score              |     28 |    10674.5806 |        95.4409 |           0 |             -635.8475 |             19 |
| valid_2025_09      | support_then_anchor_rank   |     28 |    11065.0386 |        95.4409 |           0 |             -245.3896 |             11 |
| valid_2025_09      | unanimous_then_anchor_rank |     28 |    11107.3053 |        95.4409 |           0 |             -203.1229 |              9 |
| valid_2025_10      | anchor_rank_le5_support2   |     31 |     7951.7876 |    -15925.8974 |           3 |               52.6844 |             10 |
| valid_2025_10      | anchor_top1                |     31 |     7899.1033 |    -15925.8974 |           3 |                0.0000 |              0 |
| valid_2025_10      | support2_then_anchor_rank  |     31 |     7995.0592 |    -15925.8974 |           3 |               95.9559 |              5 |
| valid_2025_10      | support_score              |     31 |     8056.3241 |    -15925.8974 |           2 |              157.2208 |             18 |
| valid_2025_10      | support_then_anchor_rank   |     31 |     8007.5128 |    -15925.8974 |           3 |              108.4095 |             11 |
| valid_2025_10      | unanimous_then_anchor_rank |     31 |     8007.5128 |    -15925.8974 |           3 |              108.4095 |             11 |
| valid_2025_11      | anchor_rank_le5_support2   |     30 |     6962.1512 |       613.3252 |           0 |             -158.6155 |              7 |
| valid_2025_11      | anchor_top1                |     30 |     7120.7667 |       613.3252 |           0 |                0.0000 |              0 |
| valid_2025_11      | support2_then_anchor_rank  |     30 |     6893.7832 |       613.3252 |           0 |             -226.9835 |              4 |
| valid_2025_11      | support_score              |     30 |     6943.3534 |       454.2571 |           0 |             -177.4133 |             18 |
| valid_2025_11      | support_then_anchor_rank   |     30 |     6941.7733 |       613.3252 |           0 |             -178.9934 |             11 |
| valid_2025_11      | unanimous_then_anchor_rank |     30 |     6941.7733 |       613.3252 |           0 |             -178.9934 |             11 |
| valid_2025_12      | anchor_rank_le5_support2   |     31 |     5951.6559 |    -11244.2975 |           1 |               23.4326 |              5 |
| valid_2025_12      | anchor_top1                |     31 |     5928.2233 |    -12550.1225 |           1 |                0.0000 |              0 |
| valid_2025_12      | support2_then_anchor_rank  |     31 |     5958.9212 |    -11244.2975 |           1 |               30.6979 |              3 |
| valid_2025_12      | support_score              |     31 |     6032.2334 |    -10239.8016 |           1 |              104.0101 |             14 |
| valid_2025_12      | support_then_anchor_rank   |     31 |     5984.0590 |    -10239.8016 |           1 |               55.8357 |              5 |
| valid_2025_12      | unanimous_then_anchor_rank |     31 |     5984.0590 |    -10239.8016 |           1 |               55.8357 |              5 |
| valid_2025_jan_feb | anchor_rank_le5_support2   |     56 |    14048.2856 |         8.7837 |           0 |             -105.2143 |              9 |
| valid_2025_jan_feb | anchor_top1                |     56 |    14153.4998 |         8.7837 |           0 |                0.0000 |              0 |
| valid_2025_jan_feb | support2_then_anchor_rank  |     56 |    14008.4835 |         8.7837 |           0 |             -145.0163 |              7 |
| valid_2025_jan_feb | support_score              |     56 |    13969.9001 |         5.7614 |           0 |             -183.5997 |             30 |
| valid_2025_jan_feb | support_then_anchor_rank   |     56 |    13954.9983 |         8.7837 |           0 |             -198.5016 |             14 |
| valid_2025_jan_feb | unanimous_then_anchor_rank |     56 |    13913.6755 |         8.7837 |           0 |             -239.8243 |             13 |

## Regime Summary

| regime                 | rule                       |   days |   mean_profit |   worst_profit |   loss_days |   mean_lift_vs_anchor |   changed_days |
|:-----------------------|:---------------------------|-------:|--------------:|---------------:|------------:|----------------------:|---------------:|
| all_5fold              | anchor_top1                |    176 |     9952.0347 |    -15925.8974 |           4 |                0.0000 |              0 |
| all_5fold              | support2_then_anchor_rank  |    176 |     9876.1295 |    -15925.8974 |           4 |              -75.9052 |             23 |
| all_5fold              | anchor_rank_le5_support2   |    176 |     9855.9466 |    -15925.8974 |           4 |              -96.0880 |             41 |
| all_5fold              | support_then_anchor_rank   |    176 |     9848.2552 |    -15925.8974 |           4 |             -103.7795 |             52 |
| all_5fold              | unanimous_then_anchor_rank |    176 |     9841.8313 |    -15925.8974 |           4 |             -110.2034 |             49 |
| all_5fold              | support_score              |    176 |     9808.2304 |    -15925.8974 |           3 |             -143.8043 |             99 |
| jan_feb_like           | anchor_top1                |     56 |    14153.4998 |         8.7837 |           0 |                0.0000 |              0 |
| jan_feb_like           | anchor_rank_le5_support2   |     56 |    14048.2856 |         8.7837 |           0 |             -105.2143 |              9 |
| jan_feb_like           | support2_then_anchor_rank  |     56 |    14008.4835 |         8.7837 |           0 |             -145.0163 |              7 |
| jan_feb_like           | support_score              |     56 |    13969.9001 |         5.7614 |           0 |             -183.5997 |             30 |
| jan_feb_like           | support_then_anchor_rank   |     56 |    13954.9983 |         8.7837 |           0 |             -198.5016 |             14 |
| jan_feb_like           | unanimous_then_anchor_rank |     56 |    13913.6755 |         8.7837 |           0 |             -239.8243 |             13 |
| late_winter_12_jan_feb | anchor_top1                |     87 |    11222.6542 |    -12550.1225 |           1 |                0.0000 |              0 |
| late_winter_12_jan_feb | anchor_rank_le5_support2   |     87 |    11163.2796 |    -11244.2975 |           1 |              -59.3746 |             14 |
| late_winter_12_jan_feb | support_score              |     87 |    11141.5361 |    -10239.8016 |           1 |              -81.1181 |             44 |
| late_winter_12_jan_feb | support2_then_anchor_rank  |     87 |    11140.2487 |    -11244.2975 |           1 |              -82.4055 |             10 |
| late_winter_12_jan_feb | support_then_anchor_rank   |     87 |    11114.7785 |    -10239.8016 |           1 |             -107.8757 |             19 |
| late_winter_12_jan_feb | unanimous_then_anchor_rank |     87 |    11088.1800 |    -10239.8016 |           1 |             -134.4742 |             18 |
| standard_09_12         | anchor_top1                |    120 |     7991.3509 |    -15925.8974 |           4 |                0.0000 |              0 |
| standard_09_12         | support2_then_anchor_rank  |    120 |     7947.6977 |    -15925.8974 |           4 |              -43.6533 |             16 |
| standard_09_12         | unanimous_then_anchor_rank |    120 |     7941.6373 |    -15925.8974 |           4 |              -49.7137 |             36 |
| standard_09_12         | support_then_anchor_rank   |    120 |     7931.7750 |    -15925.8974 |           4 |              -59.5759 |             38 |
| standard_09_12         | anchor_rank_le5_support2   |    120 |     7899.5218 |    -15925.8974 |           4 |              -91.8291 |             32 |
| standard_09_12         | support_score              |    120 |     7866.1179 |    -15925.8974 |           3 |             -125.2331 |             69 |
| winter_11_12_jan_feb   | anchor_top1                |    117 |    10170.8882 |    -12550.1225 |           1 |                0.0000 |              0 |
| winter_11_12_jan_feb   | anchor_rank_le5_support2   |    117 |    10086.0672 |    -11244.2975 |           1 |              -84.8210 |             21 |
| winter_11_12_jan_feb   | support_score              |    117 |    10065.0790 |    -10239.8016 |           1 |             -105.8092 |             62 |
| winter_11_12_jan_feb   | support2_then_anchor_rank  |    117 |    10051.4114 |    -11244.2975 |           1 |             -119.4768 |             14 |
| winter_11_12_jan_feb   | support_then_anchor_rank   |    117 |    10044.7772 |    -10239.8016 |           1 |             -126.1110 |             30 |
| winter_11_12_jan_feb   | unanimous_then_anchor_rank |    117 |    10024.9988 |    -10239.8016 |           1 |             -145.8894 |             29 |

## Decision

- Anchor-limited consensus does not beat anchor top1.
- Keep current champion-prior submit as the safer path.
