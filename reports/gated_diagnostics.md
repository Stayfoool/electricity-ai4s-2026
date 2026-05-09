# Gated Diagnostics

## Scope

This diagnostic searches simple ex-ante rules using only same-day official features and model outputs. Hidden true prices are used only for validation.

Candidate models: `ens_baseline_baseline_last_180d, lgb_baseline_last_180d, lgb_business_last_180d, cat_baseline_last_180d, ens_base_base180_cat180`.

## Cross-Fold Gate Search

For each held-out validation month, the rule is learned on the other three months and then evaluated on the held-out month.

| heldout_fold   | feature                   |   threshold | left_model                      | right_model                     |   train_mean_profit |   train_lift_vs_champion |   valid_gated_mean_profit |   valid_champion_mean_profit |   valid_gated_lift_vs_champion | best_static_model_from_train    |   valid_best_static_mean_profit |   valid_days |
|:---------------|:--------------------------|------------:|:--------------------------------|:--------------------------------|--------------------:|-------------------------:|--------------------------:|-----------------------------:|-------------------------------:|:--------------------------------|--------------------------------:|-------------:|
| valid_2025_09  | renewable_ratio_mean      |      0.3124 | lgb_business_last_180d          | ens_base_base180_cat180         |           7073.5522 |                 276.5347 |                10483.9150 |                   10632.0859 |                      -148.1709 | ens_base_base180_cat180         |                      10146.2193 |           28 |
| valid_2025_10  | champion_predicted_spread |      5.9967 | cat_baseline_last_180d          | ens_baseline_baseline_last_180d |           7953.2460 |                 182.6118 |                 7333.1640 |                    7465.7279 |                      -132.5639 | ens_baseline_baseline_last_180d |                       7465.7279 |           31 |
| valid_2025_11  | net_load_mean             |      1.9460 | ens_baseline_baseline_last_180d | lgb_business_last_180d          |           8180.5114 |                 287.2004 |                 6844.0536 |                    7087.5339 |                      -243.4802 | lgb_business_last_180d          |                       6655.2774 |           30 |
| valid_2025_12  | wind_ratio_mean           |      0.1423 | lgb_business_last_180d          | ens_baseline_baseline_last_180d |           8595.8485 |                 261.4442 |                 5841.0046 |                    5847.1620 |                        -6.1575 | ens_baseline_baseline_last_180d |                       5847.1620 |           31 |

Summary:

- Champion mean across held-out folds: `7758.1274`.
- Single-split gated selector mean: `7625.5343`.
- Gated lift vs champion: `-132.5931`.
- Best-static-model-from-train mean: `7528.5966`.

Selected model counts in held-out evaluation:

| selected_model                  |   days |
|:--------------------------------|-------:|
| ens_baseline_baseline_last_180d |     79 |
| ens_base_base180_cat180         |     21 |
| cat_baseline_last_180d          |     11 |
| lgb_business_last_180d          |      9 |

## Month-Level Pattern

|   month |   days | best_model                      |   best_model_mean_profit |   champion_mean_profit |   lift_vs_champion |   ens_baseline_baseline_last_180d__mean_profit |   lgb_baseline__mean_profit |   lgb_baseline_last_180d__mean_profit |   lgb_business__mean_profit |   lgb_business_last_180d__mean_profit |   ens_business_business_last_180d__mean_profit |   cat_baseline_last_180d__mean_profit |   ens_base_base180_cat180__mean_profit |
|--------:|-------:|:--------------------------------|-------------------------:|-----------------------:|-------------------:|-----------------------------------------------:|----------------------------:|--------------------------------------:|----------------------------:|--------------------------------------:|-----------------------------------------------:|--------------------------------------:|---------------------------------------:|
|       9 |     28 | ens_baseline_baseline_last_180d |               10632.0859 |             10632.0859 |             0.0000 |                                     10632.0859 |                   9781.2473 |                            10504.6092 |                  10523.5325 |                            10342.4305 |                                     10250.7329 |                             9923.9965 |                             10146.2193 |
|      10 |     31 | lgb_business_last_180d          |                7735.8209 |              7465.7279 |           270.0930 |                                      7465.7279 |                   7245.4625 |                             7450.1946 |                   7369.3692 |                             7735.8209 |                                      7560.8063 |                             7381.8917 |                              7538.3467 |
|      11 |     30 | cat_baseline_last_180d          |                7192.4075 |              7087.5339 |           104.8736 |                                      7087.5339 |                   6927.4704 |                             6856.0234 |                   6781.9621 |                             6655.2774 |                                      7086.4830 |                             7192.4075 |                              7102.6184 |
|      12 |     31 | ens_business_business_last_180d |                5901.9580 |              5847.1620 |            54.7960 |                                      5847.1620 |                   5861.9331 |                             5859.8522 |                   5795.3735 |                             5881.7497 |                                      5901.9580 |                             5729.3373 |                              5887.2181 |

Month is diagnostic only. It is risky as a direct gate because validation months are 2025-09..2025-12 while test months are 2026-01..2026-02.

## Feature Buckets With Largest In-Sample Selector Lift

| feature                              | bucket                         |   days | best_model_in_bucket            |   best_model_mean_profit |   champion_mean_profit |   lift_vs_champion |   champion_loss_days |   best_model_share_oracle |
|:-------------------------------------|:-------------------------------|-------:|:--------------------------------|-------------------------:|-----------------------:|-------------------:|---------------------:|--------------------------:|
| net_load_mean                        | (1.778, 2.566]                 |     40 | lgb_business_last_180d          |                8642.6057 |              8208.8666 |           433.7391 |                    6 |                    0.2000 |
| wind_ratio_mean                      | (0.0544, 0.247]                |     40 | lgb_business_last_180d          |                9120.4240 |              8704.1824 |           416.2416 |                    6 |                    0.2000 |
| renewable_ratio_mean                 | (0.103, 0.39]                  |     40 | lgb_business_last_180d          |                8420.0650 |              8031.6170 |           388.4479 |                    6 |                    0.2000 |
| renewable_ratio_max                  | (0.223, 0.75]                  |     40 | lgb_business_last_180d          |                7663.9735 |              7326.8331 |           337.1404 |                    8 |                    0.2000 |
| champion_predicted_spread            | (1.6640000000000001, 5.962]    |     40 | cat_baseline_last_180d          |                4822.1506 |              4528.3452 |           293.8054 |                    5 |                    0.2750 |
| non_market_ratio_mean                | (0.137, 0.182]                 |     40 | lgb_business_last_180d          |                8914.8505 |              8636.3117 |           278.5388 |                    6 |                    0.1500 |
| lgb180_predicted_spread              | (2.057, 6.181]                 |     40 | cat_baseline_last_180d          |                4755.0360 |              4516.5950 |           238.4410 |                    5 |                    0.2750 |
| renewable_ratio_max                  | (0.984, 1.223]                 |     40 | cat_baseline_last_180d          |                6603.8475 |              6392.7364 |           211.1111 |                    2 |                    0.2500 |
| renewable_ratio_mean                 | (0.574, 0.781]                 |     40 | cat_baseline_last_180d          |                6016.8196 |              5821.1842 |           195.6354 |                    2 |                    0.2500 |
| champion_lgb180_charge_start_diff    | (1.0, 20.0]                    |     27 | cat_baseline_last_180d          |                7092.9177 |              6915.3754 |           177.5424 |                    3 |                    0.2593 |
| renewable_ratio_std                  | (0.0582, 0.18]                 |     40 | lgb_business_last_180d          |                6404.2653 |              6234.7863 |           169.4790 |                    8 |                    0.1500 |
| net_load_std                         | (0.162, 0.538]                 |     40 | lgb_business_last_180d          |                6295.9118 |              6128.0418 |           167.8700 |                    8 |                    0.1250 |
| load_std                             | (0.0785, 0.103]                |     40 | ens_business_business_last_180d |                6436.0746 |              6269.7112 |           166.3634 |                    4 |                    0.0500 |
| tie_line_ratio_mean                  | (0.032799999999999996, 0.0603] |     40 | lgb_business_last_180d          |                7997.5468 |              7835.7660 |           161.7808 |                    5 |                    0.2000 |
| champion_cat180_discharge_start_diff | (-0.001, 1.0]                  |     62 | lgb_business_last_180d          |                7516.6749 |              7358.7998 |           157.8751 |                    7 |                    0.1613 |
| net_load_range                       | (0.559, 1.663]                 |     40 | lgb_business_last_180d          |                6044.7519 |              5887.7780 |           156.9739 |                    8 |                    0.1250 |
| champion_predicted_spread            | (9.193, 21.118]                |     40 | lgb_business_last_180d          |               10922.3579 |             10765.9569 |           156.4010 |                    3 |                    0.1500 |
| net_load_mean                        | (0.626, 1.24]                  |     40 | cat_baseline_last_180d          |                6007.8191 |              5866.6645 |           141.1546 |                    2 |                    0.2500 |
| wind_ratio_mean                      | (0.423, 0.644]                 |     40 | cat_baseline_last_180d          |                6420.8180 |              6286.8123 |           134.0056 |                    2 |                    0.2500 |
| net_load_std                         | (0.67, 0.897]                  |     40 | lgb_business                    |                9334.6614 |              9210.6881 |           123.9733 |                    1 |                    0.3000 |
| champion_cat180_charge_start_diff    | (1.0, 4.0]                     |     33 | cat_baseline_last_180d          |                7801.2065 |              7682.5350 |           118.6715 |                    3 |                    0.2727 |
| non_market_ratio_mean                | (0.203, 0.251]                 |     40 | cat_baseline_last_180d          |                6314.6127 |              6205.3248 |           109.2880 |                    2 |                    0.2250 |
| load_std                             | (0.040799999999999996, 0.0702] |     40 | lgb_business_last_180d          |                9442.5193 |              9346.9407 |            95.5786 |                    5 |                    0.1250 |
| lgb180_predicted_spread              | (9.369, 20.317]                |     40 | lgb_business_last_180d          |               11010.2343 |             10921.1187 |            89.1157 |                    2 |                    0.1250 |
| renewable_ratio_std                  | (0.216, 0.312]                 |     40 | cat_baseline_last_180d          |                9554.3863 |              9469.6256 |            84.7607 |                    1 |                    0.2500 |

## Decision

- The simple cross-fold gate does not beat the current champion.
- Do not promote a gated selector yet.
- Use the feature bucket table only as evidence for targeted feature or ensemble design, not as a submission rule.
