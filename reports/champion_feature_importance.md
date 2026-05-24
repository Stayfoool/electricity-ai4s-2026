# Champion Feature Importance And Ablation

- Config: `configs/ensemble_champion_segmented6_prior_5fold.yaml`.
- Scope: current champion 5fold ensemble.
- Importance: LightGBM gain/split, normalized by ensemble member weight and segment coverage.
- Permutation: validation feature values shuffled within the same 15-minute slot across days.
- Ablation: retrain the full ensemble after removing candidate features.

## Top Weighted Gain Features

| feature | weighted_gain_share | weighted_split_share | gain_sum | split_sum | used_models |
| --- | --- | --- | --- | --- | --- |
| 风光总加预测值 | 0.3586 | 0.1221 | 698417.8034 | 24355.0000 | 40 |
| 风电预测值 | 0.3493 | 0.1381 | 689419.3654 | 29377.0000 | 40 |
| month | 0.0572 | 0.1118 | 122814.3674 | 22280.0000 | 40 |
| 光伏预测值 | 0.0410 | 0.0965 | 88348.8479 | 20937.0000 | 40 |
| 联络线预测值 | 0.0379 | 0.0915 | 80976.7308 | 19566.0000 | 40 |
| 非市场化机组预测值 | 0.0364 | 0.0970 | 72534.3823 | 21582.0000 | 40 |
| 系统负荷预测值 | 0.0311 | 0.0941 | 63113.7819 | 20300.0000 | 40 |
| dayofweek | 0.0266 | 0.0725 | 56964.0561 | 16742.0000 | 40 |
| 水电预测值 | 0.0252 | 0.0890 | 49629.3596 | 20794.0000 | 40 |
| quarter | 0.0110 | 0.0305 | 22970.4473 | 6708.0000 | 40 |
| hour_cos | 0.0076 | 0.0082 | 22750.5428 | 1882.0000 | 13 |
| quarter_sin | 0.0060 | 0.0161 | 16941.3157 | 3883.0000 | 38 |
| quarter_cos | 0.0048 | 0.0116 | 13379.1487 | 2863.0000 | 30 |
| hour | 0.0034 | 0.0101 | 7003.2615 | 2276.0000 | 40 |
| hour_sin | 0.0025 | 0.0067 | 7244.1786 | 1570.0000 | 30 |
| is_weekend | 0.0013 | 0.0034 | 2680.1772 | 787.0000 | 36 |
| minute | 0.0001 | 0.0007 | 97.7288 | 217.0000 | 30 |

## Member/Segment Raw-Feature Importance

| member_name | segment | feature | weighted_gain_share | split_sum |
| --- | --- | --- | --- | --- |
| lgb_baseline | all | 风光总加预测值 | 0.0918 | 6538.0000 |
| lgb_baseline | all | 风电预测值 | 0.0771 | 7557.0000 |
| lgb_baseline | all | 光伏预测值 | 0.0119 | 6681.0000 |
| lgb_baseline | all | 联络线预测值 | 0.0110 | 6998.0000 |
| lgb_baseline | all | 非市场化机组预测值 | 0.0083 | 6507.0000 |
| lgb_baseline | all | 系统负荷预测值 | 0.0083 | 6662.0000 |
| lgb_baseline | all | 水电预测值 | 0.0053 | 6094.0000 |
| lgb_baseline_last_180d | all | 风电预测值 | 0.0999 | 6453.0000 |
| lgb_baseline_last_180d | all | 风光总加预测值 | 0.0680 | 5264.0000 |
| lgb_baseline_last_180d | all | 光伏预测值 | 0.0106 | 4852.0000 |
| lgb_baseline_last_180d | all | 联络线预测值 | 0.0099 | 4855.0000 |
| lgb_baseline_last_180d | all | 非市场化机组预测值 | 0.0093 | 5032.0000 |
| lgb_baseline_last_180d | all | 系统负荷预测值 | 0.0069 | 4596.0000 |
| lgb_baseline_last_180d | all | 水电预测值 | 0.0065 | 4908.0000 |
| lgb_segmented_6_last_180d | 00_24 | 风光总加预测值 | 0.0502 | 3048.0000 |
| lgb_segmented_6_last_180d | 00_24 | 风电预测值 | 0.0389 | 2615.0000 |
| lgb_segmented_6_last_180d | 00_24 | 联络线预测值 | 0.0061 | 2383.0000 |
| lgb_segmented_6_last_180d | 00_24 | 光伏预测值 | 0.0056 | 3401.0000 |
| lgb_segmented_6_last_180d | 00_24 | 水电预测值 | 0.0043 | 2943.0000 |
| lgb_segmented_6_last_180d | 00_24 | 系统负荷预测值 | 0.0042 | 2463.0000 |
| lgb_segmented_6_last_180d | 00_24 | 非市场化机组预测值 | 0.0031 | 2175.0000 |
| lgb_segmented_6_last_180d | 24_40 | 风电预测值 | 0.0362 | 4338.0000 |
| lgb_segmented_6_last_180d | 24_40 | 风光总加预测值 | 0.0283 | 2586.0000 |
| lgb_segmented_6_last_180d | 24_40 | 系统负荷预测值 | 0.0025 | 2127.0000 |
| lgb_segmented_6_last_180d | 24_40 | 非市场化机组预测值 | 0.0025 | 2049.0000 |
| lgb_segmented_6_last_180d | 24_40 | 联络线预测值 | 0.0022 | 1562.0000 |
| lgb_segmented_6_last_180d | 24_40 | 光伏预测值 | 0.0021 | 1297.0000 |
| lgb_segmented_6_last_180d | 24_40 | 水电预测值 | 0.0017 | 2050.0000 |
| lgb_segmented_6_last_180d | 40_56 | 风光总加预测值 | 0.0477 | 1780.0000 |
| lgb_segmented_6_last_180d | 40_56 | 风电预测值 | 0.0156 | 1746.0000 |
| lgb_segmented_6_last_180d | 40_56 | 非市场化机组预测值 | 0.0063 | 1750.0000 |
| lgb_segmented_6_last_180d | 40_56 | 光伏预测值 | 0.0040 | 1314.0000 |
| lgb_segmented_6_last_180d | 40_56 | 联络线预测值 | 0.0022 | 774.0000 |
| lgb_segmented_6_last_180d | 40_56 | 系统负荷预测值 | 0.0020 | 1012.0000 |
| lgb_segmented_6_last_180d | 40_56 | 水电预测值 | 0.0012 | 923.0000 |

## Same-Slot Permutation Summary

Negative `delta_avg_profit_mean` means shuffling the feature hurt profit, so the feature is useful under this diagnostic.

| feature | delta_avg_profit_mean | delta_avg_profit_min | delta_oracle_ratio_mean | delta_loss_days_total | delta_rmse_mean |
| --- | --- | --- | --- | --- | --- |
| 风光总加预测值 | -919.5847 | -1326.4008 | -0.0760 | 3 | 0.1120 |
| 风电预测值 | -480.5336 | -1635.5126 | -0.0352 | 4 | 0.0864 |
| 光伏预测值 | -167.6630 | -402.7049 | -0.0119 | 0 | 0.0025 |
| 水电预测值 | -157.9409 | -515.9933 | -0.0098 | 4 | -0.0004 |
| 非市场化机组预测值 | -60.0402 | -198.7573 | -0.0062 | 1 | 0.0003 |
| 联络线预测值 | -46.2449 | -188.5956 | -0.0034 | 2 | -0.0014 |
| 系统负荷预测值 | 1.7733 | -355.0296 | -0.0007 | 0 | -0.0011 |

## Raw Feature Drop Backtest

Positive delta means removing the feature improved validation profit; negative delta means removing it hurt.

| run_name | drop_cols | avg_profit_fold_mean | delta_avg_profit_fold_mean | avg_profit_day_weighted | delta_avg_profit_day_weighted | loss_days_total | delta_loss_days_total | rmse_mean | delta_rmse_mean | curve_z_rmse_mean | delta_curve_z_rmse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop_raw__风光总加预测值 | 风光总加预测值 | 8985.4527 | -296.9516 | 9642.4930 | -309.5417 | 6 | 1 | 0.7568 | 0.0116 | 0.8413 | 0.0110 |
| drop_raw__水电预测值 | 水电预测值 | 8963.7442 | -318.6601 | 9563.0243 | -389.0104 | 8 | 3 | 0.7438 | -0.0014 | 0.8292 | -0.0010 |
| drop_raw__光伏预测值 | 光伏预测值 | 8938.1437 | -344.2605 | 9553.8882 | -398.1464 | 7 | 2 | 0.7434 | -0.0018 | 0.8291 | -0.0012 |
| drop_raw__风电预测值 | 风电预测值 | 8938.0391 | -344.3651 | 9667.2605 | -284.7741 | 5 | 0 | 0.7520 | 0.0067 | 0.8444 | 0.0142 |
| drop_raw__系统负荷预测值 | 系统负荷预测值 | 8822.3955 | -460.0088 | 9493.7682 | -458.2665 | 8 | 3 | 0.7410 | -0.0042 | 0.8319 | 0.0016 |
| drop_raw__联络线预测值 | 联络线预测值 | 8817.3693 | -465.0350 | 9519.8986 | -432.1361 | 7 | 2 | 0.7375 | -0.0077 | 0.8365 | 0.0063 |
| drop_raw__非市场化机组预测值 | 非市场化机组预测值 | 8751.8355 | -530.5687 | 9366.6403 | -585.3943 | 7 | 2 | 0.7477 | 0.0025 | 0.8344 | 0.0041 |

## Time Feature Group Drop Backtest

| run_name | drop_cols | avg_profit_fold_mean | delta_avg_profit_fold_mean | avg_profit_day_weighted | delta_avg_profit_day_weighted | loss_days_total | delta_loss_days_total | rmse_mean | delta_rmse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop_time_clock | hour,minute,quarter,hour_sin,hour_cos,quarter_sin,quarter_cos | 9147.5574 | -134.8468 | 9847.8721 | -104.1626 | 6 | 1 | 0.7496 | 0.0043 |
| drop_time_calendar | dayofweek,month,is_weekend | 8680.5907 | -601.8136 | 9303.2895 | -648.7452 | 6 | 1 | 0.7545 | 0.0093 |
| drop_time_all | hour,minute,quarter,dayofweek,month,is_weekend,hour_sin,hour_cos,quarter_sin,quarter_cos | 8601.8418 | -680.5624 | 9191.5740 | -760.4606 | 5 | 0 | 0.7558 | 0.0106 |

## Interpretation Rules

- Gain/split 高只能说明模型经常用它，不等价于删掉会掉分。
- Permutation 可能破坏特征之间的物理相关性，所以只作为方向信号。
- 删特征回测最接近最终目标，因为它完整重训并经过 dispatch/profit 评估。
- 如果某特征 gain 高、permutation 后收益下降、删除后也下降，说明它是稳定正贡献。
- 如果某特征 gain 低、permutation 后收益上升、删除后也上升，才可以考虑从 champion 中移除。
