# Window Deep Experiment

- Model: `sequence_window_tcn_5fold`.
- Input: daily 96-point feature curve.
- Target: 89 rolling 2-hour window average prices.
- Dispatch: enumerate legal charge/discharge window pairs using predicted window scores.

## Overall
| model | mean_profit | min_profit | mean_window_rmse | loss_days | mean_regret | mean_charge_gap | mean_discharge_gap | profit_delta_vs_champion | loss_day_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence_window_tcn_5fold | 8318.7990 | 5277.9750 | 0.6430 | 9 | 4297.5036 | 14.9950 | 11.9482 | -963.6052 | 4 |

## Fold Summary
| fold | model | valid_rows | valid_days | window_mae | window_rmse | avg_profit | oracle_avg_profit | oracle_ratio | avg_regret | loss_days | mean_abs_charge_gap | mean_abs_discharge_gap | champion_avg_profit | champion_loss_days | champion_avg_regret | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | sequence_window_tcn_5fold | 2688 | 28 | 0.4299 | 0.6803 | 9009.3251 | 14361.0258 | 0.6273 | 5351.7007 | 4 | 19.2143 | 12.8571 | 11310.4282 | 1 | 3050.5976 | -2301.1031 | 3 | 2301.1031 |
| valid_2025_10 | sequence_window_tcn_5fold | 2976 | 31 | 0.6242 | 0.8846 | 7340.9981 | 13921.9943 | 0.5273 | 6580.9962 | 2 | 16.1935 | 15.2581 | 7899.1033 | 3 | 6022.8910 | -558.1052 | -1 | 558.1052 |
| valid_2025_11 | sequence_window_tcn_5fold | 2880 | 30 | 0.2162 | 0.2857 | 6782.0053 | 8007.9217 | 0.8469 | 1225.9164 | 0 | 10.5667 | 7.6000 | 7120.7667 | 0 | 887.1550 | -338.7614 | 0 | 338.7614 |
| valid_2025_12 | sequence_window_tcn_5fold | 2976 | 31 | 0.3099 | 0.5238 | 5277.9750 | 10135.5174 | 0.5207 | 4857.5424 | 3 | 15.1613 | 14.2581 | 5928.2233 | 1 | 4207.2941 | -650.2483 | 2 | 650.2483 |
| valid_2025_jan_feb | sequence_window_tcn_5fold | 5376 | 56 | 0.4709 | 0.8405 | 13183.6917 | 16655.0543 | 0.7916 | 3471.3626 | 0 | 13.8393 | 9.7679 | 14153.4998 | 0 | 2501.5545 | -969.8081 | 0 | 969.8081 |

## Fit Summary
| fold | train_days | valid_days | best_epoch | best_valid_smooth_l1 | train_window_rmse | valid_window_rmse | valid_window_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 240 | 28 | 3.0000 | 0.2526 | 0.7030 | 0.6803 | 0.4299 |
| valid_2025_10 | 268 | 31 | 13.0000 | 0.4261 | 0.6472 | 0.8846 | 0.6242 |
| valid_2025_11 | 299 | 30 | 22.0000 | 0.0756 | 0.6451 | 0.2857 | 0.2162 |
| valid_2025_12 | 329 | 31 | 27.0000 | 0.1633 | 0.5913 | 0.5238 | 0.3099 |
| valid_2025_jan_feb | 304 | 56 | 7.0000 | 0.3085 | 0.6339 | 0.8405 | 0.4709 |
