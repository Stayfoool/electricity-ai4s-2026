# Window Deep Experiment

- Model: `sequence_window_gru_5fold`.
- Input: daily 96-point feature curve.
- Target: 89 rolling 2-hour window average prices.
- Dispatch: enumerate legal charge/discharge window pairs using predicted window scores.

## Overall
| model | mean_profit | min_profit | mean_window_rmse | loss_days | mean_regret | mean_charge_gap | mean_discharge_gap | profit_delta_vs_champion | loss_day_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence_window_gru_5fold | 9098.2980 | 5891.1045 | 0.6669 | 6 | 3518.0047 | 12.3909 | 10.0249 | -184.1063 | 1 |

## Fold Summary
| fold | model | valid_rows | valid_days | window_mae | window_rmse | avg_profit | oracle_avg_profit | oracle_ratio | avg_regret | loss_days | mean_abs_charge_gap | mean_abs_discharge_gap | champion_avg_profit | champion_loss_days | champion_avg_regret | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | sequence_window_gru_5fold | 2688 | 28 | 0.4080 | 0.6880 | 10181.8905 | 14361.0258 | 0.7090 | 4179.1353 | 2 | 12.7500 | 7.2500 | 11310.4282 | 1 | 3050.5976 | -1128.5377 | 1 | 1128.5377 |
| valid_2025_10 | sequence_window_gru_5fold | 2976 | 31 | 0.6841 | 0.9909 | 8165.3567 | 13921.9943 | 0.5865 | 5756.6376 | 3 | 14.3871 | 14.3548 | 7899.1033 | 3 | 6022.8910 | 266.2534 | 0 | -266.2534 |
| valid_2025_11 | sequence_window_gru_5fold | 2880 | 30 | 0.2020 | 0.2766 | 7023.6147 | 8007.9217 | 0.8771 | 984.3070 | 0 | 8.2000 | 4.9667 | 7120.7667 | 0 | 887.1550 | -97.1520 | 0 | 97.1520 |
| valid_2025_12 | sequence_window_gru_5fold | 2976 | 31 | 0.3155 | 0.5268 | 5891.1045 | 10135.5174 | 0.5812 | 4244.4129 | 1 | 14.9032 | 13.8387 | 5928.2233 | 1 | 4207.2941 | -37.1188 | 0 | 37.1188 |
| valid_2025_jan_feb | sequence_window_gru_5fold | 5376 | 56 | 0.4748 | 0.8524 | 14229.5236 | 16655.0543 | 0.8544 | 2425.5307 | 0 | 11.7143 | 9.7143 | 14153.4998 | 0 | 2501.5545 | 76.0238 | 0 | -76.0238 |

## Fit Summary
| fold | train_days | valid_days | best_epoch | best_valid_smooth_l1 | train_window_rmse | valid_window_rmse | valid_window_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 240 | 28 | 36.0000 | 0.2431 | 0.5920 | 0.6880 | 0.4080 |
| valid_2025_10 | 268 | 31 | 6.0000 | 0.4894 | 0.6981 | 0.9909 | 0.6841 |
| valid_2025_11 | 299 | 30 | 20.0000 | 0.0693 | 0.6635 | 0.2766 | 0.2020 |
| valid_2025_12 | 329 | 31 | 19.0000 | 0.1663 | 0.6455 | 0.5268 | 0.3155 |
| valid_2025_jan_feb | 304 | 56 | 13.0000 | 0.3144 | 0.6326 | 0.8524 | 0.4748 |
