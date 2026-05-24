# Window Deep Experiment

- Model: `sequence_window_tftlike_5fold`.
- Input: daily 96-point feature curve.
- Target: 89 rolling 2-hour window average prices.
- Dispatch: enumerate legal charge/discharge window pairs using predicted window scores.

## Overall
| model | mean_profit | min_profit | mean_window_rmse | loss_days | mean_regret | mean_charge_gap | mean_discharge_gap | profit_delta_vs_champion | loss_day_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence_window_tftlike_5fold | 8429.2798 | 5785.2513 | 0.6311 | 10 | 4187.0229 | 13.2361 | 10.0174 | -853.1245 | 5 |

## Fold Summary
| fold | model | valid_rows | valid_days | window_mae | window_rmse | avg_profit | oracle_avg_profit | oracle_ratio | avg_regret | loss_days | mean_abs_charge_gap | mean_abs_discharge_gap | champion_avg_profit | champion_loss_days | champion_avg_regret | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | sequence_window_tftlike_5fold | 2688 | 28 | 0.3895 | 0.6401 | 9705.6544 | 14361.0258 | 0.6758 | 4655.3714 | 1 | 15.2143 | 7.5357 | 11310.4282 | 1 | 3050.5976 | -1604.7738 | 0 | 1604.7738 |
| valid_2025_10 | sequence_window_tftlike_5fold | 2976 | 31 | 0.6167 | 0.8780 | 6566.1227 | 13921.9943 | 0.4716 | 7355.8716 | 6 | 17.9032 | 14.0323 | 7899.1033 | 3 | 6022.8910 | -1332.9806 | 3 | 1332.9806 |
| valid_2025_11 | sequence_window_tftlike_5fold | 2880 | 30 | 0.2102 | 0.2876 | 6992.9366 | 8007.9217 | 0.8733 | 1014.9851 | 1 | 8.1000 | 4.1667 | 7120.7667 | 0 | 887.1550 | -127.8301 | 1 | 127.8301 |
| valid_2025_12 | sequence_window_tftlike_5fold | 2976 | 31 | 0.3217 | 0.5032 | 5785.2513 | 10135.5174 | 0.5708 | 4350.2661 | 2 | 14.6774 | 13.7097 | 5928.2233 | 1 | 4207.2941 | -142.9720 | 1 | 142.9720 |
| valid_2025_jan_feb | sequence_window_tftlike_5fold | 5376 | 56 | 0.4765 | 0.8465 | 13096.4338 | 16655.0543 | 0.7863 | 3558.6204 | 0 | 10.2857 | 10.6429 | 14153.4998 | 0 | 2501.5545 | -1057.0660 | 0 | 1057.0660 |

## Fit Summary
| fold | train_days | valid_days | best_epoch | best_valid_smooth_l1 | train_window_rmse | valid_window_rmse | valid_window_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 240 | 28 | 13.0000 | 0.2226 | 0.6806 | 0.6401 | 0.3895 |
| valid_2025_10 | 268 | 31 | 5.0000 | 0.4147 | 0.7160 | 0.8780 | 0.6167 |
| valid_2025_11 | 299 | 30 | 24.0000 | 0.0736 | 0.6474 | 0.2876 | 0.2102 |
| valid_2025_12 | 329 | 31 | 29.0000 | 0.1673 | 0.6131 | 0.5032 | 0.3217 |
| valid_2025_jan_feb | 304 | 56 | 5.0000 | 0.3127 | 0.6687 | 0.8465 | 0.4765 |
