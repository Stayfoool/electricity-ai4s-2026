# GRU Direct Price Experiment

## Scope

- Model: `sequence_gru_centered_5fold`.
- Target mode: `centered`.
- Directly predicts a 96-point daily curve used by dispatch.
- Does not use champion predictions or true historical prices as lag inputs.
- Inputs are official next-day forecast boundary features plus bid_space/business/time features.
- Dispatch still uses the same brute-force legal pair enumeration and current prior.

## Overall

| model | mean_profit | min_profit | mean_rmse | mean_curve_z_rmse | loss_days | mean_regret | mean_charge_gap | mean_discharge_gap | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence_gru_centered_5fold | 8834.3693 | 6174.7925 | 1.3517 | 0.8405 | 5 | 3781.9334 | 14.0556 | 10.1365 | -448.0350 | 0 | 448.0350 |

## Fold Summary

| fold | valid_rows | valid_days | mae | rmse | curve_z_rmse | mean_abs_charge_gap | mean_abs_discharge_gap | avg_profit | oracle_avg_profit | oracle_ratio | avg_regret | loss_days | model | target_mode | champion_avg_profit | champion_loss_days | champion_avg_regret | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 2688 | 28 | 0.9792 | 1.3682 | 0.9032 | 17.1786 | 9.7500 | 9819.5810 | 14361.0258 | 0.6838 | 4541.4448 | 0 | sequence_gru_centered_5fold | centered | 11310.4282 | 1 | 3050.5976 | -1490.8472 | -1 | 1490.8472 |
| valid_2025_10 | 2976 | 31 | 1.7319 | 2.0765 | 0.9886 | 17.9355 | 14.6129 | 7078.3113 | 13921.9943 | 0.5084 | 6843.6830 | 4 | sequence_gru_centered_5fold | centered | 7899.1033 | 3 | 6022.8910 | -820.7920 | 1 | 820.7920 |
| valid_2025_11 | 2880 | 30 | 0.4173 | 0.5725 | 0.7359 | 8.2667 | 4.3000 | 6938.8376 | 8007.9217 | 0.8665 | 1069.0841 | 0 | sequence_gru_centered_5fold | centered | 7120.7667 | 0 | 887.1550 | -181.9291 | 0 | 181.9291 |
| valid_2025_12 | 2976 | 31 | 0.7631 | 1.1012 | 0.8275 | 15.2903 | 12.4839 | 6174.7925 | 10135.5174 | 0.6092 | 3960.7249 | 1 | sequence_gru_centered_5fold | centered | 5928.2233 | 1 | 4207.2941 | 246.5692 | 0 | -246.5692 |
| valid_2025_jan_feb | 5376 | 56 | 1.1514 | 1.6400 | 0.7473 | 11.6071 | 9.5357 | 14160.3240 | 16655.0543 | 0.8502 | 2494.7303 | 0 | sequence_gru_centered_5fold | centered | 14153.4998 | 0 | 2501.5545 | 6.8242 | 0 | -6.8242 |

## Fit Summary

| fold | target_mode | train_days | valid_days | best_epoch | best_valid_smooth_l1 | train_price_rmse | valid_price_rmse | valid_price_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | centered | 240 | 28 | 31.0000 | 0.2663 | 0.6500 | 0.7093 | 0.4367 |
| valid_2025_10 | centered | 268 | 31 | 19.0000 | 0.3246 | 0.6692 | 0.7268 | 0.5131 |
| valid_2025_11 | centered | 299 | 30 | 30.0000 | 0.0666 | 0.6607 | 0.2696 | 0.2019 |
| valid_2025_12 | centered | 329 | 31 | 29.0000 | 0.1501 | 0.6438 | 0.4933 | 0.2971 |
| valid_2025_jan_feb | centered | 304 | 56 | 12.0000 | 0.3706 | 0.6038 | 0.8675 | 0.5526 |

## Decision

- Does not beat current champion: mean delta is -448.0/day.
- Keep as deep-learning baseline only, not a submit candidate.

## Artifacts

- `reports/backtest_sequence_gru_centered_5fold.csv`
- `reports/backtest_sequence_gru_centered_5fold_daily.csv`
- `reports/sequence_gru_centered_5fold_overall.csv`
- `reports/predictions_sequence_gru_centered_5fold.csv`
- `reports/sequence_gru_centered_5fold_fit.csv`
- `reports/sequence_gru_centered_5fold_experiment.md`
