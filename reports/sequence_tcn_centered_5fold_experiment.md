# TCN Direct Price Experiment

## Scope

- Model: `sequence_tcn_centered_5fold`.
- Target mode: `centered`.
- Directly predicts a 96-point daily curve used by dispatch.
- Does not use champion predictions or true historical prices as lag inputs.
- Inputs are official next-day forecast boundary features plus bid_space/business/time features.
- Dispatch still uses the same brute-force legal pair enumeration and current prior.

## Overall

| model | mean_profit | min_profit | mean_rmse | mean_curve_z_rmse | loss_days | mean_regret | mean_charge_gap | mean_discharge_gap | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence_tcn_centered_5fold | 8684.5134 | 6082.4167 | 1.3382 | 0.8260 | 9 | 3931.7893 | 14.3510 | 10.1224 | -597.8909 | 4 | 597.8909 |

## Fold Summary

| fold | valid_rows | valid_days | mae | rmse | curve_z_rmse | mean_abs_charge_gap | mean_abs_discharge_gap | avg_profit | oracle_avg_profit | oracle_ratio | avg_regret | loss_days | model | target_mode | champion_avg_profit | champion_loss_days | champion_avg_regret | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 2688 | 28 | 0.9793 | 1.3573 | 0.8792 | 18.2143 | 8.3214 | 10192.0372 | 14361.0258 | 0.7097 | 4168.9886 | 1 | sequence_tcn_centered_5fold | centered | 11310.4282 | 1 | 3050.5976 | -1118.3909 | 0 | 1118.3909 |
| valid_2025_10 | 2976 | 31 | 1.7342 | 2.0648 | 0.9791 | 17.0645 | 14.6129 | 6741.9486 | 13921.9943 | 0.4843 | 7180.0457 | 7 | sequence_tcn_centered_5fold | centered | 7899.1033 | 3 | 6022.8910 | -1157.1547 | 4 | 1157.1547 |
| valid_2025_11 | 2880 | 30 | 0.4245 | 0.5806 | 0.7174 | 9.1667 | 5.1667 | 6822.1115 | 8007.9217 | 0.8519 | 1185.8102 | 0 | sequence_tcn_centered_5fold | centered | 7120.7667 | 0 | 887.1550 | -298.6552 | 0 | 298.6552 |
| valid_2025_12 | 2976 | 31 | 0.7514 | 1.0897 | 0.8123 | 14.6129 | 13.0645 | 6082.4167 | 10135.5174 | 0.6001 | 4053.1008 | 1 | sequence_tcn_centered_5fold | centered | 5928.2233 | 1 | 4207.2941 | 154.1933 | 0 | -154.1933 |
| valid_2025_jan_feb | 5376 | 56 | 1.1276 | 1.5987 | 0.7423 | 12.6964 | 9.4464 | 13584.0531 | 16655.0543 | 0.8156 | 3071.0012 | 0 | sequence_tcn_centered_5fold | centered | 14153.4998 | 0 | 2501.5545 | -569.4467 | 0 | 569.4467 |

## Fit Summary

| fold | target_mode | train_days | valid_days | best_epoch | best_valid_smooth_l1 | train_price_rmse | valid_price_rmse | valid_price_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | centered | 240 | 28 | 91.0000 | 0.2617 | 0.5234 | 0.6986 | 0.4299 |
| valid_2025_10 | centered | 268 | 31 | 21.0000 | 0.3088 | 0.6439 | 0.7090 | 0.4941 |
| valid_2025_11 | centered | 299 | 30 | 27.0000 | 0.0666 | 0.6410 | 0.2696 | 0.1988 |
| valid_2025_12 | centered | 329 | 31 | 26.0000 | 0.1513 | 0.6247 | 0.4928 | 0.2981 |
| valid_2025_jan_feb | centered | 304 | 56 | 6.0000 | 0.3585 | 0.6058 | 0.8393 | 0.5375 |

## Decision

- Does not beat current champion: mean delta is -597.9/day.
- Keep as deep-learning baseline only, not a submit candidate.

## Artifacts

- `reports/backtest_sequence_tcn_centered_5fold.csv`
- `reports/backtest_sequence_tcn_centered_5fold_daily.csv`
- `reports/sequence_tcn_centered_5fold_overall.csv`
- `reports/predictions_sequence_tcn_centered_5fold.csv`
- `reports/sequence_tcn_centered_5fold_fit.csv`
- `reports/sequence_tcn_centered_5fold_experiment.md`
