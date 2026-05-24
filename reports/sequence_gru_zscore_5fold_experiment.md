# GRU Direct Price Experiment

## Scope

- Model: `sequence_gru_zscore_5fold`.
- Target mode: `zscore`.
- Directly predicts a 96-point daily curve used by dispatch.
- Does not use champion predictions or true historical prices as lag inputs.
- Inputs are official next-day forecast boundary features plus bid_space/business/time features.
- Dispatch still uses the same brute-force legal pair enumeration and current prior.

## Overall

| model | mean_profit | min_profit | mean_rmse | mean_curve_z_rmse | loss_days | mean_regret | mean_charge_gap | mean_discharge_gap | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence_gru_zscore_5fold | 8421.3842 | 6013.6918 | 1.4071 | 0.8466 | 10 | 4194.9185 | 13.9217 | 10.9541 | -861.0201 | 5 | 861.0201 |

## Fold Summary

| fold | valid_rows | valid_days | mae | rmse | curve_z_rmse | mean_abs_charge_gap | mean_abs_discharge_gap | avg_profit | oracle_avg_profit | oracle_ratio | avg_regret | loss_days | model | target_mode | champion_avg_profit | champion_loss_days | champion_avg_regret | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 2688 | 28 | 1.0896 | 1.4510 | 0.9251 | 13.5714 | 8.5000 | 9053.8984 | 14361.0258 | 0.6304 | 5307.1274 | 2 | sequence_gru_zscore_5fold | zscore | 11310.4282 | 1 | 3050.5976 | -2256.5297 | 1 | 2256.5297 |
| valid_2025_10 | 2976 | 31 | 1.8019 | 2.1378 | 0.9991 | 19.6129 | 15.1935 | 6013.6918 | 13921.9943 | 0.4320 | 7908.3024 | 6 | sequence_gru_zscore_5fold | zscore | 7899.1033 | 3 | 6022.8910 | -1885.4114 | 3 | 1885.4114 |
| valid_2025_11 | 2880 | 30 | 0.5578 | 0.6887 | 0.7271 | 11.7667 | 9.1000 | 6453.9448 | 8007.9217 | 0.8059 | 1553.9769 | 1 | sequence_gru_zscore_5fold | zscore | 7120.7667 | 0 | 887.1550 | -666.8219 | 1 | 666.8219 |
| valid_2025_12 | 2976 | 31 | 0.8314 | 1.1414 | 0.8472 | 15.0323 | 12.5484 | 6082.3723 | 10135.5174 | 0.6001 | 4053.1451 | 1 | sequence_gru_zscore_5fold | zscore | 5928.2233 | 1 | 4207.2941 | 154.1490 | 0 | -154.1490 |
| valid_2025_jan_feb | 5376 | 56 | 1.1733 | 1.6168 | 0.7345 | 9.6250 | 9.4286 | 14503.0134 | 16655.0543 | 0.8708 | 2152.0408 | 0 | sequence_gru_zscore_5fold | zscore | 14153.4998 | 0 | 2501.5545 | 349.5136 | 0 | -349.5136 |

## Fit Summary

| fold | target_mode | train_days | valid_days | best_epoch | best_valid_smooth_l1 | train_price_rmse | valid_price_rmse | valid_price_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | zscore | 240 | 28 | 14.0000 | 0.3903 | 0.7705 | 0.8273 | 0.5895 |
| valid_2025_10 | zscore | 268 | 31 | 26.0000 | 0.4353 | 0.7584 | 0.8525 | 0.6422 |
| valid_2025_11 | zscore | 299 | 30 | 27.0000 | 0.2776 | 0.7634 | 0.6717 | 0.4616 |
| valid_2025_12 | zscore | 329 | 31 | 17.0000 | 0.3380 | 0.7608 | 0.7517 | 0.5289 |
| valid_2025_jan_feb | zscore | 304 | 56 | 11.0000 | 0.3541 | 0.7817 | 0.7200 | 0.5559 |

## Decision

- Does not beat current champion: mean delta is -861.0/day.
- Keep as deep-learning baseline only, not a submit candidate.

## Artifacts

- `reports/backtest_sequence_gru_zscore_5fold.csv`
- `reports/backtest_sequence_gru_zscore_5fold_daily.csv`
- `reports/sequence_gru_zscore_5fold_overall.csv`
- `reports/predictions_sequence_gru_zscore_5fold.csv`
- `reports/sequence_gru_zscore_5fold_fit.csv`
- `reports/sequence_gru_zscore_5fold_experiment.md`
