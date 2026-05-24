# GRU Direct Price Experiment

## Scope

- Model: `sequence_gru_direct_price_5fold`.
- Directly predicts the 96-point daily electricity price curve.
- Does not use champion predictions or true historical prices as lag inputs.
- Inputs are official next-day forecast boundary features plus bid_space/business/time features.
- Dispatch still uses the same brute-force legal pair enumeration and current prior.

## Overall

| model | mean_profit | min_profit | mean_rmse | mean_curve_z_rmse | loss_days | mean_regret | mean_charge_gap | mean_discharge_gap | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence_gru_direct_price_5fold | 8652.0747 | 5590.3967 | 0.7820 | 0.8402 | 10 | 3964.2280 | 12.9925 | 10.9561 | -630.3295 | 5 | 630.3295 |

## Fold Summary

| fold | valid_rows | valid_days | mae | rmse | curve_z_rmse | mean_abs_charge_gap | mean_abs_discharge_gap | avg_profit | oracle_avg_profit | oracle_ratio | avg_regret | loss_days | champion_avg_profit | champion_loss_days | champion_avg_regret | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 2688 | 28 | 0.4591 | 0.8426 | 0.9155 | 13.0000 | 8.4286 | 9948.7143 | 14361.0258 | 0.6928 | 4412.3115 | 2 | 11310.4282 | 1 | 3050.5976 | -1361.7139 | 1 | 1361.7139 |
| valid_2025_10 | 2976 | 31 | 0.7430 | 1.1045 | 0.9749 | 15.4839 | 14.7742 | 6564.6253 | 13921.9943 | 0.4715 | 7357.3690 | 5 | 7899.1033 | 3 | 6022.8910 | -1334.4779 | 2 | 1334.4779 |
| valid_2025_11 | 2880 | 30 | 0.2379 | 0.3185 | 0.6999 | 9.5333 | 7.3333 | 6617.1304 | 8007.9217 | 0.8263 | 1390.7913 | 1 | 7120.7667 | 0 | 887.1550 | -503.6363 | 1 | 503.6363 |
| valid_2025_12 | 2976 | 31 | 0.3577 | 0.6225 | 0.8520 | 15.6774 | 14.3871 | 5590.3967 | 10135.5174 | 0.5516 | 4545.1207 | 1 | 5928.2233 | 1 | 4207.2941 | -337.8267 | 0 | 337.8267 |
| valid_2025_jan_feb | 5376 | 56 | 0.5371 | 1.0222 | 0.7584 | 11.2679 | 9.8571 | 14539.5070 | 16655.0543 | 0.8730 | 2115.5473 | 1 | 14153.4998 | 0 | 2501.5545 | 386.0071 | 1 | -386.0071 |

## Fit Summary

| fold | train_days | valid_days | best_epoch | best_valid_smooth_l1 | train_price_rmse | valid_price_rmse | valid_price_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 240 | 28 | 8.0000 | 0.2847 | 0.8350 | 0.8426 | 0.4591 |
| valid_2025_10 | 268 | 31 | 19.0000 | 0.5458 | 0.7828 | 1.1045 | 0.7430 |
| valid_2025_11 | 299 | 30 | 32.0000 | 0.0879 | 0.8464 | 0.3185 | 0.2379 |
| valid_2025_12 | 329 | 31 | 16.0000 | 0.2000 | 0.8140 | 0.6225 | 0.3577 |
| valid_2025_jan_feb | 304 | 56 | 7.0000 | 0.3740 | 0.7665 | 1.0222 | 0.5371 |

## Decision

- Does not beat current champion: mean delta is -630.3/day.
- Keep as deep-learning baseline only, not a submit candidate.

## Artifacts

- `reports/backtest_sequence_gru_direct_price_5fold.csv`
- `reports/backtest_sequence_gru_direct_price_5fold_daily.csv`
- `reports/sequence_gru_direct_price_5fold_overall.csv`
- `reports/predictions_sequence_gru_direct_price_5fold.csv`
- `reports/sequence_gru_direct_price_5fold_fit.csv`
- `reports/sequence_gru_direct_price_5fold_experiment.md`
