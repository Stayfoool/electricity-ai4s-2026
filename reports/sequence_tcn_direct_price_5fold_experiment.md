# TCN Direct Price Experiment

## Scope

- Model: `sequence_tcn_direct_price_5fold`.
- Target mode: `absolute`.
- Directly predicts a 96-point daily curve used by dispatch.
- Does not use champion predictions or true historical prices as lag inputs.
- Inputs are official next-day forecast boundary features plus bid_space/business/time features.
- Dispatch still uses the same brute-force legal pair enumeration and current prior.

## Overall

| model | mean_profit | min_profit | mean_rmse | mean_curve_z_rmse | loss_days | mean_regret | mean_charge_gap | mean_discharge_gap | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence_tcn_direct_price_5fold | 8778.8719 | 6014.9839 | 0.7656 | 0.8251 | 9 | 3837.4308 | 12.8922 | 10.3890 | -503.5324 | 4 | 503.5324 |

## Fold Summary

| fold | valid_rows | valid_days | mae | rmse | curve_z_rmse | mean_abs_charge_gap | mean_abs_discharge_gap | avg_profit | oracle_avg_profit | oracle_ratio | avg_regret | loss_days | model | target_mode | champion_avg_profit | champion_loss_days | champion_avg_regret | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 2688 | 28 | 0.4659 | 0.8426 | 0.9230 | 14.7857 | 8.3929 | 9226.0752 | 14361.0258 | 0.6424 | 5134.9506 | 2 | sequence_tcn_direct_price_5fold | absolute | 11310.4282 | 1 | 3050.5976 | -2084.3529 | 1 | 2084.3529 |
| valid_2025_10 | 2976 | 31 | 0.7151 | 1.0633 | 0.9614 | 15.0323 | 14.4839 | 7093.4632 | 13921.9943 | 0.5095 | 6828.5311 | 6 | sequence_tcn_direct_price_5fold | absolute | 7899.1033 | 3 | 6022.8910 | -805.6401 | 3 | 805.6401 |
| valid_2025_11 | 2880 | 30 | 0.2387 | 0.3045 | 0.6713 | 8.6000 | 5.2667 | 7095.8040 | 8007.9217 | 0.8861 | 912.1176 | 0 | sequence_tcn_direct_price_5fold | absolute | 7120.7667 | 0 | 887.1550 | -24.9627 | 0 | 24.9627 |
| valid_2025_12 | 2976 | 31 | 0.3499 | 0.6066 | 0.8516 | 15.0968 | 14.5161 | 6014.9839 | 10135.5174 | 0.5935 | 4120.5335 | 1 | sequence_tcn_direct_price_5fold | absolute | 5928.2233 | 1 | 4207.2941 | 86.7606 | 0 | -86.7606 |
| valid_2025_jan_feb | 5376 | 56 | 0.5150 | 1.0107 | 0.7183 | 10.9464 | 9.2857 | 14464.0332 | 16655.0543 | 0.8684 | 2191.0211 | 0 | sequence_tcn_direct_price_5fold | absolute | 14153.4998 | 0 | 2501.5545 | 310.5334 | 0 | -310.5334 |

## Fit Summary

| fold | target_mode | train_days | valid_days | best_epoch | best_valid_smooth_l1 | train_price_rmse | valid_price_rmse | valid_price_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | absolute | 240 | 28 | 5.0000 | 0.2909 | 0.8381 | 0.8426 | 0.4659 |
| valid_2025_10 | absolute | 268 | 31 | 13.0000 | 0.5182 | 0.7827 | 1.0633 | 0.7151 |
| valid_2025_11 | absolute | 299 | 30 | 20.0000 | 0.0836 | 0.8204 | 0.3045 | 0.2387 |
| valid_2025_12 | absolute | 329 | 31 | 16.0000 | 0.1950 | 0.7748 | 0.6066 | 0.3499 |
| valid_2025_jan_feb | absolute | 304 | 56 | 7.0000 | 0.3587 | 0.7507 | 1.0107 | 0.5150 |

## Decision

- Does not beat current champion: mean delta is -503.5/day.
- Keep as deep-learning baseline only, not a submit candidate.

## Artifacts

- `reports/backtest_sequence_tcn_direct_price_5fold.csv`
- `reports/backtest_sequence_tcn_direct_price_5fold_daily.csv`
- `reports/sequence_tcn_direct_price_5fold_overall.csv`
- `reports/predictions_sequence_tcn_direct_price_5fold.csv`
- `reports/sequence_tcn_direct_price_5fold_fit.csv`
- `reports/sequence_tcn_direct_price_5fold_experiment.md`
