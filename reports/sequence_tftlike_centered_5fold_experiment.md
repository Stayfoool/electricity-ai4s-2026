# GRU Direct Price Experiment

## Scope

- Model: `sequence_tftlike_centered_5fold`.
- Target mode: `centered`.
- Directly predicts a 96-point daily curve used by dispatch.
- Does not use champion predictions or true historical prices as lag inputs.
- Inputs are official next-day forecast boundary features plus bid_space/business/time features.
- Dispatch still uses the same brute-force legal pair enumeration and current prior.

## Overall

| model | mean_profit | min_profit | mean_rmse | mean_curve_z_rmse | loss_days | mean_regret | mean_charge_gap | mean_discharge_gap | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence_tftlike_centered_5fold | 8427.2573 | 5970.3163 | 1.3739 | 0.8494 | 9 | 4189.0454 | 15.0061 | 10.0409 | -855.1469 | 4 | 855.1469 |

## Fold Summary

| fold | valid_rows | valid_days | mae | rmse | curve_z_rmse | mean_abs_charge_gap | mean_abs_discharge_gap | avg_profit | oracle_avg_profit | oracle_ratio | avg_regret | loss_days | model | target_mode | champion_avg_profit | champion_loss_days | champion_avg_regret | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 2688 | 28 | 0.9967 | 1.3873 | 0.9214 | 15.4643 | 8.5000 | 9253.7078 | 14361.0258 | 0.6444 | 5107.3180 | 1 | sequence_tftlike_centered_5fold | centered | 11310.4282 | 1 | 3050.5976 | -2056.7203 | 0 | 2056.7203 |
| valid_2025_10 | 2976 | 31 | 1.7408 | 2.0757 | 0.9815 | 16.9355 | 13.9677 | 6737.3439 | 13921.9943 | 0.4839 | 7184.6504 | 6 | sequence_tftlike_centered_5fold | centered | 7899.1033 | 3 | 6022.8910 | -1161.7594 | 3 | 1161.7594 |
| valid_2025_11 | 2880 | 30 | 0.4512 | 0.6175 | 0.7498 | 12.8000 | 6.2000 | 6288.5538 | 8007.9217 | 0.7853 | 1719.3679 | 0 | sequence_tftlike_centered_5fold | centered | 7120.7667 | 0 | 887.1550 | -832.2129 | 0 | 832.2129 |
| valid_2025_12 | 2976 | 31 | 0.7856 | 1.1189 | 0.8400 | 14.5806 | 12.3226 | 5970.3163 | 10135.5174 | 0.5890 | 4165.2011 | 1 | sequence_tftlike_centered_5fold | centered | 5928.2233 | 1 | 4207.2941 | 42.0930 | 0 | -42.0930 |
| valid_2025_jan_feb | 5376 | 56 | 1.1870 | 1.6699 | 0.7543 | 15.2500 | 9.2143 | 13886.3647 | 16655.0543 | 0.8338 | 2768.6896 | 1 | sequence_tftlike_centered_5fold | centered | 14153.4998 | 0 | 2501.5545 | -267.1351 | 1 | 267.1351 |

## Fit Summary

| fold | target_mode | train_days | valid_days | best_epoch | best_valid_smooth_l1 | train_price_rmse | valid_price_rmse | valid_price_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | centered | 240 | 28 | 18.0000 | 0.2625 | 0.6601 | 0.7083 | 0.4276 |
| valid_2025_10 | centered | 268 | 31 | 32.0000 | 0.3132 | 0.6468 | 0.7138 | 0.5023 |
| valid_2025_11 | centered | 299 | 30 | 14.0000 | 0.0739 | 0.6734 | 0.2849 | 0.2140 |
| valid_2025_12 | centered | 329 | 31 | 11.0000 | 0.1545 | 0.6567 | 0.4999 | 0.3013 |
| valid_2025_jan_feb | centered | 304 | 56 | 24.0000 | 0.3555 | 0.5942 | 0.8498 | 0.5355 |

## Decision

- Does not beat current champion: mean delta is -855.1/day.
- Keep as deep-learning baseline only, not a submit candidate.

## Artifacts

- `reports/backtest_sequence_tftlike_centered_5fold.csv`
- `reports/backtest_sequence_tftlike_centered_5fold_daily.csv`
- `reports/sequence_tftlike_centered_5fold_overall.csv`
- `reports/predictions_sequence_tftlike_centered_5fold.csv`
- `reports/sequence_tftlike_centered_5fold_fit.csv`
- `reports/sequence_tftlike_centered_5fold_experiment.md`
