# GRU Direct Price Experiment

## Scope

- Model: `sequence_tftlike_direct_price_5fold`.
- Target mode: `absolute`.
- Directly predicts a 96-point daily curve used by dispatch.
- Does not use champion predictions or true historical prices as lag inputs.
- Inputs are official next-day forecast boundary features plus bid_space/business/time features.
- Dispatch still uses the same brute-force legal pair enumeration and current prior.

## Overall

| model | mean_profit | min_profit | mean_rmse | mean_curve_z_rmse | loss_days | mean_regret | mean_charge_gap | mean_discharge_gap | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence_tftlike_direct_price_5fold | 8563.3213 | 5911.3821 | 0.7564 | 0.8262 | 10 | 4052.9814 | 12.8703 | 10.3318 | -719.0829 | 5 | 719.0829 |

## Fold Summary

| fold | valid_rows | valid_days | mae | rmse | curve_z_rmse | mean_abs_charge_gap | mean_abs_discharge_gap | avg_profit | oracle_avg_profit | oracle_ratio | avg_regret | loss_days | model | target_mode | champion_avg_profit | champion_loss_days | champion_avg_regret | profit_delta_vs_champion | loss_day_delta_vs_champion | regret_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 2688 | 28 | 0.4539 | 0.7872 | 0.8805 | 15.3214 | 8.2143 | 9662.1023 | 14361.0258 | 0.6728 | 4698.9235 | 2 | sequence_tftlike_direct_price_5fold | absolute | 11310.4282 | 1 | 3050.5976 | -1648.3258 | 1 | 1648.3258 |
| valid_2025_10 | 2976 | 31 | 0.7165 | 1.0818 | 1.0130 | 15.9677 | 14.8387 | 5911.3821 | 13921.9943 | 0.4246 | 8010.6121 | 6 | sequence_tftlike_direct_price_5fold | absolute | 7899.1033 | 3 | 6022.8910 | -1987.7211 | 3 | 1987.7211 |
| valid_2025_11 | 2880 | 30 | 0.2432 | 0.3181 | 0.6776 | 7.8667 | 4.5333 | 7140.4535 | 8007.9217 | 0.8917 | 867.4682 | 1 | sequence_tftlike_direct_price_5fold | absolute | 7120.7667 | 0 | 887.1550 | 19.6868 | 1 | -19.6868 |
| valid_2025_12 | 2976 | 31 | 0.3492 | 0.5810 | 0.8322 | 13.8387 | 13.3226 | 6036.5718 | 10135.5174 | 0.5956 | 4098.9456 | 1 | sequence_tftlike_direct_price_5fold | absolute | 5928.2233 | 1 | 4207.2941 | 108.3485 | 0 | -108.3485 |
| valid_2025_jan_feb | 5376 | 56 | 0.5208 | 1.0137 | 0.7278 | 11.3571 | 10.7500 | 14066.0969 | 16655.0543 | 0.8446 | 2588.9574 | 0 | sequence_tftlike_direct_price_5fold | absolute | 14153.4998 | 0 | 2501.5545 | -87.4030 | 0 | 87.4030 |

## Fit Summary

| fold | target_mode | train_days | valid_days | best_epoch | best_valid_smooth_l1 | train_price_rmse | valid_price_rmse | valid_price_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | absolute | 240 | 28 | 26.0000 | 0.2742 | 0.7854 | 0.7872 | 0.4539 |
| valid_2025_10 | absolute | 268 | 31 | 5.0000 | 0.5203 | 0.8305 | 1.0818 | 0.7165 |
| valid_2025_11 | absolute | 299 | 30 | 35.0000 | 0.0912 | 0.8014 | 0.3181 | 0.2432 |
| valid_2025_12 | absolute | 329 | 31 | 29.0000 | 0.1891 | 0.7679 | 0.5810 | 0.3492 |
| valid_2025_jan_feb | absolute | 304 | 56 | 4.0000 | 0.3630 | 0.7757 | 1.0137 | 0.5208 |

## Decision

- Does not beat current champion: mean delta is -719.1/day.
- Keep as deep-learning baseline only, not a submit candidate.

## Artifacts

- `reports/backtest_sequence_tftlike_direct_price_5fold.csv`
- `reports/backtest_sequence_tftlike_direct_price_5fold_daily.csv`
- `reports/sequence_tftlike_direct_price_5fold_overall.csv`
- `reports/predictions_sequence_tftlike_direct_price_5fold.csv`
- `reports/sequence_tftlike_direct_price_5fold_fit.csv`
- `reports/sequence_tftlike_direct_price_5fold_experiment.md`
