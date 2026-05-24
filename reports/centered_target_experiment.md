# Centered Target Champion 5fold Experiment

## Purpose

Test whether training the current champion ensemble on daily centered prices improves the relative intraday curve shape and therefore charge/discharge window selection.

Only the target mode changed from `absolute` to `centered`. The raw feature columns, folds, ensemble weights, LightGBM parameters, and dispatch prior were kept the same as `configs/ensemble_champion_segmented6_prior_5fold.yaml`.

## Result

| metric | champion | centered | delta centered - champion |
| --- | ---: | ---: | ---: |
| fold_mean_profit | 9282.404 | 8058.914 | -1223.490 |
| all_5fold_day_weighted_profit | 9952.035 | 8755.415 | -1196.619 |
| standard_09_12_profit | 7991.351 | 6729.910 | -1261.441 |
| Jan-Feb-like profit | 14153.500 | 13095.783 | -1057.717 |
| test_like_weighted_profit | 10152.642 | 8990.449 | -1162.193 |
| loss_days | 5 | 11 | +6 |
| all_5fold_p90_regret | 9990.909 | 13389.113 | +3398.203 |

## Fold Detail

| fold | champion_profit | centered_profit | delta_profit | champion_loss_days | centered_loss_days | centered_regret_delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid_2025_09 | 11310.428 | 9839.472 | -1470.957 | 1 | 2 | +1470.957 |
| valid_2025_10 | 7899.103 | 5676.825 | -2222.278 | 3 | 6 | +2222.278 |
| valid_2025_11 | 7120.767 | 6054.775 | -1065.992 | 0 | 2 | +1065.992 |
| valid_2025_12 | 5928.223 | 5627.717 | -300.506 | 1 | 1 | +300.506 |
| valid_2025_jan_feb | 14153.500 | 13095.783 | -1057.717 | 0 | 0 | +1057.717 |

## Decision

Reject `shape_centered_champion_5fold` as a submit candidate and as a mainline target direction.

Interpretation: daily-centered training did not translate into better legal 2-hour charge/discharge window ordering. It worsened every validation fold, doubled loss days, and increased high-regret tail risk.

## Artifacts

- `configs/ensemble_champion_segmented6_prior_centered_5fold.yaml`
- `reports/backtest_ens_champion_segmented6_prior_centered_5fold.csv`
- `reports/backtest_ens_champion_segmented6_prior_centered_5fold_daily.csv`
