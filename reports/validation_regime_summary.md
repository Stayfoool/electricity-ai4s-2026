# Validation Regime Summary

This report re-summarizes daily backtest outputs under multiple validation regimes.
It is intended to reduce overfitting to a single trailing validation split.

## Regimes

- `standard_09_12`: rolling validation on September to December 2025.
- `jan_feb_like`: pseudo-test fold using January-February 2025.
- `winter_11_12_jan_feb`: winter-focused view combining November, December, and Jan-Feb-like.
- `late_winter_12_jan_feb`: stricter late-winter view combining December and Jan-Feb-like.
- `all_5fold`: all available 5-fold validation days.

## standard_09_12

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold | 120 | 7991.3509 | -15925.8974 | 4 | 0.6908 | 3576.3927 | 11045.4592 | 12.6333 | 9.9750 |
| ens_champion_segmented6_prior_5fold_cap_only | 120 | 7720.3443 | -15925.8974 | 5 | 0.6674 | 3847.3993 | 13036.3968 | 13.1667 | 10.9250 |
| ens_champion_segmented6_prior_5fold_holiday_only | 120 | 7572.2950 | -10239.8016 | 5 | 0.6546 | 3995.4487 | 11926.1808 | 13.4500 | 10.5167 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 120 | 7451.4888 | -15989.5246 | 4 | 0.6442 | 4116.2548 | 13003.7233 | 13.1667 | 10.5167 |

## jan_feb_like

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold_holiday_only | 56 | 14370.6631 | 5.7614 | 0 | 0.8628 | 2284.3912 | 8119.4717 | 14.7321 | 9.5714 |
| ens_champion_segmented6_prior_5fold | 56 | 14153.4998 | 8.7837 | 0 | 0.8498 | 2501.5545 | 8728.8520 | 13.7321 | 9.5000 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 56 | 14053.0723 | 8.7837 | 0 | 0.8438 | 2601.9820 | 9275.2604 | 13.6071 | 9.1071 |
| ens_champion_segmented6_prior_5fold_cap_only | 56 | 13644.8040 | 5.7614 | 0 | 0.8193 | 3010.2503 | 11627.8775 | 13.9464 | 10.3036 |

## winter_11_12_jan_feb

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold_holiday_only | 117 | 10221.7204 | -10239.8016 | 1 | 0.8042 | 2488.7217 | 6992.6333 | 13.0769 | 9.3846 |
| ens_champion_segmented6_prior_5fold | 117 | 10170.8882 | -12550.1225 | 1 | 0.8002 | 2539.5540 | 7189.1142 | 12.7265 | 9.4872 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 117 | 10055.0381 | -10239.8016 | 1 | 0.7911 | 2655.4040 | 7189.1142 | 12.4103 | 9.2564 |
| ens_champion_segmented6_prior_5fold_cap_only | 117 | 9872.5342 | -7083.8907 | 2 | 0.7767 | 2837.9080 | 8038.1561 | 13.0000 | 10.3846 |

## late_winter_12_jan_feb

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold_holiday_only | 87 | 11356.4751 | -10239.8016 | 1 | 0.7924 | 2975.5259 | 8604.0822 | 14.5862 | 10.9885 |
| ens_champion_segmented6_prior_5fold | 87 | 11222.6542 | -12550.1225 | 1 | 0.7830 | 3109.3467 | 9367.5085 | 14.1494 | 11.2759 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 87 | 11183.7611 | -10239.8016 | 1 | 0.7803 | 3148.2398 | 11279.0030 | 13.8966 | 10.8966 |
| ens_champion_segmented6_prior_5fold_cap_only | 87 | 10957.8537 | -7083.8907 | 1 | 0.7646 | 3374.1472 | 12261.3961 | 14.1379 | 11.7701 |

## all_5fold

| model | days | mean_profit | worst_day_profit | loss_days | oracle_ratio | mean_regret | p90_regret | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold | 176 | 9952.0347 | -15925.8974 | 4 | 0.7547 | 3234.3987 | 9990.9094 | 12.9830 | 9.8239 |
| ens_champion_segmented6_prior_5fold_holiday_only | 176 | 9735.4121 | -10239.8016 | 5 | 0.7383 | 3451.0213 | 11377.1067 | 13.8580 | 10.2159 |
| ens_champion_segmented6_prior_5fold_cap_only | 176 | 9605.3997 | -15925.8974 | 5 | 0.7284 | 3581.0337 | 12306.8008 | 13.4148 | 10.7273 |
| ens_champion_segmented6_prior_5fold_cap_holiday | 176 | 9551.9927 | -15989.5246 | 4 | 0.7244 | 3634.4407 | 12762.6375 | 13.3068 | 10.0682 |

## Artifacts

- `reports/validation_regime_summary.csv`

## Readout

- The current champion remains best on the standard September-December rolling view.
- `holiday_only` is better on `jan_feb_like`, `winter_11_12_jan_feb`, and `late_winter_12_jan_feb`.
- This confirms the validation concern: a strategy can look worse on trailing autumn folds while looking better on winter-like folds.
- Promotion should therefore not use one scalar mean only. Future experiments should report both standard rolling and winter-focused regimes.
- For final January-February 2026 submission candidates, winter-focused lift should be considered, but only if the strategy does not materially increase worst-day risk.

## Suggested Promotion Rule

A candidate can be considered for final submission only if one of these is true:

- It improves `standard_09_12` mean profit without worsening `jan_feb_like` and worst-day risk.
- It improves `jan_feb_like` and winter regimes substantially, while the `standard_09_12` loss is explainable and acceptable for a winter-only submit candidate.
- It reduces loss days or worst-day loss enough to justify a small mean-profit tradeoff.

This report should be regenerated after every material experiment with:

```bash
python scripts/summarize_validation_regimes.py --pattern 'backtest_*5fold*_daily.csv'
```
