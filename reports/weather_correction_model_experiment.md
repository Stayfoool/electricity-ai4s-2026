# Weather Correction Feature Experiment

## Goal

Use NWP weather forecasts to correct renewable forecast error, then feed the corrected renewable / bid-space features into the current champion ensemble.

The tested config is `configs/ensemble_champion_segmented6_prior_weather_correction.yaml`.

## Implementation

The fold-level weather correction is leak-safe for validation months:

- Fit a LightGBM model on hourly rows with `times <= train_end` only.
- Target is `renewable_error = 风光总加预测值 - 风光总加实际值`.
- Inputs include month, hour, renewable forecast, wind/solar forecast, GHI, cloud cover, precipitation, 100m wind, temperature, and pressure.
- Add these 15-minute aligned features to downstream price models:
  - `weather_renewable_error_pred`
  - `weather_renewable_corrected`
  - `weather_bid_space_corrected`
  - `weather_bid_space_delta`
  - `weather_correction_abs`

## Backtest Result

Compared with current champion `ens_champion_segmented6_prior` on the same 2025-09 to 2025-12 rolling folds:

| model | mean_profit | worst_fold | loss_days | oracle_ratio | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| champion | 8064.630 | 5928.223 | 5 | 0.7073 | 0.4377 | 0.6929 |
| weather_correction | 7676.675 | 5914.634 | 5 | 0.6743 | 0.4411 | 0.6939 |

Fold delta, weather minus champion:

| fold | avg_profit_delta | oracle_ratio_delta | loss_days_delta |
|---|---:|---:|---:|
| 2025-09 | -803.310 | -0.0559 | 0 |
| 2025-10 | -326.681 | -0.0235 | -1 |
| 2025-11 | -408.242 | -0.0510 | 0 |
| 2025-12 | -13.590 | -0.0013 | +1 |

Daily comparison:

- Days improved: 22 / 120
- Days worsened: 48 / 120
- Same selected charge/discharge pair: 37 / 120
- Mean daily profit delta: -377.403
- Worst daily delta: -7565.278 on 2025-09-17
- Best daily delta: +3461.350 on 2025-10-05

## Conclusion

Do not promote this version to champion.

The prior diagnostic showed weather can materially improve renewable and bid-space reconstruction error, but feeding the correction directly into the price ensemble does not improve dispatch profit. It changes selected windows often enough to hurt 9/10/11 month profitability.

Practical implication: keep weather correction as a diagnostic / candidate reranking signal, not as a direct champion replacement yet. A safer next step is to test weather correction only in gated reranking or bad-day avoidance, instead of letting it freely perturb the point-price model.

## Artifacts

- `reports/backtest_ens_champion_segmented6_prior_weather_correction.csv`
- `reports/backtest_ens_champion_segmented6_prior_weather_correction_daily.csv`
