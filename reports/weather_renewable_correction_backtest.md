# Weather Renewable Correction Backtest

This experiment predicts `renewable_error = forecast - actual` using only past fold data, then subtracts the predicted error from renewable forecast.

Methods:

- `orig`: original renewable forecast.
- `month_hour_mean`: historical month x hour mean renewable-error correction.
- `lgb_weather`: LightGBM correction using NWP and renewable forecast features.

## Overall Error Summary

| method | renewable_mae | renewable_rmse | bid_space_mae | bid_space_rmse |
| --- | --- | --- | --- | --- |
| orig | 0.2964 | 0.4274 | 0.3749 | 0.5222 |
| month_hour_mean | 0.2319 | 0.3361 | 0.2873 | 0.4189 |
| lgb_weather | 0.1055 | 0.1405 | 0.1687 | 0.2193 |

## Fold Error Summary

| fold | hours | days | orig_renewable_mae | mh_renewable_mae | lgb_renewable_mae | orig_bid_space_mae | mh_bid_space_mae | lgb_bid_space_mae | orig_bid_space_rmse | mh_bid_space_rmse | lgb_bid_space_rmse | lgb_best_iteration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 697 | 30 | 0.2670 | 0.2177 | 0.1193 | 0.3211 | 0.2613 | 0.1712 | 0.4490 | 0.3651 | 0.2217 | 108 |
| valid_2025_10 | 744 | 31 | 0.1600 | 0.1653 | 0.0958 | 0.2265 | 0.1758 | 0.1432 | 0.3026 | 0.2315 | 0.1811 | 95 |
| valid_2025_11 | 720 | 30 | 0.4089 | 0.2872 | 0.0966 | 0.4759 | 0.3489 | 0.1551 | 0.6242 | 0.5002 | 0.2044 | 66 |
| valid_2025_12 | 744 | 31 | 0.3516 | 0.2584 | 0.1110 | 0.4761 | 0.3636 | 0.2052 | 0.6383 | 0.5144 | 0.2617 | 96 |

## Hourly Bid-Space Dispatch Proxy Summary

This is only a proxy: hourly bid-space signals are repeated to 15-minute slots, then legal 8-slot pairs are selected. It tests directionality, not final champion impact.

| method | days | mean_profit | mean_oracle_profit | mean_regret | loss_days | mean_abs_charge_gap | mean_abs_discharge_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lgb | 121 | 6819.4164 | 11211.7118 | 4392.2953 | 7 | 12.7190 | 12.3802 |
| mh | 121 | 6805.1764 | 11211.7118 | 4406.5354 | 8 | 12.7686 | 12.1818 |
| orig | 121 | 6805.1764 | 11211.7118 | 4406.5354 | 8 | 12.7686 | 12.1818 |

## Readout

- Use this to decide whether corrected renewable improves bid-space accuracy.
- Do not promote to champion unless fold-level bid-space error and profit proxy improve.
- If useful, next step is adding corrected renewable/bid_space as model or reranker features.

## Artifacts

Tracked report artifacts:

- `reports/weather_renewable_correction_folds.csv`
- `reports/weather_renewable_correction_daily_dispatch.csv`

Local cache artifact, intentionally ignored by Git:

- `reports/weather_renewable_correction_predictions.csv`
