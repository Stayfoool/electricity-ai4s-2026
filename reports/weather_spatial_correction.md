# Weather Spatial Correction Backtest

## Goal

Use fold-safe spatially selected NWP grid cells to correct renewable forecast error, then compare renewable and bid-space reconstruction against the old coarse approach.

## Methods

- `orig`: original renewable forecast.
- `mh`: month x hour mean renewable-error correction.
- `spatial_raw`: LightGBM predicts raw renewable forecast error from selected cells.
- `spatial_resid`: month-hour mean plus LightGBM residual correction.

## Error Summary

| method | renewable_mae | renewable_rmse | bid_space_mae | bid_space_rmse |
| --- | --- | --- | --- | --- |
| orig | 0.2969 | 0.4115 | 0.3749 | 0.5035 |
| mh | 0.2321 | 0.3265 | 0.2875 | 0.4029 |
| spatial_raw | 0.1060 | 0.1400 | 0.1661 | 0.2151 |
| spatial_resid | 0.1892 | 0.2477 | 0.2378 | 0.3173 |

## Fold Error Summary

| fold | hours | days | raw_best_iteration | resid_best_iteration | orig_renewable_mae | orig_renewable_rmse | orig_bid_space_mae | orig_bid_space_rmse | mh_renewable_mae | mh_renewable_rmse | mh_bid_space_mae | mh_bid_space_rmse | spatial_raw_renewable_mae | spatial_raw_renewable_rmse | spatial_raw_bid_space_mae | spatial_raw_bid_space_rmse | spatial_resid_renewable_mae | spatial_resid_renewable_rmse | spatial_resid_bid_space_mae | spatial_resid_bid_space_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 697 | 30 | 96 | 149 | 0.2670 | 0.3849 | 0.3211 | 0.4490 | 0.2177 | 0.3066 | 0.2613 | 0.3652 | 0.1199 | 0.1569 | 0.1699 | 0.2216 | 0.1870 | 0.2465 | 0.2295 | 0.3048 |
| valid_2025_10 | 744 | 31 | 123 | 198 | 0.1600 | 0.2346 | 0.2265 | 0.3026 | 0.1651 | 0.2046 | 0.1757 | 0.2315 | 0.0939 | 0.1210 | 0.1391 | 0.1776 | 0.1566 | 0.1886 | 0.1654 | 0.2085 |
| valid_2025_11 | 720 | 30 | 69 | 97 | 0.4089 | 0.5433 | 0.4759 | 0.6242 | 0.2874 | 0.4236 | 0.3491 | 0.5005 | 0.0994 | 0.1333 | 0.1544 | 0.2015 | 0.1914 | 0.2466 | 0.2334 | 0.3067 |
| valid_2025_12 | 744 | 31 | 85 | 20 | 0.3516 | 0.4834 | 0.4761 | 0.6383 | 0.2584 | 0.3713 | 0.3638 | 0.5146 | 0.1110 | 0.1488 | 0.2010 | 0.2598 | 0.2220 | 0.3090 | 0.3227 | 0.4491 |

## Hourly Bid-Space Dispatch Proxy

| method | days | mean_profit | mean_oracle_profit | mean_regret | loss_days |
| --- | --- | --- | --- | --- | --- |
| mh | 121 | 6805.1764 | 11211.7118 | 4406.5354 | 8 |
| orig | 121 | 6805.1764 | 11211.7118 | 4406.5354 | 8 |
| spatial_raw | 121 | 6540.8331 | 11211.7118 | 4670.8787 | 10 |
| spatial_resid | 121 | 6557.0956 | 11211.7118 | 4654.6162 | 10 |

## Top Feature Importance

| fold | model | feature | gain | split |
| --- | --- | --- | --- | --- |
| valid_2025_12 | spatial_raw | renewable_forecast | 4783.2326 | 294 |
| valid_2025_11 | spatial_raw | renewable_forecast | 4489.8489 | 267 |
| valid_2025_10 | spatial_raw | renewable_forecast | 3963.8207 | 311 |
| valid_2025_09 | spatial_raw | renewable_forecast | 3773.5118 | 298 |
| valid_2025_11 | spatial_resid | wind_forecast | 927.8922 | 225 |
| valid_2025_10 | spatial_resid | wind_forecast | 860.5688 | 307 |
| valid_2025_09 | spatial_resid | wind_forecast | 846.8910 | 262 |
| valid_2025_12 | spatial_resid | wind_forecast | 673.3499 | 67 |
| valid_2025_12 | spatial_raw | solar_forecast | 672.4544 | 135 |
| valid_2025_11 | spatial_raw | solar_forecast | 588.2918 | 110 |
| valid_2025_10 | spatial_raw | solar_forecast | 520.8456 | 160 |
| valid_2025_09 | spatial_raw | solar_forecast | 453.8632 | 125 |
| valid_2025_11 | spatial_resid | renewable_forecast | 382.1212 | 168 |
| valid_2025_10 | spatial_resid | renewable_forecast | 331.9322 | 243 |
| valid_2025_09 | spatial_resid | renewable_forecast | 322.2304 | 222 |
| valid_2025_10 | spatial_raw | wind_forecast | 287.0383 | 157 |
| valid_2025_11 | spatial_raw | wind_forecast | 282.9443 | 89 |
| valid_2025_12 | spatial_raw | wind_forecast | 281.2762 | 82 |
| valid_2025_09 | spatial_raw | wind_forecast | 249.2768 | 106 |
| valid_2025_12 | spatial_resid | renewable_forecast | 240.4908 | 61 |
| valid_2025_10 | spatial_resid | month | 222.9727 | 350 |
| valid_2025_11 | spatial_resid | month | 212.5592 | 225 |
| valid_2025_09 | spatial_resid | month | 190.0366 | 283 |
| valid_2025_12 | spatial_raw | month | 151.3098 | 205 |
| valid_2025_10 | spatial_raw | month | 136.8263 | 240 |
| valid_2025_10 | spatial_resid | solar_forecast | 132.9532 | 169 |
| valid_2025_11 | spatial_resid | solar_forecast | 131.9508 | 100 |
| valid_2025_09 | spatial_raw | month | 129.9670 | 227 |
| valid_2025_12 | spatial_resid | month | 125.1180 | 62 |
| valid_2025_09 | spatial_resid | solar_forecast | 119.2934 | 124 |
| valid_2025_11 | spatial_raw | month | 117.4078 | 164 |
| valid_2025_09 | spatial_resid | sp_v100_lat38p70_lon111p90 | 101.8777 | 118 |
| valid_2025_10 | spatial_resid | sp_v100_lat38p70_lon111p90 | 97.5908 | 151 |
| valid_2025_12 | spatial_resid | solar_forecast | 78.1979 | 24 |
| valid_2025_11 | spatial_resid | sp_v100_lat38p70_lon111p90 | 66.9065 | 80 |
| valid_2025_09 | spatial_resid | sp_v100_lat38p60_lon111p90 | 55.0760 | 92 |
| valid_2025_10 | spatial_resid | sp_v100_lat38p60_lon111p90 | 54.4884 | 89 |
| valid_2025_10 | spatial_raw | sp_wind_speed_lat40p10_lon113p00 | 48.5193 | 63 |
| valid_2025_09 | spatial_resid | sp_wind_speed_lat39p80_lon112p00 | 44.0484 | 58 |
| valid_2025_09 | spatial_raw | sp_wind_speed_lat40p10_lon113p00 | 41.2575 | 47 |

## Readout

- This still evaluates a bid-space proxy, not the final champion price model.
- Promote only if spatial correction improves bid-space error and does not worsen the dispatch proxy.
- If `spatial_resid` beats `spatial_raw`, weather adds signal beyond month/hour bias.

## Artifacts

- `reports/weather_spatial_correction_folds.csv`
- `reports/weather_spatial_correction_daily_dispatch.csv`
- `reports/weather_spatial_selected_features.csv`
- `reports/weather_spatial_correction_importance.csv`
