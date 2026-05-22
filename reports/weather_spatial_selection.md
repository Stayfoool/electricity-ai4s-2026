# Weather Spatial Selection Diagnostics

## Goal

Find whether specific NWP grid cells, using only core weather variables, explain renewable forecast errors better than coarse full-grid averages.

## Method

- NetCDF files scanned: `424`.
- Core weather variables: `ghi`, `tcc`, `u100`, `v100`, `wind_speed`.
- Targets: `renewable_error`, `wind_error`, `solar_error`, plus month-hour residual versions.
- Each fold uses only hours before that fold's `train_end` for spatial ranking.
- Correlation is computed per grid cell with streaming accumulators, avoiding a huge expanded table.
- Month-hour residual targets reduce pure seasonal/daytime effects in the target side.

## Top Focus Cells

| fold | scope | target | weather_var | rank | n_hours | pearson | abs_pearson | lat | lon | cell_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | all | renewable_error_mh_resid | ghi | 1 | 5808 | -0.0714 | 0.0714 | 46.5000 | 117.6000 | lat=46.50,lon=117.60 |
| valid_2025_09 | all | renewable_error_mh_resid | ghi | 2 | 5808 | -0.0714 | 0.0714 | 46.5000 | 117.4000 | lat=46.50,lon=117.40 |
| valid_2025_09 | all | renewable_error_mh_resid | ghi | 3 | 5808 | -0.0713 | 0.0713 | 46.5000 | 117.5000 | lat=46.50,lon=117.50 |
| valid_2025_09 | all | renewable_error_mh_resid | ghi | 4 | 5808 | -0.0711 | 0.0711 | 46.4000 | 117.4000 | lat=46.40,lon=117.40 |
| valid_2025_09 | all | renewable_error_mh_resid | ghi | 5 | 5808 | -0.0704 | 0.0704 | 46.0000 | 116.8000 | lat=46.00,lon=116.80 |
| valid_2025_10 | all | renewable_error_mh_resid | ghi | 1 | 6505 | -0.0764 | 0.0764 | 46.5000 | 117.4000 | lat=46.50,lon=117.40 |
| valid_2025_10 | all | renewable_error_mh_resid | ghi | 2 | 6505 | -0.0761 | 0.0761 | 46.4000 | 117.4000 | lat=46.40,lon=117.40 |
| valid_2025_10 | all | renewable_error_mh_resid | ghi | 3 | 6505 | -0.0760 | 0.0760 | 46.5000 | 117.5000 | lat=46.50,lon=117.50 |
| valid_2025_10 | all | renewable_error_mh_resid | ghi | 4 | 6505 | -0.0759 | 0.0759 | 46.5000 | 117.6000 | lat=46.50,lon=117.60 |
| valid_2025_10 | all | renewable_error_mh_resid | ghi | 5 | 6505 | -0.0758 | 0.0758 | 46.1000 | 116.9000 | lat=46.10,lon=116.90 |
| valid_2025_11 | all | renewable_error_mh_resid | ghi | 1 | 7249 | 0.0776 | 0.0776 | 36.4000 | 108.7000 | lat=36.40,lon=108.70 |
| valid_2025_11 | all | renewable_error_mh_resid | ghi | 2 | 7249 | 0.0765 | 0.0765 | 36.3000 | 108.8000 | lat=36.30,lon=108.80 |
| valid_2025_11 | all | renewable_error_mh_resid | ghi | 3 | 7249 | 0.0765 | 0.0765 | 36.3000 | 108.7000 | lat=36.30,lon=108.70 |
| valid_2025_11 | all | renewable_error_mh_resid | ghi | 4 | 7249 | 0.0761 | 0.0761 | 36.4000 | 108.6000 | lat=36.40,lon=108.60 |
| valid_2025_11 | all | renewable_error_mh_resid | ghi | 5 | 7249 | 0.0761 | 0.0761 | 36.3000 | 108.3000 | lat=36.30,lon=108.30 |
| valid_2025_12 | all | renewable_error_mh_resid | ghi | 1 | 7969 | 0.0797 | 0.0797 | 36.4000 | 108.7000 | lat=36.40,lon=108.70 |
| valid_2025_12 | all | renewable_error_mh_resid | ghi | 2 | 7969 | 0.0792 | 0.0792 | 36.3000 | 108.7000 | lat=36.30,lon=108.70 |
| valid_2025_12 | all | renewable_error_mh_resid | ghi | 3 | 7969 | 0.0790 | 0.0790 | 36.2000 | 108.5000 | lat=36.20,lon=108.50 |
| valid_2025_12 | all | renewable_error_mh_resid | ghi | 4 | 7969 | 0.0787 | 0.0787 | 36.3000 | 108.8000 | lat=36.30,lon=108.80 |
| valid_2025_12 | all | renewable_error_mh_resid | ghi | 5 | 7969 | 0.0787 | 0.0787 | 36.3000 | 108.6000 | lat=36.30,lon=108.60 |
| valid_2025_09 | all | renewable_error_mh_resid | tcc | 1 | 5808 | 0.2222 | 0.2222 | 45.4000 | 117.9000 | lat=45.40,lon=117.90 |
| valid_2025_09 | all | renewable_error_mh_resid | tcc | 2 | 5808 | 0.2208 | 0.2208 | 45.5000 | 117.9000 | lat=45.50,lon=117.90 |
| valid_2025_09 | all | renewable_error_mh_resid | tcc | 3 | 5808 | 0.2193 | 0.2193 | 45.4000 | 117.8000 | lat=45.40,lon=117.80 |
| valid_2025_09 | all | renewable_error_mh_resid | tcc | 4 | 5808 | 0.2190 | 0.2190 | 45.6000 | 118.3000 | lat=45.60,lon=118.30 |
| valid_2025_09 | all | renewable_error_mh_resid | tcc | 5 | 5808 | 0.2187 | 0.2187 | 45.5000 | 118.0000 | lat=45.50,lon=118.00 |
| valid_2025_10 | all | renewable_error_mh_resid | tcc | 1 | 6505 | 0.2288 | 0.2288 | 45.4000 | 117.9000 | lat=45.40,lon=117.90 |
| valid_2025_10 | all | renewable_error_mh_resid | tcc | 2 | 6505 | 0.2277 | 0.2277 | 45.5000 | 117.9000 | lat=45.50,lon=117.90 |
| valid_2025_10 | all | renewable_error_mh_resid | tcc | 3 | 6505 | 0.2274 | 0.2274 | 45.4000 | 117.8000 | lat=45.40,lon=117.80 |
| valid_2025_10 | all | renewable_error_mh_resid | tcc | 4 | 6505 | 0.2269 | 0.2269 | 45.5000 | 118.0000 | lat=45.50,lon=118.00 |
| valid_2025_10 | all | renewable_error_mh_resid | tcc | 5 | 6505 | 0.2262 | 0.2262 | 45.4000 | 118.0000 | lat=45.40,lon=118.00 |
| valid_2025_11 | all | renewable_error_mh_resid | tcc | 1 | 7249 | 0.2218 | 0.2218 | 45.4000 | 117.9000 | lat=45.40,lon=117.90 |
| valid_2025_11 | all | renewable_error_mh_resid | tcc | 2 | 7249 | 0.2212 | 0.2212 | 45.4000 | 117.8000 | lat=45.40,lon=117.80 |
| valid_2025_11 | all | renewable_error_mh_resid | tcc | 3 | 7249 | 0.2199 | 0.2199 | 45.5000 | 117.9000 | lat=45.50,lon=117.90 |
| valid_2025_11 | all | renewable_error_mh_resid | tcc | 4 | 7249 | 0.2194 | 0.2194 | 46.5000 | 117.1000 | lat=46.50,lon=117.10 |
| valid_2025_11 | all | renewable_error_mh_resid | tcc | 5 | 7249 | 0.2194 | 0.2194 | 45.5000 | 118.0000 | lat=45.50,lon=118.00 |
| valid_2025_12 | all | renewable_error_mh_resid | tcc | 1 | 7969 | 0.2279 | 0.2279 | 45.4000 | 117.8000 | lat=45.40,lon=117.80 |
| valid_2025_12 | all | renewable_error_mh_resid | tcc | 2 | 7969 | 0.2275 | 0.2275 | 45.4000 | 117.9000 | lat=45.40,lon=117.90 |
| valid_2025_12 | all | renewable_error_mh_resid | tcc | 3 | 7969 | 0.2272 | 0.2272 | 46.5000 | 115.8000 | lat=46.50,lon=115.80 |
| valid_2025_12 | all | renewable_error_mh_resid | tcc | 4 | 7969 | 0.2260 | 0.2260 | 46.5000 | 117.1000 | lat=46.50,lon=117.10 |
| valid_2025_12 | all | renewable_error_mh_resid | tcc | 5 | 7969 | 0.2258 | 0.2258 | 45.4000 | 117.6000 | lat=45.40,lon=117.60 |
| valid_2025_09 | all | renewable_error_mh_resid | u100 | 1 | 5808 | 0.3394 | 0.3394 | 40.3000 | 112.9000 | lat=40.30,lon=112.90 |
| valid_2025_09 | all | renewable_error_mh_resid | u100 | 2 | 5808 | 0.3331 | 0.3331 | 38.9000 | 112.2000 | lat=38.90,lon=112.20 |
| valid_2025_09 | all | renewable_error_mh_resid | u100 | 3 | 5808 | 0.3327 | 0.3327 | 40.6000 | 112.7000 | lat=40.60,lon=112.70 |
| valid_2025_09 | all | renewable_error_mh_resid | u100 | 4 | 5808 | 0.3318 | 0.3318 | 40.6000 | 112.6000 | lat=40.60,lon=112.60 |
| valid_2025_09 | all | renewable_error_mh_resid | u100 | 5 | 5808 | 0.3273 | 0.3273 | 39.4000 | 112.3000 | lat=39.40,lon=112.30 |
| valid_2025_10 | all | renewable_error_mh_resid | u100 | 1 | 6505 | 0.3397 | 0.3397 | 40.3000 | 112.9000 | lat=40.30,lon=112.90 |
| valid_2025_10 | all | renewable_error_mh_resid | u100 | 2 | 6505 | 0.3388 | 0.3388 | 40.6000 | 112.6000 | lat=40.60,lon=112.60 |
| valid_2025_10 | all | renewable_error_mh_resid | u100 | 3 | 6505 | 0.3341 | 0.3341 | 40.6000 | 112.7000 | lat=40.60,lon=112.70 |
| valid_2025_10 | all | renewable_error_mh_resid | u100 | 4 | 6505 | 0.3277 | 0.3277 | 38.9000 | 112.2000 | lat=38.90,lon=112.20 |
| valid_2025_10 | all | renewable_error_mh_resid | u100 | 5 | 6505 | 0.3253 | 0.3253 | 40.6000 | 112.5000 | lat=40.60,lon=112.50 |
| valid_2025_11 | all | renewable_error_mh_resid | u100 | 1 | 7249 | 0.3198 | 0.3198 | 40.3000 | 112.9000 | lat=40.30,lon=112.90 |
| valid_2025_11 | all | renewable_error_mh_resid | u100 | 2 | 7249 | 0.3183 | 0.3183 | 40.6000 | 112.6000 | lat=40.60,lon=112.60 |
| valid_2025_11 | all | renewable_error_mh_resid | u100 | 3 | 7249 | 0.3131 | 0.3131 | 40.6000 | 112.7000 | lat=40.60,lon=112.70 |
| valid_2025_11 | all | renewable_error_mh_resid | u100 | 4 | 7249 | 0.3078 | 0.3078 | 38.9000 | 112.2000 | lat=38.90,lon=112.20 |
| valid_2025_11 | all | renewable_error_mh_resid | u100 | 5 | 7249 | 0.3044 | 0.3044 | 40.6000 | 112.5000 | lat=40.60,lon=112.50 |
| valid_2025_12 | all | renewable_error_mh_resid | u100 | 1 | 7969 | 0.3350 | 0.3350 | 40.6000 | 112.6000 | lat=40.60,lon=112.60 |
| valid_2025_12 | all | renewable_error_mh_resid | u100 | 2 | 7969 | 0.3309 | 0.3309 | 40.3000 | 112.9000 | lat=40.30,lon=112.90 |
| valid_2025_12 | all | renewable_error_mh_resid | u100 | 3 | 7969 | 0.3285 | 0.3285 | 40.6000 | 112.7000 | lat=40.60,lon=112.70 |
| valid_2025_12 | all | renewable_error_mh_resid | u100 | 4 | 7969 | 0.3282 | 0.3282 | 41.2000 | 107.0000 | lat=41.20,lon=107.00 |
| valid_2025_12 | all | renewable_error_mh_resid | u100 | 5 | 7969 | 0.3261 | 0.3261 | 41.3000 | 107.1000 | lat=41.30,lon=107.10 |
| valid_2025_09 | all | renewable_error_mh_resid | v100 | 1 | 5808 | -0.3133 | 0.3133 | 41.0000 | 106.6000 | lat=41.00,lon=106.60 |
| valid_2025_09 | all | renewable_error_mh_resid | v100 | 2 | 5808 | -0.3117 | 0.3117 | 40.9000 | 106.5000 | lat=40.90,lon=106.50 |
| valid_2025_09 | all | renewable_error_mh_resid | v100 | 3 | 5808 | -0.3080 | 0.3080 | 41.0000 | 106.7000 | lat=41.00,lon=106.70 |
| valid_2025_09 | all | renewable_error_mh_resid | v100 | 4 | 5808 | -0.2997 | 0.2997 | 40.8000 | 106.4000 | lat=40.80,lon=106.40 |
| valid_2025_09 | all | renewable_error_mh_resid | v100 | 5 | 5808 | -0.2975 | 0.2975 | 41.0000 | 106.5000 | lat=41.00,lon=106.50 |
| valid_2025_10 | all | renewable_error_mh_resid | v100 | 1 | 6505 | -0.3086 | 0.3086 | 40.9000 | 106.5000 | lat=40.90,lon=106.50 |
| valid_2025_10 | all | renewable_error_mh_resid | v100 | 2 | 6505 | -0.3084 | 0.3084 | 41.0000 | 106.6000 | lat=41.00,lon=106.60 |
| valid_2025_10 | all | renewable_error_mh_resid | v100 | 3 | 6505 | -0.3038 | 0.3038 | 41.0000 | 106.7000 | lat=41.00,lon=106.70 |
| valid_2025_10 | all | renewable_error_mh_resid | v100 | 4 | 6505 | -0.2977 | 0.2977 | 40.8000 | 106.4000 | lat=40.80,lon=106.40 |
| valid_2025_10 | all | renewable_error_mh_resid | v100 | 5 | 6505 | -0.2917 | 0.2917 | 41.0000 | 106.5000 | lat=41.00,lon=106.50 |
| valid_2025_11 | all | renewable_error_mh_resid | v100 | 1 | 7249 | -0.2894 | 0.2894 | 41.0000 | 106.6000 | lat=41.00,lon=106.60 |
| valid_2025_11 | all | renewable_error_mh_resid | v100 | 2 | 7249 | -0.2890 | 0.2890 | 40.9000 | 106.5000 | lat=40.90,lon=106.50 |
| valid_2025_11 | all | renewable_error_mh_resid | v100 | 3 | 7249 | -0.2849 | 0.2849 | 41.0000 | 106.7000 | lat=41.00,lon=106.70 |
| valid_2025_11 | all | renewable_error_mh_resid | v100 | 4 | 7249 | -0.2783 | 0.2783 | 40.8000 | 106.4000 | lat=40.80,lon=106.40 |
| valid_2025_11 | all | renewable_error_mh_resid | v100 | 5 | 7249 | -0.2716 | 0.2716 | 41.0000 | 106.5000 | lat=41.00,lon=106.50 |
| valid_2025_12 | all | renewable_error_mh_resid | v100 | 1 | 7969 | -0.3088 | 0.3088 | 41.0000 | 106.6000 | lat=41.00,lon=106.60 |
| valid_2025_12 | all | renewable_error_mh_resid | v100 | 2 | 7969 | -0.3045 | 0.3045 | 40.9000 | 106.5000 | lat=40.90,lon=106.50 |
| valid_2025_12 | all | renewable_error_mh_resid | v100 | 3 | 7969 | -0.3025 | 0.3025 | 41.0000 | 106.7000 | lat=41.00,lon=106.70 |
| valid_2025_12 | all | renewable_error_mh_resid | v100 | 4 | 7969 | -0.2917 | 0.2917 | 40.8000 | 106.4000 | lat=40.80,lon=106.40 |
| valid_2025_12 | all | renewable_error_mh_resid | v100 | 5 | 7969 | -0.2899 | 0.2899 | 41.1000 | 106.8000 | lat=41.10,lon=106.80 |

## Stable Top-10 Cells Across Folds

| scope | target | weather_var | cell_id | folds_seen | mean_abs_pearson | max_abs_pearson | mean_rank | lat | lon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | renewable_error | ghi | lat=38.80,lon=106.00 | 2 | 0.6476 | 0.6498 | 1.0000 | 38.8000 | 106.0000 |
| all | renewable_error | ghi | lat=38.90,lon=106.10 | 2 | 0.6472 | 0.6493 | 2.0000 | 38.9000 | 106.1000 |
| all | renewable_error | ghi | lat=38.80,lon=106.10 | 2 | 0.6470 | 0.6491 | 3.0000 | 38.8000 | 106.1000 |
| all | renewable_error | ghi | lat=39.00,lon=106.20 | 2 | 0.6465 | 0.6487 | 4.0000 | 39.0000 | 106.2000 |
| all | renewable_error | ghi | lat=39.20,lon=106.60 | 2 | 0.6453 | 0.6478 | 5.5000 | 39.2000 | 106.6000 |
| all | renewable_error | ghi | lat=38.90,lon=106.20 | 2 | 0.6452 | 0.6473 | 6.5000 | 38.9000 | 106.2000 |
| all | renewable_error | ghi | lat=39.20,lon=106.50 | 2 | 0.6448 | 0.6472 | 8.0000 | 39.2000 | 106.5000 |
| all | renewable_error | ghi | lat=39.10,lon=109.80 | 2 | 0.6414 | 0.6477 | 5.0000 | 39.1000 | 109.8000 |
| all | renewable_error | ghi | lat=39.10,lon=109.70 | 2 | 0.6409 | 0.6473 | 6.5000 | 39.1000 | 109.7000 |
| all | renewable_error | ghi | lat=39.20,lon=109.80 | 2 | 0.6406 | 0.6471 | 9.0000 | 39.2000 | 109.8000 |
| all | renewable_error | tcc | lat=45.50,lon=117.90 | 3 | 0.1929 | 0.1951 | 2.0000 | 45.5000 | 117.9000 |
| all | renewable_error | tcc | lat=45.40,lon=116.80 | 3 | 0.1922 | 0.1946 | 3.6667 | 45.4000 | 116.8000 |
| all | renewable_error | tcc | lat=45.30,lon=117.50 | 3 | 0.1917 | 0.1935 | 4.0000 | 45.3000 | 117.5000 |
| all | renewable_error | tcc | lat=45.40,lon=117.20 | 3 | 0.1910 | 0.1929 | 5.0000 | 45.4000 | 117.2000 |
| all | renewable_error | tcc | lat=45.30,lon=117.20 | 3 | 0.1894 | 0.1913 | 9.3333 | 45.3000 | 117.2000 |
| all | renewable_error | tcc | lat=45.50,lon=116.60 | 2 | 0.1930 | 0.1942 | 2.5000 | 45.5000 | 116.6000 |
| all | renewable_error | tcc | lat=45.50,lon=116.70 | 2 | 0.1924 | 0.1932 | 4.0000 | 45.5000 | 116.7000 |
| all | renewable_error | tcc | lat=45.30,lon=117.40 | 2 | 0.1904 | 0.1908 | 9.5000 | 45.3000 | 117.4000 |
| all | renewable_error | tcc | lat=45.60,lon=116.90 | 2 | 0.1901 | 0.1918 | 7.5000 | 45.6000 | 116.9000 |
| all | renewable_error | u100 | lat=40.00,lon=115.00 | 4 | 0.4556 | 0.4678 | 1.5000 | 40.0000 | 115.0000 |
| all | renewable_error | u100 | lat=40.00,lon=114.90 | 4 | 0.4505 | 0.4610 | 4.0000 | 40.0000 | 114.9000 |
| all | renewable_error | u100 | lat=41.10,lon=112.30 | 4 | 0.4488 | 0.4553 | 3.5000 | 41.1000 | 112.3000 |
| all | renewable_error | u100 | lat=40.60,lon=117.30 | 4 | 0.4451 | 0.4524 | 5.5000 | 40.6000 | 117.3000 |
| all | renewable_error | u100 | lat=41.00,lon=112.30 | 4 | 0.4447 | 0.4559 | 5.5000 | 41.0000 | 112.3000 |
| all | renewable_error | u100 | lat=37.90,lon=111.30 | 3 | 0.4486 | 0.4641 | 4.0000 | 37.9000 | 111.3000 |
| all | renewable_error | u100 | lat=40.70,lon=112.10 | 3 | 0.4397 | 0.4435 | 7.6667 | 40.7000 | 112.1000 |
| all | renewable_error | u100 | lat=36.70,lon=111.90 | 3 | 0.4390 | 0.4443 | 7.6667 | 36.7000 | 111.9000 |
| all | renewable_error | u100 | lat=36.60,lon=111.80 | 3 | 0.4382 | 0.4500 | 8.0000 | 36.6000 | 111.8000 |
| all | renewable_error | u100 | lat=36.80,lon=111.90 | 2 | 0.4371 | 0.4389 | 6.5000 | 36.8000 | 111.9000 |
| all | renewable_error | u100 | lat=36.60,lon=111.90 | 2 | 0.4363 | 0.4464 | 7.5000 | 36.6000 | 111.9000 |
| all | renewable_error | v100 | lat=37.70,lon=102.50 | 4 | 0.4179 | 0.4318 | 1.5000 | 37.7000 | 102.5000 |
| all | renewable_error | v100 | lat=37.90,lon=102.30 | 4 | 0.4021 | 0.4141 | 2.5000 | 37.9000 | 102.3000 |
| all | renewable_error | v100 | lat=38.10,lon=101.60 | 4 | 0.3965 | 0.4107 | 3.7500 | 38.1000 | 101.6000 |
| all | renewable_error | v100 | lat=38.10,lon=101.80 | 4 | 0.3892 | 0.4037 | 5.7500 | 38.1000 | 101.8000 |
| all | renewable_error | v100 | lat=37.40,lon=102.80 | 4 | 0.3885 | 0.4014 | 5.5000 | 37.4000 | 102.8000 |
| all | renewable_error | v100 | lat=36.20,lon=101.70 | 3 | 0.3979 | 0.4043 | 4.3333 | 36.2000 | 101.7000 |
| all | renewable_error | v100 | lat=37.70,lon=102.60 | 3 | 0.3878 | 0.3931 | 8.3333 | 37.7000 | 102.6000 |
| all | renewable_error | v100 | lat=38.90,lon=99.90 | 2 | 0.3902 | 0.3916 | 7.5000 | 38.9000 | 99.9000 |
| all | renewable_error | v100 | lat=38.10,lon=101.90 | 2 | 0.3888 | 0.3954 | 8.0000 | 38.1000 | 101.9000 |
| all | renewable_error | v100 | lat=38.40,lon=100.60 | 2 | 0.3853 | 0.3866 | 9.5000 | 38.4000 | 100.6000 |

## Readout

- Stable cells across folds are safer candidates for correction features than one-fold top cells.
- If winter-like scopes show stronger and more stable cells than `all`, weather should be treated as a winter-specific signal.
- This report does not yet prove profit lift; it only decides whether spatial weather correction is worth backtesting.

## Artifacts

- `reports/weather_spatial_top_cells.csv`
- `reports/weather_spatial_stability.csv`
- `reports/weather_spatial_top_cells.png`
