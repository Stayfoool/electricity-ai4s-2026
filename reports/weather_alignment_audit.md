# Weather Alignment Audit

Purpose: verify whether the competition NWP `.nc` files are aligned correctly before using weather as price, bid-space, or reranking features.

## Verdict

- Current project alignment is correct: file date `D` maps to Beijing target day `D+1`, and `lead_time=0..23` maps to local hours `00:00..23:00`.
- This is equivalent to `time_coord_utc + lead_time + 8h`; sampled files have `time=16:00 UTC`, so `+8h` lands on midnight Beijing time of the target day.
- The local NetCDF files do **not** contain any `night` channel or `night` suffix field. Available channels are listed below.
- Test-period weather coverage is complete under the current alignment; train coverage misses 2025-01-01 because `2024-12-31.nc` is not provided.

## NetCDF Inventory

- Current alignment train/test coverage: `8713/8737` train hours and `1416/1416` test hours.
- Files: `424`.
- File date range: `2025-01-01` to `2026-02-28`.
- Channels: `ghi,sp,t2m,tcc,tp,u100,v100`.

| file | issue_date_from_file | time_coord_utc | time_plus_8_bjt | target_date_by_rule | lead_min | lead_max | lead_count | channels | has_night_channel | lat_count | lon_count | lat_min | lat_max | lon_min | lon_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20250101.nc | 2025-01-01 | 2025-01-01T16:00:00 | 2025-01-02T00:00:00 | 2025-01-02 | 0 | 23 | 24 | ghi,sp,t2m,tcc,tp,u100,v100 | False | 104 | 225 | 36.2000 | 46.5000 | 96.1000 | 118.5000 |
| 20250801.nc | 2025-08-01 | 2025-08-01T16:00:00 | 2025-08-02T00:00:00 | 2025-08-02 | 0 | 23 | 24 | ghi,sp,t2m,tcc,tp,u100,v100 | False | 104 | 225 | 36.2000 | 46.5000 | 96.1000 | 118.5000 |
| 20251231.nc | 2025-12-31 | 2025-12-31T16:00:00 | 2026-01-01T00:00:00 | 2026-01-01 | 0 | 23 | 24 | ghi,sp,t2m,tcc,tp,u100,v100 | False | 104 | 225 | 36.2000 | 46.5000 | 96.1000 | 118.5000 |
| 20260101.nc | 2026-01-01 | 2026-01-01T16:00:00 | 2026-01-02T00:00:00 | 2026-01-02 | 0 | 23 | 24 | ghi,sp,t2m,tcc,tp,u100,v100 | False | 104 | 225 | 36.2000 | 46.5000 | 96.1000 | 118.5000 |
| 20260227.nc | 2026-02-27 | 2026-02-27T16:00:00 | 2026-02-28T00:00:00 | 2026-02-28 | 0 | 23 | 24 | ghi,sp,t2m,tcc,tp,u100,v100 | False | 104 | 225 | 36.2000 | 46.5000 | 96.1000 | 118.5000 |
| 20260228.nc | 2026-02-28 | 2026-02-28T16:00:00 | 2026-03-01T00:00:00 | 2026-03-01 | 0 | 23 | 24 | ghi,sp,t2m,tcc,tp,u100,v100 | False | 104 | 225 | 36.2000 | 46.5000 | 96.1000 | 118.5000 |

## Coverage By Alignment Hypothesis

| alignment | weather_hours | weather_min | weather_max | train_hours | train_join_hours | train_coverage | test_hours | test_join_hours | test_coverage | test_missing_hours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_utc_plus8 | 10176 | 2025-01-02 | 2026-03-01 | 8737 | 8713 | 0.9973 | 1416 | 1416 | 1.0000 | 0 |
| filename_d_plus_1 | 10176 | 2025-01-02 | 2026-03-01 | 8737 | 8713 | 0.9973 | 1416 | 1416 | 1.0000 | 0 |
| no_utc_plus8 | 10176 | 2025-01-01 | 2026-03-01 | 8737 | 8721 | 0.9982 | 1416 | 1416 | 1.0000 | 0 |
| filename_same_day | 10176 | 2025-01-01 | 2026-02-28 | 8737 | 8737 | 1.0000 | 1416 | 1416 | 1.0000 | 0 |

## Alignment Signal Checks

These probes compare weather features with actual renewable output, renewable forecast error, bid-space error, and hourly price. `month_hour_residual` removes the month x hour average first, so it tests information beyond seasonal intraday pattern.

| probe | alignment | rows | pearson | spearman |
| --- | --- | --- | --- | --- |
| month_hour_residual:all:wind_speed_mean->bid_space_error | current_utc_plus8 | 8713 | -0.4522 | -0.3908 |
| month_hour_residual:all:wind_speed_mean->bid_space_error | filename_same_day | 8737 | -0.1412 | -0.1128 |
| month_hour_residual:all:wind_speed_mean->bid_space_error | no_utc_plus8 | 8721 | -0.2732 | -0.2480 |
| month_hour_residual:all:wind_speed_mean->price_hour | current_utc_plus8 | 8713 | -0.4308 | -0.5354 |
| month_hour_residual:all:wind_speed_mean->price_hour | filename_same_day | 8737 | -0.2213 | -0.2602 |
| month_hour_residual:all:wind_speed_mean->price_hour | no_utc_plus8 | 8721 | -0.3598 | -0.4420 |
| month_hour_residual:all:wind_speed_mean->renewable_error | current_utc_plus8 | 8713 | 0.4622 | 0.4019 |
| month_hour_residual:all:wind_speed_mean->renewable_error | filename_same_day | 8737 | 0.1517 | 0.1185 |
| month_hour_residual:all:wind_speed_mean->renewable_error | no_utc_plus8 | 8721 | 0.2928 | 0.2640 |
| month_hour_residual:all:wind_speed_mean->wind_error | current_utc_plus8 | 8713 | 0.4404 | 0.3702 |
| month_hour_residual:all:wind_speed_mean->wind_error | filename_same_day | 8737 | 0.1397 | 0.1048 |
| month_hour_residual:all:wind_speed_mean->wind_error | no_utc_plus8 | 8721 | 0.2754 | 0.2328 |
| month_hour_residual:midday_10_17:ghi_mean->price_hour | current_utc_plus8 | 2904 | -0.3261 | -0.3154 |
| month_hour_residual:midday_10_17:ghi_mean->price_hour | filename_same_day | 2912 | -0.2689 | -0.2822 |
| month_hour_residual:midday_10_17:ghi_mean->price_hour | no_utc_plus8 | 2906 | -0.1025 | -0.0772 |
| month_hour_residual:midday_10_17:ghi_mean->solar_error | current_utc_plus8 | 2904 | 0.2896 | 0.2537 |
| month_hour_residual:midday_10_17:ghi_mean->solar_error | filename_same_day | 2912 | 0.2540 | 0.2150 |
| month_hour_residual:midday_10_17:ghi_mean->solar_error | no_utc_plus8 | 2906 | 0.0959 | 0.0024 |
| raw:all:wind_speed_mean->wind_actual | current_utc_plus8 | 8713 | 0.7365 | 0.7605 |
| raw:all:wind_speed_mean->wind_actual | filename_same_day | 8737 | 0.4614 | 0.4718 |
| raw:all:wind_speed_mean->wind_actual | no_utc_plus8 | 8721 | 0.5939 | 0.6109 |
| raw:midday_10_17:ghi_mean->solar_actual | current_utc_plus8 | 2904 | 0.5218 | 0.4554 |
| raw:midday_10_17:ghi_mean->solar_actual | filename_same_day | 2912 | 0.4357 | 0.3614 |
| raw:midday_10_17:ghi_mean->solar_actual | no_utc_plus8 | 2906 | 0.1852 | 0.3670 |

## Top Residual Correlations For Error Targets

| alignment | scope | target | feature | kind | rows | pearson | spearman | abs_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_utc_plus8 | all | bid_space_error | wind_speed_mean | month_hour_residual | 8713 | -0.4522 | -0.3908 | 0.3908 |
| current_utc_plus8 | all | bid_space_error | wind_speed_max | month_hour_residual | 8713 | -0.3032 | -0.2805 | 0.2805 |
| current_utc_plus8 | all | bid_space_error | u100_mean | month_hour_residual | 8713 | -0.2914 | -0.2455 | 0.2455 |
| current_utc_plus8 | all | bid_space_error | sp_mean | month_hour_residual | 8713 | 0.1210 | 0.1290 | 0.1290 |
| current_utc_plus8 | all | bid_space_error | v100_mean | month_hour_residual | 8713 | 0.1609 | 0.1047 | 0.1047 |
| current_utc_plus8 | all | bid_space_error | ghi_mean | month_hour_residual | 8713 | -0.1079 | -0.0506 | 0.0506 |
| current_utc_plus8 | all | bid_space_error | t2m_mean | month_hour_residual | 8713 | 0.0910 | 0.0437 | 0.0437 |
| current_utc_plus8 | all | bid_space_error | ghi_max | month_hour_residual | 8713 | -0.0485 | -0.0160 | 0.0160 |
| current_utc_plus8 | all | renewable_error | wind_speed_mean | month_hour_residual | 8713 | 0.4622 | 0.4019 | 0.4019 |
| current_utc_plus8 | all | renewable_error | wind_speed_max | month_hour_residual | 8713 | 0.3177 | 0.2898 | 0.2898 |
| current_utc_plus8 | all | renewable_error | u100_mean | month_hour_residual | 8713 | 0.3357 | 0.2672 | 0.2672 |
| current_utc_plus8 | all | renewable_error | sp_mean | month_hour_residual | 8713 | -0.1540 | -0.1567 | 0.1567 |
| current_utc_plus8 | all | renewable_error | v100_mean | month_hour_residual | 8713 | -0.1252 | -0.0809 | 0.0809 |
| current_utc_plus8 | all | renewable_error | ghi_mean | month_hour_residual | 8713 | 0.1370 | 0.0766 | 0.0766 |
| current_utc_plus8 | all | renewable_error | tcc_mean | month_hour_residual | 8713 | -0.0715 | -0.0477 | 0.0477 |
| current_utc_plus8 | all | renewable_error | ghi_max | month_hour_residual | 8713 | 0.0663 | 0.0356 | 0.0356 |
| current_utc_plus8 | all | solar_error | wind_speed_mean | month_hour_residual | 8713 | 0.2528 | 0.2938 | 0.2938 |
| current_utc_plus8 | all | solar_error | u100_mean | month_hour_residual | 8713 | 0.2200 | 0.2296 | 0.2296 |
| current_utc_plus8 | all | solar_error | wind_speed_max | month_hour_residual | 8713 | 0.1658 | 0.2069 | 0.2069 |
| current_utc_plus8 | all | solar_error | ghi_mean | month_hour_residual | 8713 | 0.2682 | 0.1303 | 0.1303 |
| current_utc_plus8 | all | solar_error | sp_mean | month_hour_residual | 8713 | -0.0661 | -0.1230 | 0.1230 |
| current_utc_plus8 | all | solar_error | tcc_mean | month_hour_residual | 8713 | -0.1579 | -0.0736 | 0.0736 |
| current_utc_plus8 | all | solar_error | v100_mean | month_hour_residual | 8713 | -0.0802 | -0.0401 | 0.0401 |
| current_utc_plus8 | all | solar_error | t2m_mean | month_hour_residual | 8713 | -0.0402 | -0.0240 | 0.0240 |
| current_utc_plus8 | all | wind_error | wind_speed_mean | month_hour_residual | 8713 | 0.4404 | 0.3702 | 0.3702 |
| current_utc_plus8 | all | wind_error | wind_speed_max | month_hour_residual | 8713 | 0.3073 | 0.2740 | 0.2740 |
| current_utc_plus8 | all | wind_error | u100_mean | month_hour_residual | 8713 | 0.2991 | 0.2344 | 0.2344 |
| current_utc_plus8 | all | wind_error | sp_mean | month_hour_residual | 8713 | -0.1572 | -0.1453 | 0.1453 |
| current_utc_plus8 | all | wind_error | v100_mean | month_hour_residual | 8713 | -0.1125 | -0.0733 | 0.0733 |
| current_utc_plus8 | all | wind_error | t2m_mean | month_hour_residual | 8713 | -0.0641 | -0.0319 | 0.0319 |
| current_utc_plus8 | all | wind_error | ghi_max | month_hour_residual | 8713 | 0.0485 | 0.0315 | 0.0315 |
| current_utc_plus8 | all | wind_error | tcc_mean | month_hour_residual | 8713 | 0.0000 | 0.0113 | 0.0113 |
| current_utc_plus8 | daylight | bid_space_error | wind_speed_mean | month_hour_residual | 4343 | -0.5314 | -0.4767 | 0.4767 |
| current_utc_plus8 | daylight | bid_space_error | u100_mean | month_hour_residual | 4343 | -0.3841 | -0.3413 | 0.3413 |
| current_utc_plus8 | daylight | bid_space_error | wind_speed_max | month_hour_residual | 4343 | -0.3292 | -0.2954 | 0.2954 |
| current_utc_plus8 | daylight | bid_space_error | v100_mean | month_hour_residual | 4343 | 0.2192 | 0.1652 | 0.1652 |
| current_utc_plus8 | daylight | bid_space_error | sp_mean | month_hour_residual | 4343 | 0.1094 | 0.1203 | 0.1203 |
| current_utc_plus8 | daylight | bid_space_error | tcc_mean | month_hour_residual | 4343 | 0.1158 | 0.1129 | 0.1129 |
| current_utc_plus8 | daylight | bid_space_error | ghi_mean | month_hour_residual | 4343 | -0.1285 | -0.1062 | 0.1062 |
| current_utc_plus8 | daylight | bid_space_error | t2m_mean | month_hour_residual | 4343 | 0.1293 | 0.0795 | 0.0795 |
| current_utc_plus8 | daylight | renewable_error | wind_speed_mean | month_hour_residual | 4343 | 0.5250 | 0.4754 | 0.4754 |
| current_utc_plus8 | daylight | renewable_error | u100_mean | month_hour_residual | 4343 | 0.4253 | 0.3597 | 0.3597 |
| current_utc_plus8 | daylight | renewable_error | wind_speed_max | month_hour_residual | 4343 | 0.3402 | 0.3059 | 0.3059 |
| current_utc_plus8 | daylight | renewable_error | tcc_mean | month_hour_residual | 4343 | -0.1747 | -0.1622 | 0.1622 |
| current_utc_plus8 | daylight | renewable_error | sp_mean | month_hour_residual | 4343 | -0.1403 | -0.1441 | 0.1441 |
| current_utc_plus8 | daylight | renewable_error | ghi_mean | month_hour_residual | 4343 | 0.1602 | 0.1359 | 0.1359 |
| current_utc_plus8 | daylight | renewable_error | v100_mean | month_hour_residual | 4343 | -0.1694 | -0.1242 | 0.1242 |
| current_utc_plus8 | daylight | renewable_error | ghi_max | month_hour_residual | 4343 | 0.0818 | 0.0708 | 0.0708 |
| current_utc_plus8 | daylight | solar_error | wind_speed_mean | month_hour_residual | 4343 | 0.3181 | 0.3221 | 0.3221 |
| current_utc_plus8 | daylight | solar_error | u100_mean | month_hour_residual | 4343 | 0.2937 | 0.2692 | 0.2692 |
| current_utc_plus8 | daylight | solar_error | tcc_mean | month_hour_residual | 4343 | -0.2403 | -0.2173 | 0.2173 |
| current_utc_plus8 | daylight | solar_error | ghi_mean | month_hour_residual | 4343 | 0.2696 | 0.2109 | 0.2109 |
| current_utc_plus8 | daylight | solar_error | wind_speed_max | month_hour_residual | 4343 | 0.2142 | 0.2096 | 0.2096 |
| current_utc_plus8 | daylight | solar_error | v100_mean | month_hour_residual | 4343 | -0.1159 | -0.1051 | 0.1051 |
| current_utc_plus8 | daylight | solar_error | sp_mean | month_hour_residual | 4343 | -0.0788 | -0.0866 | 0.0866 |
| current_utc_plus8 | daylight | solar_error | t2m_mean | month_hour_residual | 4343 | -0.0544 | -0.0470 | 0.0470 |
| current_utc_plus8 | daylight | wind_error | wind_speed_mean | month_hour_residual | 4343 | 0.5054 | 0.4323 | 0.4323 |
| current_utc_plus8 | daylight | wind_error | u100_mean | month_hour_residual | 4343 | 0.3827 | 0.3134 | 0.3134 |
| current_utc_plus8 | daylight | wind_error | wind_speed_max | month_hour_residual | 4343 | 0.3216 | 0.2792 | 0.2792 |
| current_utc_plus8 | daylight | wind_error | sp_mean | month_hour_residual | 4343 | -0.1397 | -0.1290 | 0.1290 |
| current_utc_plus8 | daylight | wind_error | v100_mean | month_hour_residual | 4343 | -0.1532 | -0.1065 | 0.1065 |
| current_utc_plus8 | daylight | wind_error | t2m_mean | month_hour_residual | 4343 | -0.1100 | -0.0781 | 0.0781 |
| current_utc_plus8 | daylight | wind_error | ghi_max | month_hour_residual | 4343 | 0.0671 | 0.0618 | 0.0618 |
| current_utc_plus8 | daylight | wind_error | tcc_mean | month_hour_residual | 4343 | -0.0683 | -0.0529 | 0.0529 |
| current_utc_plus8 | midday_10_17 | bid_space_error | wind_speed_mean | month_hour_residual | 2904 | -0.5585 | -0.5186 | 0.5186 |
| current_utc_plus8 | midday_10_17 | bid_space_error | u100_mean | month_hour_residual | 2904 | -0.4338 | -0.4025 | 0.4025 |
| current_utc_plus8 | midday_10_17 | bid_space_error | wind_speed_max | month_hour_residual | 2904 | -0.3500 | -0.3204 | 0.3204 |
| current_utc_plus8 | midday_10_17 | bid_space_error | v100_mean | month_hour_residual | 2904 | 0.2259 | 0.1765 | 0.1765 |
| current_utc_plus8 | midday_10_17 | bid_space_error | tcc_mean | month_hour_residual | 2904 | 0.1629 | 0.1664 | 0.1664 |
| current_utc_plus8 | midday_10_17 | bid_space_error | ghi_mean | month_hour_residual | 2904 | -0.1442 | -0.1345 | 0.1345 |
| current_utc_plus8 | midday_10_17 | bid_space_error | sp_mean | month_hour_residual | 2904 | 0.1157 | 0.1318 | 0.1318 |
| current_utc_plus8 | midday_10_17 | bid_space_error | t2m_mean | month_hour_residual | 2904 | 0.1416 | 0.0924 | 0.0924 |
| current_utc_plus8 | midday_10_17 | renewable_error | wind_speed_mean | month_hour_residual | 2904 | 0.5614 | 0.5314 | 0.5314 |
| current_utc_plus8 | midday_10_17 | renewable_error | u100_mean | month_hour_residual | 2904 | 0.4788 | 0.4272 | 0.4272 |
| current_utc_plus8 | midday_10_17 | renewable_error | wind_speed_max | month_hour_residual | 2904 | 0.3637 | 0.3377 | 0.3377 |
| current_utc_plus8 | midday_10_17 | renewable_error | tcc_mean | month_hour_residual | 2904 | -0.2271 | -0.2196 | 0.2196 |
| current_utc_plus8 | midday_10_17 | renewable_error | ghi_mean | month_hour_residual | 2904 | 0.1769 | 0.1663 | 0.1663 |
| current_utc_plus8 | midday_10_17 | renewable_error | sp_mean | month_hour_residual | 2904 | -0.1439 | -0.1522 | 0.1522 |
| current_utc_plus8 | midday_10_17 | renewable_error | v100_mean | month_hour_residual | 2904 | -0.1831 | -0.1440 | 0.1440 |
| current_utc_plus8 | midday_10_17 | renewable_error | t2m_mean | month_hour_residual | 2904 | -0.1252 | -0.0943 | 0.0943 |

## Readout For Next Experiments

- Weather alignment itself is not the likely reason previous full-weather features underperformed; the `UTC+8` mapping is already implemented.
- The useful weather signal is more likely in selective use: wind-speed/GHI/cloud residuals as confidence, rerank, or pair-level features rather than blindly adding all weather aggregates to the point price model.
- Because `holiday_only` beat local winter validation but lost online, promote future weather changes only if they improve broad 5-fold stability or are validated by an online submit.

## Artifacts

- `reports/weather_alignment_metadata.csv`
- `reports/weather_alignment_coverage.csv`
- `reports/weather_alignment_signal_summary.csv`
- `reports/weather_alignment_top_correlations.csv`
