# Weather vs Renewable Diagnostics

This report checks whether official NWP forecasts can explain renewable forecast value and forecast error.

## Time Alignment

- Competition document says each `.nc` file is published on date `D` and forecasts date `D+1`.
- The `.nc` coordinate `time` is UTC. Example: `20250101.nc` has `time=2025-01-01 16:00 UTC`, which equals `2025-01-02 00:00` Beijing time.
- `lead_time=0..23` maps to Beijing hours `00:00..23:00` of the target day.
- Boundary data is 15-minute Beijing time. For this diagnostic, 15-minute renewable values are averaged to hourly values before joining NWP.

## Units

| variable | unit |
| --- | --- |
| u100 | m/s |
| v100 | m/s |
| t2m | K |
| tp | m |
| tcc | 0-1 |
| sp | Pa |
| ghi | W/m^2 |
| wind_speed | m/s |

## Join Coverage

- Joined hourly rows: `8713`.
- Date range: `2025-01-02` to `2025-12-31`.
- Unique days: `364`.
- 2025-01-01 is expected to be absent because the matching NWP file would be 2024-12-31, which is not provided locally.

## Top Weather Features For Renewable Forecast Error

Target is `renewable_error = renewable_forecast - renewable_actual`.
Positive correlation means larger weather feature values tend to come with more over-forecasting of wind+solar output.

| scope | target | feature | rows | pearson | spearman | abs_spearman |
| --- | --- | --- | --- | --- | --- | --- |
| all | renewable_error | ghi_mean | 8713 | 0.5605 | 0.5109 | 0.5109 |
| all | renewable_error | ghi_max | 8713 | 0.5264 | 0.4940 | 0.4940 |
| all | renewable_error | u100_mean | 8713 | 0.3518 | 0.2880 | 0.2880 |
| all | renewable_error | wind_speed_max | 8713 | 0.3079 | 0.2704 | 0.2704 |
| all | renewable_error | wind_speed_mean | 8713 | 0.3336 | 0.2551 | 0.2551 |
| all | renewable_error | v100_mean | 8713 | -0.1564 | -0.1029 | 0.1029 |
| all | renewable_error | tp_mean | 8713 | -0.1025 | -0.0766 | 0.0766 |
| all | renewable_error | tcc_mean | 8713 | -0.1135 | -0.0676 | 0.0676 |
| all | renewable_error | t2m_mean | 8713 | 0.0026 | 0.0365 | 0.0365 |
| all | renewable_error | sp_mean | 8713 | -0.0185 | -0.0355 | 0.0355 |
| all | renewable_error | tcc_max | 8713 | 0.0190 | -0.0002 | 0.0002 |
| daylight | renewable_error | ghi_mean | 4343 | 0.4285 | 0.4894 | 0.4894 |
| daylight | renewable_error | ghi_max | 4343 | 0.3557 | 0.4145 | 0.4145 |
| daylight | renewable_error | u100_mean | 4343 | 0.4568 | 0.3972 | 0.3972 |
| daylight | renewable_error | wind_speed_max | 4343 | 0.4045 | 0.3583 | 0.3583 |
| daylight | renewable_error | wind_speed_mean | 4343 | 0.4145 | 0.3402 | 0.3402 |
| daylight | renewable_error | tcc_mean | 4343 | -0.2435 | -0.2155 | 0.2155 |
| daylight | renewable_error | tp_mean | 4343 | -0.1874 | -0.1904 | 0.1904 |
| daylight | renewable_error | t2m_mean | 4343 | -0.1870 | -0.1519 | 0.1519 |
| daylight | renewable_error | sp_mean | 4343 | 0.0948 | 0.0998 | 0.0998 |
| daylight | renewable_error | v100_mean | 4343 | -0.1414 | -0.0890 | 0.0890 |
| daylight | renewable_error | tcc_max | 4343 | 0.0328 | 0.0145 | 0.0145 |
| midday_10_17 | renewable_error | u100_mean | 2904 | 0.4718 | 0.4319 | 0.4319 |
| midday_10_17 | renewable_error | wind_speed_mean | 2904 | 0.4646 | 0.4246 | 0.4246 |
| midday_10_17 | renewable_error | wind_speed_max | 2904 | 0.4271 | 0.3914 | 0.3914 |
| midday_10_17 | renewable_error | tcc_mean | 2904 | -0.2674 | -0.2511 | 0.2511 |
| midday_10_17 | renewable_error | ghi_mean | 2904 | 0.1969 | 0.2412 | 0.2412 |
| midday_10_17 | renewable_error | t2m_mean | 2904 | -0.1987 | -0.1664 | 0.1664 |
| midday_10_17 | renewable_error | tp_mean | 2904 | -0.1818 | -0.1590 | 0.1590 |
| midday_10_17 | renewable_error | ghi_max | 2904 | 0.0681 | 0.1077 | 0.1077 |
| midday_10_17 | renewable_error | v100_mean | 2904 | -0.1543 | -0.1047 | 0.1047 |
| midday_10_17 | renewable_error | sp_mean | 2904 | 0.0434 | 0.0308 | 0.0308 |
| midday_10_17 | renewable_error | tcc_max | 2904 | 0.0282 | 0.0211 | 0.0211 |
| spring_02_05 | renewable_error | ghi_mean | 2880 | 0.6336 | 0.5563 | 0.5563 |
| spring_02_05 | renewable_error | ghi_max | 2880 | 0.5983 | 0.5414 | 0.5414 |
| spring_02_05 | renewable_error | wind_speed_mean | 2880 | 0.3996 | 0.3501 | 0.3501 |
| spring_02_05 | renewable_error | wind_speed_max | 2880 | 0.3534 | 0.3436 | 0.3436 |
| spring_02_05 | renewable_error | u100_mean | 2880 | 0.3461 | 0.2879 | 0.2879 |
| spring_02_05 | renewable_error | v100_mean | 2880 | -0.2289 | -0.1870 | 0.1870 |
| spring_02_05 | renewable_error | t2m_mean | 2880 | 0.1495 | 0.1787 | 0.1787 |
| spring_02_05 | renewable_error | sp_mean | 2880 | -0.1133 | -0.1388 | 0.1388 |
| spring_02_05 | renewable_error | tp_mean | 2880 | 0.0128 | 0.0689 | 0.0689 |
| spring_02_05 | renewable_error | tcc_mean | 2880 | -0.0457 | 0.0068 | 0.0068 |
| spring_02_05 | renewable_error | tcc_max | 2880 | nan | nan | nan |
| winter_11_12 | renewable_error | ghi_mean | 1464 | 0.7345 | 0.5941 | 0.5941 |
| winter_11_12 | renewable_error | ghi_max | 1464 | 0.6911 | 0.5835 | 0.5835 |
| winter_11_12 | renewable_error | wind_speed_max | 1464 | 0.3258 | 0.3681 | 0.3681 |
| winter_11_12 | renewable_error | u100_mean | 1464 | 0.3338 | 0.3677 | 0.3677 |
| winter_11_12 | renewable_error | wind_speed_mean | 1464 | 0.2973 | 0.3493 | 0.3493 |
| winter_11_12 | renewable_error | t2m_mean | 1464 | 0.3011 | 0.3178 | 0.3178 |
| winter_11_12 | renewable_error | sp_mean | 1464 | -0.1306 | -0.1801 | 0.1801 |
| winter_11_12 | renewable_error | tp_mean | 1464 | 0.1173 | 0.1459 | 0.1459 |
| winter_11_12 | renewable_error | tcc_mean | 1464 | 0.0256 | 0.0944 | 0.0944 |
| winter_11_12 | renewable_error | v100_mean | 1464 | -0.0241 | -0.0120 | 0.0120 |
| winter_11_12 | renewable_error | tcc_max | 1464 | nan | nan | nan |

## Top Weather Features After Removing Month-Hour Mean

This is stricter: both weather features and renewable error are residualized by month x hour. It tests whether weather adds signal beyond seasonal intraday pattern.

| scope | target | feature | rows | pearson | spearman | abs_spearman |
| --- | --- | --- | --- | --- | --- | --- |
| all | renewable_error_mh_resid | wind_speed_mean_mh_resid | 8713 | 0.4622 | 0.4019 | 0.4019 |
| all | renewable_error_mh_resid | wind_speed_max_mh_resid | 8713 | 0.3177 | 0.2898 | 0.2898 |
| all | renewable_error_mh_resid | u100_mean_mh_resid | 8713 | 0.3357 | 0.2672 | 0.2672 |
| all | renewable_error_mh_resid | sp_mean_mh_resid | 8713 | -0.1540 | -0.1567 | 0.1567 |
| all | renewable_error_mh_resid | v100_mean_mh_resid | 8713 | -0.1252 | -0.0809 | 0.0809 |
| all | renewable_error_mh_resid | ghi_mean_mh_resid | 8713 | 0.1370 | 0.0766 | 0.0766 |
| all | renewable_error_mh_resid | tcc_mean_mh_resid | 8713 | -0.0715 | -0.0477 | 0.0477 |
| all | renewable_error_mh_resid | ghi_max_mh_resid | 8713 | 0.0663 | 0.0356 | 0.0356 |
| all | renewable_error_mh_resid | t2m_mean_mh_resid | 8713 | -0.0688 | -0.0286 | 0.0286 |
| all | renewable_error_mh_resid | tcc_max_mh_resid | 8713 | 0.0097 | -0.0056 | 0.0056 |
| all | renewable_error_mh_resid | tp_mean_mh_resid | 8713 | 0.0217 | -0.0014 | 0.0014 |
| daylight | renewable_error_mh_resid | wind_speed_mean_mh_resid | 4343 | 0.5250 | 0.4754 | 0.4754 |
| daylight | renewable_error_mh_resid | u100_mean_mh_resid | 4343 | 0.4253 | 0.3597 | 0.3597 |
| daylight | renewable_error_mh_resid | wind_speed_max_mh_resid | 4343 | 0.3402 | 0.3059 | 0.3059 |
| daylight | renewable_error_mh_resid | tcc_mean_mh_resid | 4343 | -0.1747 | -0.1622 | 0.1622 |
| daylight | renewable_error_mh_resid | sp_mean_mh_resid | 4343 | -0.1403 | -0.1441 | 0.1441 |
| daylight | renewable_error_mh_resid | ghi_mean_mh_resid | 4343 | 0.1602 | 0.1359 | 0.1359 |
| daylight | renewable_error_mh_resid | v100_mean_mh_resid | 4343 | -0.1694 | -0.1242 | 0.1242 |
| daylight | renewable_error_mh_resid | ghi_max_mh_resid | 4343 | 0.0818 | 0.0708 | 0.0708 |
| daylight | renewable_error_mh_resid | t2m_mean_mh_resid | 4343 | -0.1065 | -0.0696 | 0.0696 |
| daylight | renewable_error_mh_resid | tp_mean_mh_resid | 4343 | 0.0095 | -0.0283 | 0.0283 |
| daylight | renewable_error_mh_resid | tcc_max_mh_resid | 4343 | 0.0113 | -0.0010 | 0.0010 |
| midday_10_17 | renewable_error_mh_resid | wind_speed_mean_mh_resid | 2904 | 0.5614 | 0.5314 | 0.5314 |
| midday_10_17 | renewable_error_mh_resid | u100_mean_mh_resid | 2904 | 0.4788 | 0.4272 | 0.4272 |
| midday_10_17 | renewable_error_mh_resid | wind_speed_max_mh_resid | 2904 | 0.3637 | 0.3377 | 0.3377 |
| midday_10_17 | renewable_error_mh_resid | tcc_mean_mh_resid | 2904 | -0.2271 | -0.2196 | 0.2196 |
| midday_10_17 | renewable_error_mh_resid | ghi_mean_mh_resid | 2904 | 0.1769 | 0.1663 | 0.1663 |
| midday_10_17 | renewable_error_mh_resid | sp_mean_mh_resid | 2904 | -0.1439 | -0.1522 | 0.1522 |
| midday_10_17 | renewable_error_mh_resid | v100_mean_mh_resid | 2904 | -0.1831 | -0.1440 | 0.1440 |
| midday_10_17 | renewable_error_mh_resid | t2m_mean_mh_resid | 2904 | -0.1252 | -0.0943 | 0.0943 |
| midday_10_17 | renewable_error_mh_resid | ghi_max_mh_resid | 2904 | 0.0955 | 0.0856 | 0.0856 |
| midday_10_17 | renewable_error_mh_resid | tp_mean_mh_resid | 2904 | 0.0041 | -0.0308 | 0.0308 |
| midday_10_17 | renewable_error_mh_resid | tcc_max_mh_resid | 2904 | 0.0018 | -0.0007 | 0.0007 |
| spring_02_05 | renewable_error_mh_resid | wind_speed_mean_mh_resid | 2880 | 0.5590 | 0.5136 | 0.5136 |
| spring_02_05 | renewable_error_mh_resid | wind_speed_max_mh_resid | 2880 | 0.4207 | 0.4002 | 0.4002 |
| spring_02_05 | renewable_error_mh_resid | u100_mean_mh_resid | 2880 | 0.4008 | 0.3743 | 0.3743 |
| spring_02_05 | renewable_error_mh_resid | v100_mean_mh_resid | 2880 | -0.2310 | -0.2094 | 0.2094 |
| spring_02_05 | renewable_error_mh_resid | sp_mean_mh_resid | 2880 | -0.1665 | -0.1742 | 0.1742 |
| spring_02_05 | renewable_error_mh_resid | tcc_mean_mh_resid | 2880 | -0.0827 | -0.0699 | 0.0699 |
| spring_02_05 | renewable_error_mh_resid | t2m_mean_mh_resid | 2880 | -0.0953 | -0.0531 | 0.0531 |
| spring_02_05 | renewable_error_mh_resid | ghi_mean_mh_resid | 2880 | 0.0506 | -0.0131 | 0.0131 |
| spring_02_05 | renewable_error_mh_resid | tp_mean_mh_resid | 2880 | -0.0078 | 0.0097 | 0.0097 |
| spring_02_05 | renewable_error_mh_resid | ghi_max_mh_resid | 2880 | 0.0050 | -0.0057 | 0.0057 |
| spring_02_05 | renewable_error_mh_resid | tcc_max_mh_resid | 2880 | nan | nan | nan |
| winter_11_12 | renewable_error_mh_resid | wind_speed_mean_mh_resid | 1464 | 0.5832 | 0.6070 | 0.6070 |
| winter_11_12 | renewable_error_mh_resid | u100_mean_mh_resid | 1464 | 0.5137 | 0.5111 | 0.5111 |
| winter_11_12 | renewable_error_mh_resid | wind_speed_max_mh_resid | 1464 | 0.4466 | 0.4531 | 0.4531 |
| winter_11_12 | renewable_error_mh_resid | sp_mean_mh_resid | 1464 | -0.2215 | -0.2598 | 0.2598 |
| winter_11_12 | renewable_error_mh_resid | tp_mean_mh_resid | 1464 | 0.1533 | 0.1611 | 0.1611 |
| winter_11_12 | renewable_error_mh_resid | tcc_mean_mh_resid | 1464 | 0.0509 | 0.0992 | 0.0992 |
| winter_11_12 | renewable_error_mh_resid | ghi_max_mh_resid | 1464 | -0.1137 | -0.0907 | 0.0907 |
| winter_11_12 | renewable_error_mh_resid | t2m_mean_mh_resid | 1464 | -0.1258 | -0.0903 | 0.0903 |
| winter_11_12 | renewable_error_mh_resid | ghi_mean_mh_resid | 1464 | 0.0048 | -0.0797 | 0.0797 |
| winter_11_12 | renewable_error_mh_resid | v100_mean_mh_resid | 1464 | 0.0208 | 0.0283 | 0.0283 |
| winter_11_12 | renewable_error_mh_resid | tcc_max_mh_resid | 1464 | nan | nan | nan |

## Top Weather Features For Renewable Actual Output

| scope | target | feature | rows | pearson | spearman | abs_spearman |
| --- | --- | --- | --- | --- | --- | --- |
| all | renewable_actual | u100_mean | 8713 | 0.5615 | 0.5655 | 0.5655 |
| all | renewable_actual | ghi_mean | 8713 | 0.5731 | 0.5269 | 0.5269 |
| all | renewable_actual | ghi_max | 8713 | 0.5461 | 0.5073 | 0.5073 |
| all | renewable_actual | wind_speed_mean | 8713 | 0.4762 | 0.4730 | 0.4730 |
| all | renewable_actual | wind_speed_max | 8713 | 0.4171 | 0.4308 | 0.4308 |
| all | renewable_actual | tcc_mean | 8713 | -0.2271 | -0.2300 | 0.2300 |
| all | renewable_actual | tp_mean | 8713 | -0.2077 | -0.2129 | 0.2129 |
| all | renewable_actual | v100_mean | 8713 | -0.1280 | -0.0881 | 0.0881 |
| all | renewable_actual | t2m_mean | 8713 | -0.0555 | -0.0635 | 0.0635 |
| all | renewable_actual | tcc_max | 8713 | 0.0047 | -0.0198 | 0.0198 |
| daylight | renewable_actual | u100_mean | 4343 | 0.5685 | 0.5876 | 0.5876 |
| daylight | renewable_actual | wind_speed_mean | 4343 | 0.4766 | 0.4969 | 0.4969 |
| daylight | renewable_actual | ghi_mean | 4343 | 0.5013 | 0.4892 | 0.4892 |
| daylight | renewable_actual | wind_speed_max | 4343 | 0.4427 | 0.4635 | 0.4635 |
| daylight | renewable_actual | ghi_max | 4343 | 0.4201 | 0.4008 | 0.4008 |
| daylight | renewable_actual | tcc_mean | 4343 | -0.3456 | -0.3559 | 0.3559 |
| daylight | renewable_actual | tp_mean | 4343 | -0.3153 | -0.3462 | 0.3462 |
| daylight | renewable_actual | t2m_mean | 4343 | -0.2226 | -0.2387 | 0.2387 |
| daylight | renewable_actual | v100_mean | 4343 | -0.1556 | -0.1318 | 0.1318 |
| daylight | renewable_actual | sp_mean | 4343 | 0.1031 | 0.1252 | 0.1252 |
| midday_10_17 | renewable_actual | u100_mean | 2904 | 0.5686 | 0.5984 | 0.5984 |
| midday_10_17 | renewable_actual | wind_speed_mean | 2904 | 0.5277 | 0.5690 | 0.5690 |
| midday_10_17 | renewable_actual | wind_speed_max | 2904 | 0.4491 | 0.4783 | 0.4783 |
| midday_10_17 | renewable_actual | tcc_mean | 2904 | -0.3681 | -0.3621 | 0.3621 |
| midday_10_17 | renewable_actual | tp_mean | 2904 | -0.3254 | -0.3149 | 0.3149 |
| midday_10_17 | renewable_actual | ghi_mean | 2904 | 0.2788 | 0.2397 | 0.2397 |
| midday_10_17 | renewable_actual | t2m_mean | 2904 | -0.1737 | -0.2016 | 0.2016 |
| midday_10_17 | renewable_actual | v100_mean | 2904 | -0.1826 | -0.1633 | 0.1633 |
| midday_10_17 | renewable_actual | ghi_max | 2904 | 0.1129 | 0.0843 | 0.0843 |
| midday_10_17 | renewable_actual | sp_mean | 2904 | 0.0122 | 0.0393 | 0.0393 |
| spring_02_05 | renewable_actual | ghi_mean | 2880 | 0.7018 | 0.6720 | 0.6720 |
| spring_02_05 | renewable_actual | ghi_max | 2880 | 0.6873 | 0.6582 | 0.6582 |
| spring_02_05 | renewable_actual | u100_mean | 2880 | 0.4933 | 0.5072 | 0.5072 |
| spring_02_05 | renewable_actual | wind_speed_mean | 2880 | 0.4838 | 0.4852 | 0.4852 |
| spring_02_05 | renewable_actual | wind_speed_max | 2880 | 0.3907 | 0.4001 | 0.4001 |
| spring_02_05 | renewable_actual | t2m_mean | 2880 | 0.2884 | 0.2784 | 0.2784 |
| spring_02_05 | renewable_actual | sp_mean | 2880 | -0.2231 | -0.2109 | 0.2109 |
| spring_02_05 | renewable_actual | v100_mean | 2880 | -0.1598 | -0.1517 | 0.1517 |
| spring_02_05 | renewable_actual | tcc_mean | 2880 | -0.0608 | -0.0688 | 0.0688 |
| spring_02_05 | renewable_actual | tp_mean | 2880 | -0.0437 | -0.0604 | 0.0604 |
| winter_11_12 | renewable_actual | ghi_mean | 1464 | 0.5989 | 0.5857 | 0.5857 |
| winter_11_12 | renewable_actual | ghi_max | 1464 | 0.5744 | 0.5669 | 0.5669 |
| winter_11_12 | renewable_actual | u100_mean | 1464 | 0.5369 | 0.5231 | 0.5231 |
| winter_11_12 | renewable_actual | wind_speed_max | 1464 | 0.3912 | 0.3866 | 0.3866 |
| winter_11_12 | renewable_actual | wind_speed_mean | 1464 | 0.4018 | 0.3788 | 0.3788 |
| winter_11_12 | renewable_actual | t2m_mean | 1464 | 0.3050 | 0.3599 | 0.3599 |
| winter_11_12 | renewable_actual | sp_mean | 1464 | -0.0887 | -0.1238 | 0.1238 |
| winter_11_12 | renewable_actual | v100_mean | 1464 | 0.0552 | 0.0671 | 0.0671 |
| winter_11_12 | renewable_actual | tp_mean | 1464 | 0.1171 | -0.0146 | 0.0146 |
| winter_11_12 | renewable_actual | tcc_mean | 1464 | -0.0595 | -0.0072 | 0.0072 |

## Midday Month-Segment Bias With Weather Means

| month | segment | hours | renewable_error_mean | renewable_abs_error_mean | solar_error_mean | wind_error_mean | ghi_mean | ghi_max | tcc_mean | tcc_max | tp_mean | wind_speed_mean | wind_speed_max | u100_mean | v100_mean | t2m_mean | sp_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 10_14 | 120 | 0.9254 | 0.9254 | 0.4468 | 0.4786 | 354.7637 | 520.1589 | 0.2487 | 1.0000 | 0.0005 | 6.1034 | 19.0693 | 3.9650 | -0.1444 | 273.9114 | 86605.9390 |
| 12 | 10_14 | 124 | 0.7903 | 0.7903 | 0.4474 | 0.3430 | 283.6632 | 448.2286 | 0.3105 | 1.0000 | 0.0002 | 5.1495 | 17.6599 | 3.0921 | 0.0680 | 267.0644 | 86408.7718 |
| 3 | 10_14 | 124 | 0.6498 | 0.6506 | 0.4133 | 0.2365 | 562.9423 | 746.4554 | 0.3958 | 1.0000 | 0.0005 | 6.0762 | 17.0831 | 2.8485 | -1.0093 | 275.9402 | 86333.3067 |
| 9 | 10_14 | 116 | 0.6111 | 0.6111 | 0.4192 | 0.1919 | 554.1849 | 745.9984 | 0.4233 | 1.0000 | 0.0013 | 4.8076 | 15.9998 | 1.2783 | 0.6742 | 291.1267 | 86164.9291 |
| 2 | 10_14 | 112 | 0.6039 | 0.6039 | 0.4043 | 0.1996 | 439.0484 | 599.0226 | 0.2386 | 1.0000 | 0.0001 | 5.0867 | 16.3704 | 2.6499 | -0.6448 | 266.0901 | 86543.5238 |
| 4 | 10_14 | 120 | 0.5745 | 0.5817 | 0.3678 | 0.2066 | 672.8449 | 871.9716 | 0.4200 | 1.0000 | 0.0006 | 6.8973 | 18.1938 | 3.5206 | -1.8930 | 285.2436 | 85960.3852 |
| 11 | 14_18 | 120 | 0.5619 | 0.5689 | 0.2071 | 0.3548 | 283.0726 | 520.1200 | 0.2567 | 1.0000 | 0.0005 | 6.6156 | 19.7777 | 4.2673 | -0.3767 | 275.9186 | 86477.5324 |
| 3 | 14_18 | 124 | 0.5206 | 0.5264 | 0.2867 | 0.2339 | 553.9831 | 827.4170 | 0.4315 | 1.0000 | 0.0005 | 7.0329 | 17.7780 | 3.7093 | -1.6390 | 278.8753 | 86166.4648 |
| 5 | 10_14 | 124 | 0.5039 | 0.5041 | 0.2964 | 0.2075 | 707.3722 | 927.8317 | 0.4866 | 1.0000 | 0.0013 | 6.5169 | 17.3183 | 2.1053 | -1.0511 | 291.0066 | 85843.4795 |
| 6 | 10_14 | 120 | 0.4905 | 0.4993 | 0.3843 | 0.1062 | 695.2602 | 914.2997 | 0.5459 | 1.0000 | 0.0015 | 5.0817 | 14.4153 | 0.8720 | -0.7489 | 296.7358 | 85683.0168 |
| 12 | 14_18 | 124 | 0.4858 | 0.4917 | 0.2391 | 0.2467 | 242.1629 | 462.1324 | 0.3156 | 1.0000 | 0.0002 | 5.4695 | 17.5495 | 3.3037 | -0.2110 | 269.5538 | 86300.6407 |
| 4 | 14_18 | 120 | 0.4510 | 0.4606 | 0.2386 | 0.2124 | 628.5760 | 923.1188 | 0.4533 | 1.0000 | 0.0006 | 7.7699 | 18.5568 | 4.4555 | -2.5798 | 287.9697 | 85790.0829 |
| 9 | 14_18 | 116 | 0.4275 | 0.4332 | 0.2757 | 0.1518 | 499.8853 | 776.4896 | 0.4402 | 1.0000 | 0.0015 | 5.3990 | 15.5612 | 1.8546 | 0.5021 | 293.4646 | 86000.9518 |
| 2 | 14_18 | 112 | 0.4099 | 0.4264 | 0.2825 | 0.1274 | 452.9457 | 675.9499 | 0.2524 | 1.0000 | 0.0001 | 5.7447 | 16.3920 | 3.1677 | -1.1143 | 269.4224 | 86394.3831 |
| 5 | 14_18 | 124 | 0.3842 | 0.3905 | 0.2138 | 0.1704 | 643.0352 | 975.4614 | 0.5429 | 1.0000 | 0.0015 | 6.9911 | 17.5899 | 2.6747 | -1.5157 | 293.3354 | 85679.6157 |
| 8 | 10_14 | 124 | 0.3597 | 0.3617 | 0.2924 | 0.0673 | 577.0461 | 806.3704 | 0.5262 | 1.0000 | 0.0039 | 4.4015 | 13.3648 | 0.8150 | 0.5211 | 296.3453 | 85802.3712 |
| 6 | 14_18 | 120 | 0.3568 | 0.3793 | 0.2694 | 0.0874 | 631.8257 | 953.2844 | 0.5905 | 1.0000 | 0.0017 | 5.4480 | 14.6181 | 1.1726 | -1.0153 | 298.8016 | 85522.8916 |
| 10 | 10_14 | 124 | 0.3558 | 0.3704 | 0.2417 | 0.1141 | 420.0843 | 617.6752 | 0.4717 | 0.9997 | 0.0020 | 4.3143 | 14.5806 | 0.6280 | 0.0500 | 281.2219 | 86686.3536 |

## Readout

- Strong raw correlations can be mostly time-of-day seasonality; residual correlations are more useful for correction.
- If residual weather-error correlation is weak, NWP probably does not add much beyond the provided renewable forecast.
- If GHI/TCC/wind residual correlations are stable in specific months/segments, use them as candidates for renewable-error correction or bid-space reliability features.

## Artifacts

- `reports/weather_hourly_spatial_features.csv`
- `reports/weather_renewable_joined_hourly.csv`
- `reports/weather_renewable_corr.csv`
- `reports/weather_renewable_residual_corr.csv`
- `reports/weather_renewable_month_segment.csv`
