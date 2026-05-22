# Pair E2E Bid-Space Weather Residual Experiment

## Scope

- Model: `lgb_pair_spread_bidspace_weather_resid_5fold`.
- Target: legal pair true spread/profit, one row per legal charge/discharge pair.
- Candidate space: all 3321 legal pairs per day.
- Core inputs: bid_space, selected business ratios/ranks, selected NWP month-hour residuals, pair time features.
- Hardware: AutoDL no-GPU mode; NWP features generated from competition zip into `artifacts/features/nwp_grid_15min.csv`.

## Regime Comparison

| model | regime | days | mean_profit | mean_oracle_ratio | loss_days | mean_abs_charge_gap | mean_abs_discharge_gap | champion_mean_profit | champion_loss_days | profit_delta_vs_champion | loss_day_delta_vs_champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| champion_prior_5fold | standard_09_12 | 120 | 7991.3509 | 0.7037 | 5 | 12.6333 | 9.9750 | 7991.3509 | 5 | 0.0000 | 0 |
| champion_prior_5fold | jan_feb_like | 56 | 14153.4998 | 0.8498 | 0 | 13.7321 | 9.5000 | 14153.4998 | 0 | 0.0000 | 0 |
| champion_prior_5fold | all_5fold | 176 | 9952.0347 | 0.7502 | 5 | 12.9830 | 9.8239 | 9952.0347 | 5 | 0.0000 | 0 |
| pair_e2e_bidspace_weather_resid | standard_09_12 | 120 | 6795.1616 | 0.6094 | 10 | 14.6583 | 11.4667 | 7991.3509 | 5 | -1196.1893 | 5 |
| pair_e2e_bidspace_weather_resid | jan_feb_like | 56 | 12716.1671 | 0.7635 | 0 | 13.0536 | 10.5179 | 14153.4998 | 0 | -1437.3327 | 0 |
| pair_e2e_bidspace_weather_resid | all_5fold | 176 | 8679.1179 | 0.6584 | 10 | 14.1477 | 11.1648 | 9952.0347 | 5 | -1272.9168 | 5 |

## Fold Results

| fold | train_rows | valid_rows | valid_days | train_window_days | train_start | train_end | target_mode | mae | rmse | curve_z_rmse | mean_abs_charge_gap | mean_abs_discharge_gap | avg_profit | oracle_avg_profit | oracle_ratio | avg_regret | loss_days | trade_days | trade_days_ratio | best_iteration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 597780 | 92988 | 28 | 180 | 2025-03-05 00:00:00 | 2025-08-31 23:45:00 | pair_spread | 4.1210 | 6.9442 | 0.0000 | 13.3929 | 8.6786 | 8685.4477 | 14361.0258 | 0.6048 | 5675.5782 | 0 | 28 | 1.0000 | 500 |
| valid_2025_10 | 591138 | 102951 | 31 | 180 | 2025-04-04 00:00:00 | 2025-09-30 23:45:00 | pair_spread | 4.6138 | 6.6176 | 0.0000 | 16.6452 | 16.1935 | 6385.6957 | 13921.9943 | 0.4587 | 7536.2986 | 8 | 31 | 1.0000 | 500 |
| valid_2025_11 | 591138 | 99630 | 30 | 180 | 2025-05-05 00:00:00 | 2025-10-31 23:45:00 | pair_spread | 1.9963 | 2.6335 | 0.0000 | 14.1667 | 7.2000 | 6413.0515 | 8007.9217 | 0.8008 | 1594.8701 | 1 | 30 | 1.0000 | 491 |
| valid_2025_12 | 591138 | 102951 | 31 | 180 | 2025-06-04 00:00:00 | 2025-11-30 23:45:00 | pair_spread | 2.9862 | 4.7765 | 0.0000 | 14.2903 | 13.3871 | 5867.0564 | 10135.5174 | 0.5789 | 4268.4610 | 1 | 31 | 1.0000 | 438 |
| valid_2025_jan_feb | 591138 | 185976 | 56 | 180 | 2025-07-05 00:00:00 | 2025-12-31 23:45:00 | pair_spread | 5.2162 | 8.7608 | 0.0000 | 13.0536 | 10.5179 | 12716.1671 | 16655.0543 | 0.7635 | 3938.8872 | 0 | 56 | 1.0000 | 500 |

## Decision

- Reject this full 3321-pair E2E model as a direct champion replacement.
- It runs successfully after feature slimming, but underperforms current champion on standard 09-12, Jan-Feb-like, and all_5fold regimes.
- The direction is still useful as a reranker candidate: constrain it to champion top-K pairs instead of letting it freely choose among all 3321 pairs.

## Next

- Build top-K pair reranker using the same bid_space/weather residual features.
- Use champion-prior top10 as candidate set; train/evaluate held-out true profit only within top10.
