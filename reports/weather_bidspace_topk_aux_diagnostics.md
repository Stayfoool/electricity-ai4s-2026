# Weather + Bid-Space Top-K Auxiliary Switch

## Scope

- Anchor: current online champion-prior top1 pair.
- Candidate set: champion-prior top10 legal charge/discharge pairs.
- Goal: keep champion price prediction unchanged and only use weather/bid_space as a small auxiliary signal inside top-k.
- Validation: for each held-out fold, choose one simple rule using only other folds.
- No price model retraining and no submit file generation in this experiment.

## Rule Family

- Gate: switch is allowed only when top1 confidence is low.
- Scores: bid_space, net_load, cloud/GHI residual, wind residual, and conservative combinations with expert support.
- Constraints: training changed rate <= 10%, training lift >= 5/day, loss days not above anchor, and negative changed days not above positive changed days.

## Overall

| regime | method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank | profit_delta_vs_anchor | loss_day_delta_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_5fold | oracle_top10 | 176 | 10689.6177 | -9394.2116 | 2 | 737.5830 | 131 | 0 | 131 | 4.8977 | 737.5830 | -2 |
| all_5fold | anchor_top1 | 176 | 9952.0347 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 | 0.0000 | 0 |
| all_5fold | weather_bidspace_topk_aux | 176 | 9941.8242 | -15989.5246 | 4 | -10.2105 | 4 | 3 | 8 | 1.0909 | -10.2105 | 0 |

## Regime Summary

| regime | method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard_09_12 | anchor_top1 | 120 | 7991.3509 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| standard_09_12 | oracle_top10 | 120 | 8709.5839 | -9394.2116 | 2 | 718.2329 | 93 | 0 | 93 | 5.3000 |
| standard_09_12 | weather_bidspace_topk_aux | 120 | 7976.1104 | -15989.5246 | 4 | -15.2405 | 3 | 3 | 6 | 1.1083 |
| jan_feb_like | anchor_top1 | 56 | 14153.4998 | 8.7837 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| jan_feb_like | oracle_top10 | 56 | 14932.5473 | 8.7837 | 0 | 779.0475 | 38 | 0 | 38 | 4.0357 |
| jan_feb_like | weather_bidspace_topk_aux | 56 | 14154.0679 | 8.7837 | 0 | 0.5681 | 1 | 0 | 2 | 1.0536 |
| winter_11_12_jan_feb | anchor_top1 | 117 | 10170.8882 | -12550.1225 | 1 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| winter_11_12_jan_feb | oracle_top10 | 117 | 10758.6165 | -9394.2116 | 1 | 587.7283 | 81 | 0 | 81 | 4.2906 |
| winter_11_12_jan_feb | weather_bidspace_topk_aux | 117 | 10157.1603 | -12550.1225 | 1 | -13.7279 | 3 | 1 | 5 | 1.0769 |
| all_5fold | anchor_top1 | 176 | 9952.0347 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| all_5fold | oracle_top10 | 176 | 10689.6177 | -9394.2116 | 2 | 737.5830 | 131 | 0 | 131 | 4.8977 |
| all_5fold | weather_bidspace_topk_aux | 176 | 9941.8242 | -15989.5246 | 4 | -10.2105 | 4 | 3 | 8 | 1.0909 |

## Fold Summary

| fold | method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | anchor_top1 | 28 | 11310.4282 | 0.0000 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_09 | oracle_top10 | 28 | 11887.8514 | 127.2545 | 0 | 577.4232 | 23 | 0 | 23 | 5.7143 |
| valid_2025_09 | weather_bidspace_topk_aux | 28 | 11305.8834 | 0.0000 | 0 | -4.5448 | 1 | 1 | 2 | 1.1786 |
| valid_2025_10 | anchor_top1 | 31 | 7899.1033 | -15925.8974 | 3 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_10 | oracle_top10 | 31 | 9346.9273 | -9224.9472 | 1 | 1447.8240 | 27 | 0 | 27 | 6.4516 |
| valid_2025_10 | weather_bidspace_topk_aux | 31 | 7897.0508 | -15989.5246 | 3 | -2.0525 | 0 | 1 | 1 | 1.0645 |
| valid_2025_11 | anchor_top1 | 30 | 7120.7667 | 613.3252 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_11 | oracle_top10 | 30 | 7353.7912 | 676.9524 | 0 | 233.0245 | 22 | 0 | 22 | 5.0000 |
| valid_2025_11 | weather_bidspace_topk_aux | 30 | 7120.7667 | 613.3252 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_12 | anchor_top1 | 31 | 5928.2233 | -12550.1225 | 1 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_12 | oracle_top10 | 31 | 6513.6048 | -9394.2116 | 1 | 585.3815 | 21 | 0 | 21 | 4.0645 |
| valid_2025_12 | weather_bidspace_topk_aux | 31 | 5875.3853 | -12550.1225 | 1 | -52.8380 | 2 | 1 | 3 | 1.1935 |
| valid_2025_jan_feb | anchor_top1 | 56 | 14153.4998 | 8.7837 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_jan_feb | oracle_top10 | 56 | 14932.5473 | 8.7837 | 0 | 779.0475 | 38 | 0 | 38 | 4.0357 |
| valid_2025_jan_feb | weather_bidspace_topk_aux | 56 | 14154.0679 | 8.7837 | 0 | 0.5681 | 1 | 0 | 2 | 1.0536 |

## Selected Rules By Held-Out Fold

| heldout_fold | rule_id | selected_from_train | score_name | gate_feature | gate_quantile | gate_threshold | max_rank | score_margin | train_mean_lift_vs_anchor | train_changed_days | train_negative_lift_days | train_loss_days | train_anchor_loss_days | valid_rows | train_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank5__margin0.10 | True | weather_wind_score | top1_top2_gap | 0.1000 | -0.0753 | 5 | 0.1000 | 57.5009 | 7.0000 | 0.0000 | 4.0000 | 4.0000 | 280 | 1480 |
| valid_2025_10 | aux_netload_score__gate_top5_spread_std_q0.20__rank5__margin0.20 | True | netload_score | top5_spread_std | 0.2000 | 0.0371 | 5 | 0.2000 | 32.4401 | 11.0000 | 1.0000 | 1.0000 | 1.0000 | 310 | 1450 |
| valid_2025_11 | aux_netload_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 | True | netload_score | top1_top2_gap | 0.2000 | -0.0472 | 3 | 0.1000 | 50.9273 | 14.0000 | 3.0000 | 4.0000 | 4.0000 | 300 | 1460 |
| valid_2025_12 | aux_weather_wind_score__gate_top1_top2_gap_q0.20__rank5__margin0.20 | True | weather_wind_score | top1_top2_gap | 0.2000 | -0.0436 | 5 | 0.2000 | 53.0328 | 12.0000 | 2.0000 | 3.0000 | 3.0000 | 310 | 1450 |
| valid_2025_jan_feb | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank3__margin0.20 | True | weather_wind_score | top1_top2_gap | 0.1000 | -0.0696 | 3 | 0.2000 | 60.3931 | 3.0000 | 0.0000 | 4.0000 | 4.0000 | 560 | 1200 |

## Changed-Day Audit

| date | fold | candidate_rank | true_profit | top1_true_profit | lift_vs_anchor | charge_start | discharge_start | top1_charge_start | top1_discharge_start | rule_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-12-01 | valid_2025_12 | 4 | 1335.9556 | 3043.0490 | -1707.0933 | 52 | 71 | 53 | 71 | aux_weather_wind_score__gate_top1_top2_gap_q0.20__rank5__margin0.20 |
| 2025-09-27 | valid_2025_09 | 4 | 1768.8372 | 1959.7189 | -190.8817 | 54 | 73 | 55 | 73 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank5__margin0.10 |
| 2025-10-08 | valid_2025_10 | 3 | -15989.5246 | -15925.8974 | -63.6272 | 49 | 88 | 46 | 88 | aux_netload_score__gate_top5_spread_std_q0.20__rank5__margin0.20 |
| 2025-01-26 | valid_2025_jan_feb | 3 | 6493.5031 | 6493.5031 | 0.0000 | 58 | 88 | 55 | 88 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank3__margin0.20 |
| 2025-12-22 | valid_2025_12 | 3 | 3262.5311 | 3240.2616 | 22.2695 | 48 | 73 | 49 | 73 | aux_weather_wind_score__gate_top1_top2_gap_q0.20__rank5__margin0.20 |
| 2025-02-03 | valid_2025_jan_feb | 2 | 8971.4404 | 8939.6268 | 31.8136 | 55 | 75 | 55 | 73 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank3__margin0.20 |
| 2025-12-29 | valid_2025_12 | 2 | 11102.9528 | 11056.1073 | 46.8456 | 52 | 88 | 49 | 88 | aux_weather_wind_score__gate_top1_top2_gap_q0.20__rank5__margin0.20 |
| 2025-09-21 | valid_2025_09 | 3 | 522.8855 | 459.2582 | 63.6272 | 54 | 69 | 55 | 69 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank5__margin0.10 |

## Selected Rank Counts

| method | candidate_rank | days |
| --- | --- | --- |
| anchor_top1 | 1 | 176 |
| oracle_top10 | 1 | 45 |
| oracle_top10 | 2 | 12 |
| oracle_top10 | 3 | 10 |
| oracle_top10 | 4 | 16 |
| oracle_top10 | 5 | 16 |
| oracle_top10 | 6 | 16 |
| oracle_top10 | 7 | 11 |
| oracle_top10 | 8 | 24 |
| oracle_top10 | 9 | 6 |
| oracle_top10 | 10 | 20 |
| weather_bidspace_topk_aux | 1 | 168 |
| weather_bidspace_topk_aux | 2 | 2 |
| weather_bidspace_topk_aux | 3 | 4 |
| weather_bidspace_topk_aux | 4 | 2 |

## Top Passing Train Rules

| days | mean_profit | anchor_mean_profit | mean_lift_vs_anchor | total_lift_vs_anchor | worst_profit | p10_profit | loss_days | anchor_loss_days | changed_days | positive_lift_days | negative_lift_days | changed_rate | rule_id | score_name | gate_feature | gate_quantile | gate_threshold | max_rank | score_margin | passes_constraints | heldout_fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 120.0000 | 8051.7440 | 7991.3509 | 60.3931 | 7247.1709 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 3.0000 | 3.0000 | 0.0000 | 0.0250 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank3__margin0.00 | weather_wind_score | top1_top2_gap | 0.1000 | -0.0696 | 3 | 0.0000 | True | valid_2025_jan_feb |
| 120.0000 | 8051.7440 | 7991.3509 | 60.3931 | 7247.1709 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 3.0000 | 3.0000 | 0.0000 | 0.0250 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank3__margin0.05 | weather_wind_score | top1_top2_gap | 0.1000 | -0.0696 | 3 | 0.0500 | True | valid_2025_jan_feb |
| 120.0000 | 8051.7440 | 7991.3509 | 60.3931 | 7247.1709 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 3.0000 | 3.0000 | 0.0000 | 0.0250 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank3__margin0.10 | weather_wind_score | top1_top2_gap | 0.1000 | -0.0696 | 3 | 0.1000 | True | valid_2025_jan_feb |
| 120.0000 | 8051.7440 | 7991.3509 | 60.3931 | 7247.1709 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 3.0000 | 3.0000 | 0.0000 | 0.0250 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank3__margin0.20 | weather_wind_score | top1_top2_gap | 0.1000 | -0.0696 | 3 | 0.2000 | True | valid_2025_jan_feb |
| 120.0000 | 8051.7025 | 7991.3509 | 60.3515 | 7242.1857 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 8.0000 | 5.0000 | 2.0000 | 0.0667 | aux_weather_wind_score__gate_top1_top2_gap_q0.20__rank3__margin0.20 | weather_wind_score | top1_top2_gap | 0.2000 | -0.0436 | 3 | 0.2000 | True | valid_2025_jan_feb |
| 120.0000 | 8051.2138 | 7991.3509 | 59.8629 | 7183.5437 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 2.0000 | 2.0000 | 0.0000 | 0.0167 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank2__margin0.00 | weather_wind_score | top1_top2_gap | 0.1000 | -0.0696 | 2 | 0.0000 | True | valid_2025_jan_feb |
| 120.0000 | 8051.2138 | 7991.3509 | 59.8629 | 7183.5437 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 2.0000 | 2.0000 | 0.0000 | 0.0167 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank2__margin0.05 | weather_wind_score | top1_top2_gap | 0.1000 | -0.0696 | 2 | 0.0500 | True | valid_2025_jan_feb |
| 120.0000 | 8051.2138 | 7991.3509 | 59.8629 | 7183.5437 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 2.0000 | 2.0000 | 0.0000 | 0.0167 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank2__margin0.10 | weather_wind_score | top1_top2_gap | 0.1000 | -0.0696 | 2 | 0.1000 | True | valid_2025_jan_feb |
| 120.0000 | 8051.2138 | 7991.3509 | 59.8629 | 7183.5437 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 2.0000 | 2.0000 | 0.0000 | 0.0167 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank2__margin0.20 | weather_wind_score | top1_top2_gap | 0.1000 | -0.0696 | 2 | 0.2000 | True | valid_2025_jan_feb |
| 120.0000 | 8051.0740 | 7991.3509 | 59.7230 | 7166.7620 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 5.0000 | 3.0000 | 1.0000 | 0.0417 | aux_weather_wind_score__gate_top1_top2_gap_q0.20__rank2__margin0.20 | weather_wind_score | top1_top2_gap | 0.2000 | -0.0436 | 2 | 0.2000 | True | valid_2025_jan_feb |
| 120.0000 | 8051.0740 | 7991.3509 | 59.7230 | 7166.7620 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 6.0000 | 3.0000 | 1.0000 | 0.0500 | aux_weather_wind_score__gate_top1_top2_gap_q0.20__rank2__margin0.00 | weather_wind_score | top1_top2_gap | 0.2000 | -0.0436 | 2 | 0.0000 | True | valid_2025_jan_feb |
| 120.0000 | 8051.0740 | 7991.3509 | 59.7230 | 7166.7620 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 6.0000 | 3.0000 | 1.0000 | 0.0500 | aux_weather_wind_score__gate_top1_top2_gap_q0.20__rank2__margin0.05 | weather_wind_score | top1_top2_gap | 0.2000 | -0.0436 | 2 | 0.0500 | True | valid_2025_jan_feb |
| 120.0000 | 8051.0740 | 7991.3509 | 59.7230 | 7166.7620 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 6.0000 | 3.0000 | 1.0000 | 0.0500 | aux_weather_wind_score__gate_top1_top2_gap_q0.20__rank2__margin0.10 | weather_wind_score | top1_top2_gap | 0.2000 | -0.0436 | 2 | 0.1000 | True | valid_2025_jan_feb |
| 148.0000 | 9752.5422 | 9695.0413 | 57.5009 | 8510.1366 | -15925.8974 | 1312.3296 | 4.0000 | 4.0000 | 7.0000 | 5.0000 | 0.0000 | 0.0473 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank5__margin0.00 | weather_wind_score | top1_top2_gap | 0.1000 | -0.0753 | 5 | 0.0000 | True | valid_2025_09 |
| 148.0000 | 9752.5422 | 9695.0413 | 57.5009 | 8510.1366 | -15925.8974 | 1312.3296 | 4.0000 | 4.0000 | 7.0000 | 5.0000 | 0.0000 | 0.0473 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank5__margin0.05 | weather_wind_score | top1_top2_gap | 0.1000 | -0.0753 | 5 | 0.0500 | True | valid_2025_09 |
| 148.0000 | 9752.5422 | 9695.0413 | 57.5009 | 8510.1366 | -15925.8974 | 1312.3296 | 4.0000 | 4.0000 | 7.0000 | 5.0000 | 0.0000 | 0.0473 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank5__margin0.10 | weather_wind_score | top1_top2_gap | 0.1000 | -0.0753 | 5 | 0.1000 | True | valid_2025_09 |
| 145.0000 | 10865.3306 | 10812.2978 | 53.0328 | 7689.7588 | -15925.8974 | 1495.0969 | 3.0000 | 3.0000 | 12.0000 | 5.0000 | 2.0000 | 0.0828 | aux_weather_wind_score__gate_top1_top2_gap_q0.20__rank5__margin0.20 | weather_wind_score | top1_top2_gap | 0.2000 | -0.0436 | 5 | 0.2000 | True | valid_2025_12 |
| 148.0000 | 9747.0699 | 9695.0413 | 52.0286 | 7700.2318 | -15925.8974 | 1312.3296 | 4.0000 | 4.0000 | 6.0000 | 4.0000 | 0.0000 | 0.0405 | aux_weather_wind_score__gate_top1_top2_gap_q0.10__rank5__margin0.20 | weather_wind_score | top1_top2_gap | 0.1000 | -0.0753 | 5 | 0.2000 | True | valid_2025_09 |
| 146.0000 | 10584.7294 | 10533.8021 | 50.9273 | 7435.3866 | -15925.8974 | 1164.6823 | 4.0000 | 4.0000 | 14.0000 | 9.0000 | 3.0000 | 0.0959 | aux_netload_score__gate_top1_top2_gap_q0.20__rank3__margin0.00 | netload_score | top1_top2_gap | 0.2000 | -0.0472 | 3 | 0.0000 | True | valid_2025_11 |
| 146.0000 | 10584.7294 | 10533.8021 | 50.9273 | 7435.3866 | -15925.8974 | 1164.6823 | 4.0000 | 4.0000 | 14.0000 | 9.0000 | 3.0000 | 0.0959 | aux_netload_score__gate_top1_top2_gap_q0.20__rank3__margin0.05 | netload_score | top1_top2_gap | 0.2000 | -0.0472 | 3 | 0.0500 | True | valid_2025_11 |

## Decision

- The auxiliary switch does not beat anchor top1.
- Do not promote this rule family as currently defined.

## Artifacts

- `reports/weather_bidspace_topk_aux_selected_daily.csv` rows: `528`
- `reports/weather_bidspace_topk_aux_rules.csv`
- `reports/weather_bidspace_topk_aux_rule_search.csv`
- `reports/weather_bidspace_topk_aux_by_fold.csv`
- `reports/weather_bidspace_topk_aux_by_regime.csv`
- `reports/weather_bidspace_topk_aux_overall.csv`
