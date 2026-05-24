# Current Prior Top-K Conservative Switch

## Scope

- Anchor: current online champion-prior top1 pair.
- Candidate set: champion-prior top10 legal charge/discharge pairs.
- Goal: default to top1 and switch only on a small number of high-risk days.
- Validation: for each held-out fold, choose one simple rule using only other folds.
- No price model retraining and no submit file generation in this experiment.

## Rule Family

- Gate: switch is allowed only when a top1 confidence feature is below a training quantile threshold.
- Candidate: among top-k alternatives, choose the highest auxiliary score.
- Auxiliary scores use only prediction-time features: expert support/rank, bid_space, net_load, renewable ratio, and timing features.
- Constraints: training changed rate <= 10%, training lift >= 5/day, loss days not above anchor, and negative changed days not above positive changed days.

## Overall

| regime | method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank | profit_delta_vs_anchor | loss_day_delta_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_5fold | oracle_top10 | 176 | 10689.6177 | -9394.2116 | 2 | 737.5830 | 131 | 0 | 131 | 4.8977 | 737.5830 | -2 |
| all_5fold | conservative_switch | 176 | 9977.6876 | -15925.8974 | 4 | 25.6529 | 9 | 4 | 16 | 1.1477 | 25.6529 | 0 |
| all_5fold | anchor_top1 | 176 | 9952.0347 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 | 0.0000 | 0 |

## Regime Summary

| regime | method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard_09_12 | anchor_top1 | 120 | 7991.3509 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| standard_09_12 | conservative_switch | 120 | 8021.7786 | -15925.8974 | 4 | 30.4277 | 6 | 3 | 11 | 1.1500 |
| standard_09_12 | oracle_top10 | 120 | 8709.5839 | -9394.2116 | 2 | 718.2329 | 93 | 0 | 93 | 5.3000 |
| jan_feb_like | anchor_top1 | 56 | 14153.4998 | 8.7837 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| jan_feb_like | conservative_switch | 56 | 14168.9213 | 8.7837 | 0 | 15.4214 | 3 | 1 | 5 | 1.1429 |
| jan_feb_like | oracle_top10 | 56 | 14932.5473 | 8.7837 | 0 | 779.0475 | 38 | 0 | 38 | 4.0357 |
| winter_11_12_jan_feb | anchor_top1 | 117 | 10170.8882 | -12550.1225 | 1 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| winter_11_12_jan_feb | conservative_switch | 117 | 10182.6397 | -12550.1225 | 1 | 11.7515 | 4 | 2 | 7 | 1.0855 |
| winter_11_12_jan_feb | oracle_top10 | 117 | 10758.6165 | -9394.2116 | 1 | 587.7283 | 81 | 0 | 81 | 4.2906 |
| all_5fold | anchor_top1 | 176 | 9952.0347 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| all_5fold | conservative_switch | 176 | 9977.6876 | -15925.8974 | 4 | 25.6529 | 9 | 4 | 16 | 1.1477 |
| all_5fold | oracle_top10 | 176 | 10689.6177 | -9394.2116 | 2 | 737.5830 | 131 | 0 | 131 | 4.8977 |

## Fold Summary

| fold | method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | anchor_top1 | 28 | 11310.4282 | 0.0000 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_09 | conservative_switch | 28 | 11329.5463 | 0.0000 | 0 | 19.1182 | 3 | 1 | 5 | 1.3214 |
| valid_2025_09 | oracle_top10 | 28 | 11887.8514 | 127.2545 | 0 | 577.4232 | 23 | 0 | 23 | 5.7143 |
| valid_2025_10 | anchor_top1 | 31 | 7899.1033 | -15925.8974 | 3 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_10 | conservative_switch | 31 | 7983.1253 | -15925.8974 | 3 | 84.0220 | 2 | 1 | 4 | 1.2258 |
| valid_2025_10 | oracle_top10 | 31 | 9346.9273 | -9224.9472 | 1 | 1447.8240 | 27 | 0 | 27 | 6.4516 |
| valid_2025_11 | anchor_top1 | 30 | 7120.7667 | 613.3252 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_11 | conservative_switch | 30 | 7120.7667 | 613.3252 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_11 | oracle_top10 | 30 | 7353.7912 | 676.9524 | 0 | 233.0245 | 22 | 0 | 22 | 5.0000 |
| valid_2025_12 | anchor_top1 | 31 | 5928.2233 | -12550.1225 | 1 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_12 | conservative_switch | 31 | 5944.7178 | -12550.1225 | 1 | 16.4944 | 1 | 1 | 2 | 1.0645 |
| valid_2025_12 | oracle_top10 | 31 | 6513.6048 | -9394.2116 | 1 | 585.3815 | 21 | 0 | 21 | 4.0645 |
| valid_2025_jan_feb | anchor_top1 | 56 | 14153.4998 | 8.7837 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_jan_feb | conservative_switch | 56 | 14168.9213 | 8.7837 | 0 | 15.4214 | 3 | 1 | 5 | 1.1429 |
| valid_2025_jan_feb | oracle_top10 | 56 | 14932.5473 | 8.7837 | 0 | 779.0475 | 38 | 0 | 38 | 4.0357 |

## Selected Rules By Held-Out Fold

| heldout_fold | rule_id | selected_from_train | score_name | gate_feature | gate_quantile | gate_threshold | max_rank | score_margin | train_mean_lift_vs_anchor | train_changed_days | train_negative_lift_days | train_loss_days | train_anchor_loss_days | valid_rows | train_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 | True | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0368 | 3 | 0.1000 | 26.8893 | 11.0000 | 3.0000 | 4.0000 | 4.0000 | 280 | 1480 |
| valid_2025_10 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 | True | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0270 | 3 | 0.1000 | 17.5621 | 13.0000 | 3.0000 | 1.0000 | 1.0000 | 310 | 1450 |
| valid_2025_11 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 | True | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0472 | 3 | 0.1000 | 41.6675 | 13.0000 | 2.0000 | 4.0000 | 4.0000 | 300 | 1460 |
| valid_2025_12 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 | True | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0436 | 3 | 0.1000 | 24.1554 | 13.0000 | 3.0000 | 3.0000 | 3.0000 | 310 | 1450 |
| valid_2025_jan_feb | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 | True | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0436 | 3 | 0.1000 | 26.2521 | 10.0000 | 3.0000 | 4.0000 | 4.0000 | 560 | 1200 |

## Changed-Day Audit

| date | fold | candidate_rank | true_profit | top1_true_profit | lift_vs_anchor | charge_start | discharge_start | top1_charge_start | top1_discharge_start | rule_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-10-06 | valid_2025_10 | 3 | 12110.3374 | 13839.5253 | -1729.1879 | 52 | 69 | 52 | 67 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-09-14 | valid_2025_09 | 3 | 35931.8149 | 36565.6567 | -633.8418 | 51 | 69 | 52 | 69 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-12-22 | valid_2025_12 | 2 | 2899.8559 | 3240.2616 | -340.4057 | 49 | 71 | 49 | 73 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-02-09 | valid_2025_jan_feb | 3 | 8078.0408 | 8176.0999 | -98.0591 | 55 | 73 | 55 | 71 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-09-20 | valid_2025_09 | 2 | 14214.3247 | 14214.3247 | 0.0000 | 47 | 66 | 46 | 66 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-10-28 | valid_2025_10 | 3 | 10234.5937 | 10234.5937 | 0.0000 | 53 | 71 | 55 | 71 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-02-26 | valid_2025_jan_feb | 3 | 38176.3420 | 38176.3420 | 0.0000 | 50 | 71 | 52 | 71 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-02-03 | valid_2025_jan_feb | 2 | 8971.4404 | 8939.6268 | 31.8136 | 55 | 75 | 55 | 73 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-09-21 | valid_2025_09 | 3 | 522.8855 | 459.2582 | 63.6272 | 54 | 69 | 55 | 69 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-02-05 | valid_2025_jan_feb | 2 | 279.1041 | 199.4491 | 79.6549 | 57 | 75 | 57 | 73 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-09-01 | valid_2025_09 | 3 | 7460.2935 | 6959.2290 | 501.0645 | 54 | 71 | 55 | 71 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-09-03 | valid_2025_09 | 3 | 7811.2518 | 7206.7931 | 604.4587 | 51 | 69 | 52 | 69 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-02-11 | valid_2025_jan_feb | 3 | 18721.4554 | 17871.2651 | 850.1903 | 16 | 73 | 17 | 73 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-12-10 | valid_2025_12 | 2 | 4094.0277 | 3242.2945 | 851.7333 | 50 | 88 | 49 | 88 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-10-14 | valid_2025_10 | 3 | 9869.1921 | 8996.6718 | 872.5203 | 54 | 73 | 55 | 73 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |
| 2025-10-05 | valid_2025_10 | 2 | 15833.8574 | 12372.5071 | 3461.3503 | 32 | 64 | 32 | 62 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 |

## Selected Rank Counts

| method | candidate_rank | days |
| --- | --- | --- |
| anchor_top1 | 1 | 176 |
| conservative_switch | 1 | 160 |
| conservative_switch | 2 | 6 |
| conservative_switch | 3 | 10 |
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

## Top Passing Train Rules

| days | mean_profit | anchor_mean_profit | mean_lift_vs_anchor | total_lift_vs_anchor | worst_profit | p10_profit | loss_days | anchor_loss_days | changed_days | positive_lift_days | negative_lift_days | changed_rate | rule_id | score_name | gate_feature | gate_quantile | gate_threshold | max_rank | score_margin | passes_constraints | heldout_fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 146.0000 | 10575.4695 | 10533.8021 | 41.6675 | 6083.4478 | -15925.8974 | 1164.6823 | 4.0000 | 4.0000 | 13.0000 | 8.0000 | 2.0000 | 0.0890 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.00 | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0472 | 3 | 0.0000 | True | valid_2025_11 |
| 146.0000 | 10575.4695 | 10533.8021 | 41.6675 | 6083.4478 | -15925.8974 | 1164.6823 | 4.0000 | 4.0000 | 13.0000 | 8.0000 | 2.0000 | 0.0890 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0472 | 3 | 0.1000 | True | valid_2025_11 |
| 148.0000 | 9721.9306 | 9695.0413 | 26.8893 | 3979.6101 | -15925.8974 | 1312.3296 | 4.0000 | 4.0000 | 11.0000 | 6.0000 | 3.0000 | 0.0743 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.00 | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0368 | 3 | 0.0000 | True | valid_2025_09 |
| 148.0000 | 9721.9306 | 9695.0413 | 26.8893 | 3979.6101 | -15925.8974 | 1312.3296 | 4.0000 | 4.0000 | 11.0000 | 6.0000 | 3.0000 | 0.0743 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0368 | 3 | 0.1000 | True | valid_2025_09 |
| 120.0000 | 8017.6031 | 7991.3509 | 26.2521 | 3150.2545 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 10.0000 | 5.0000 | 3.0000 | 0.0833 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.00 | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0436 | 3 | 0.0000 | True | valid_2025_jan_feb |
| 120.0000 | 8017.6031 | 7991.3509 | 26.2521 | 3150.2545 | -15925.8974 | 947.2304 | 4.0000 | 4.0000 | 10.0000 | 5.0000 | 3.0000 | 0.0833 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0436 | 3 | 0.1000 | True | valid_2025_jan_feb |
| 145.0000 | 10836.4532 | 10812.2978 | 24.1554 | 3502.5267 | -15925.8974 | 1495.0969 | 3.0000 | 3.0000 | 13.0000 | 7.0000 | 3.0000 | 0.0897 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.00 | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0436 | 3 | 0.0000 | True | valid_2025_12 |
| 145.0000 | 10836.4532 | 10812.2978 | 24.1554 | 3502.5267 | -15925.8974 | 1495.0969 | 3.0000 | 3.0000 | 13.0000 | 7.0000 | 3.0000 | 0.0897 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0436 | 3 | 0.1000 | True | valid_2025_12 |
| 145.0000 | 10408.4994 | 10390.9373 | 17.5621 | 2546.5084 | -12550.1225 | 2219.7990 | 1.0000 | 1.0000 | 13.0000 | 8.0000 | 3.0000 | 0.0897 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.00 | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0270 | 3 | 0.0000 | True | valid_2025_10 |
| 145.0000 | 10408.4994 | 10390.9373 | 17.5621 | 2546.5084 | -12550.1225 | 2219.7990 | 1.0000 | 1.0000 | 13.0000 | 8.0000 | 3.0000 | 0.0897 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0270 | 3 | 0.1000 | True | valid_2025_10 |
| 120.0000 | 8005.1510 | 7991.3509 | 13.8000 | 1656.0006 | -15925.8974 | 1414.0174 | 4.0000 | 4.0000 | 12.0000 | 6.0000 | 4.0000 | 0.1000 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank5__margin0.00 | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0436 | 5 | 0.0000 | True | valid_2025_jan_feb |
| 120.0000 | 8005.1510 | 7991.3509 | 13.8000 | 1656.0006 | -15925.8974 | 1414.0174 | 4.0000 | 4.0000 | 12.0000 | 6.0000 | 4.0000 | 0.1000 | switch_bidspace_rank_score__gate_top1_top2_gap_q0.20__rank5__margin0.10 | bidspace_rank_score | top1_top2_gap | 0.2000 | -0.0436 | 5 | 0.1000 | True | valid_2025_jan_feb |
| 148.0000 | 9706.2324 | 9695.0413 | 11.1910 | 1656.2742 | -15925.8974 | 1312.3296 | 4.0000 | 4.0000 | 14.0000 | 4.0000 | 4.0000 | 0.0946 | switch_support_score__gate_top5_spread_std_q0.33__rank3__margin0.10 | support_score | top5_spread_std | 0.3300 | 0.0438 | 3 | 0.1000 | True | valid_2025_09 |
| 148.0000 | 9701.1358 | 9695.0413 | 6.0945 | 901.9797 | -15925.8974 | 1542.7642 | 4.0000 | 4.0000 | 13.0000 | 5.0000 | 5.0000 | 0.0878 | switch_margin_combo_score__gate_top1_top2_gap_q0.20__rank3__margin0.00 | margin_combo_score | top1_top2_gap | 0.2000 | -0.0368 | 3 | 0.0000 | True | valid_2025_09 |
| 148.0000 | 9701.1358 | 9695.0413 | 6.0945 | 901.9797 | -15925.8974 | 1542.7642 | 4.0000 | 4.0000 | 13.0000 | 5.0000 | 5.0000 | 0.0878 | switch_margin_combo_score__gate_top1_top2_gap_q0.20__rank3__margin0.10 | margin_combo_score | top1_top2_gap | 0.2000 | -0.0368 | 3 | 0.1000 | True | valid_2025_09 |

## Decision

- Conservative switching is positive but below the promotion threshold.
- Keep it as a diagnostic; do not submit until the signal is stronger.

## Artifacts

- `reports/current_prior_topk_conservative_switch_selected_daily.csv` rows: `528`
- `reports/current_prior_topk_conservative_switch_rules.csv`
- `reports/current_prior_topk_conservative_switch_rule_search.csv`
- `reports/current_prior_topk_conservative_switch_by_fold.csv`
- `reports/current_prior_topk_conservative_switch_by_regime.csv`
- `reports/current_prior_topk_conservative_switch_overall.csv`
