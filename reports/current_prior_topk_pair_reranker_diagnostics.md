# Current Prior Top-K Pair Reranker

## Scope

- Anchor: `ens_champion_segmented6_prior` current champion with dispatch prior.
- Candidate set: anchor top10 legal charge/discharge pairs.
- Added features: bid_space window means/ranks, selected weather month-hour residuals, candidate timing, top1 gap/confidence, and support from two existing experts.
- Validation: outer held-out fold; threshold selected by inner out-of-fold predictions.

## Overall

| regime | method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank | profit_delta_vs_anchor | loss_day_delta_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_5fold | oracle_top10 | 176 | 10689.6177 | -9394.2116 | 2 | 737.5830 | 131 | 0 | 155 | 5.6193 | 737.5830 | -2 |
| all_5fold | gbr_lift | 176 | 9952.1298 | -15925.8974 | 4 | 0.0952 | 2 | 3 | 7 | 1.1648 | 0.0952 | 0 |
| all_5fold | anchor_top1 | 176 | 9952.0347 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 | 0.0000 | 0 |
| all_5fold | gbr_true_profit | 176 | 9875.5600 | -15925.8974 | 5 | -76.4747 | 8 | 19 | 33 | 2.0852 | -76.4747 | 1 |

## Regime Summary

| regime | method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard_09_12 | anchor_top1 | 120 | 7991.3509 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| standard_09_12 | gbr_lift | 120 | 7991.3509 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| standard_09_12 | gbr_true_profit | 120 | 7910.8600 | -15925.8974 | 5 | -80.4910 | 4 | 9 | 16 | 1.8250 |
| standard_09_12 | oracle_top10 | 120 | 8709.5839 | -9394.2116 | 2 | 718.2329 | 93 | 0 | 102 | 5.6083 |
| jan_feb_like | anchor_top1 | 56 | 14153.4998 | 8.7837 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| jan_feb_like | gbr_lift | 56 | 14153.7989 | 8.7837 | 0 | 0.2990 | 2 | 3 | 7 | 1.5179 |
| jan_feb_like | gbr_true_profit | 56 | 14085.6315 | 7.6575 | 0 | -67.8683 | 4 | 10 | 17 | 2.6429 |
| jan_feb_like | oracle_top10 | 56 | 14932.5473 | 8.7837 | 0 | 779.0475 | 38 | 0 | 53 | 5.6429 |
| winter_11_12_jan_feb | anchor_top1 | 117 | 10170.8882 | -12550.1225 | 1 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| winter_11_12_jan_feb | gbr_lift | 117 | 10171.0313 | -12550.1225 | 1 | 0.1431 | 2 | 3 | 7 | 1.2479 |
| winter_11_12_jan_feb | gbr_true_profit | 117 | 10130.4816 | -12550.1225 | 1 | -40.4065 | 8 | 13 | 26 | 2.2564 |
| winter_11_12_jan_feb | oracle_top10 | 117 | 10758.6165 | -9394.2116 | 1 | 587.7283 | 81 | 0 | 101 | 5.2564 |
| all_5fold | anchor_top1 | 176 | 9952.0347 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| all_5fold | gbr_lift | 176 | 9952.1298 | -15925.8974 | 4 | 0.0952 | 2 | 3 | 7 | 1.1648 |
| all_5fold | gbr_true_profit | 176 | 9875.5600 | -15925.8974 | 5 | -76.4747 | 8 | 19 | 33 | 2.0852 |
| all_5fold | oracle_top10 | 176 | 10689.6177 | -9394.2116 | 2 | 737.5830 | 131 | 0 | 155 | 5.6193 |

## Fold Summary

| fold | method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | anchor_top1 | 28 | 11310.4282 | 0.0000 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_09 | gbr_lift | 28 | 11310.4282 | 0.0000 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_09 | gbr_true_profit | 28 | 11231.9167 | 0.0000 | 0 | -78.5115 | 0 | 1 | 1 | 1.3214 |
| valid_2025_09 | oracle_top10 | 28 | 11887.8514 | 127.2545 | 0 | 577.4232 | 23 | 0 | 26 | 6.0000 |
| valid_2025_10 | anchor_top1 | 31 | 7899.1033 | -15925.8974 | 3 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_10 | gbr_lift | 31 | 7899.1033 | -15925.8974 | 3 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_10 | gbr_true_profit | 31 | 7688.3401 | -15925.8974 | 4 | -210.7632 | 0 | 5 | 6 | 2.1290 |
| valid_2025_10 | oracle_top10 | 31 | 9346.9273 | -9224.9472 | 1 | 1447.8240 | 27 | 0 | 28 | 6.6452 |
| valid_2025_11 | anchor_top1 | 30 | 7120.7667 | 613.3252 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_11 | gbr_lift | 30 | 7120.7667 | 613.3252 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_11 | gbr_true_profit | 30 | 7089.8688 | 453.6717 | 0 | -30.8979 | 4 | 3 | 9 | 2.8333 |
| valid_2025_11 | oracle_top10 | 30 | 7353.7912 | 676.9524 | 0 | 233.0245 | 22 | 0 | 24 | 5.4000 |
| valid_2025_12 | anchor_top1 | 31 | 5928.2233 | -12550.1225 | 1 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_12 | gbr_lift | 31 | 5928.2233 | -12550.1225 | 1 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_12 | gbr_true_profit | 31 | 5928.2233 | -12550.1225 | 1 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_12 | oracle_top10 | 31 | 6513.6048 | -9394.2116 | 1 | 585.3815 | 21 | 0 | 24 | 4.4194 |
| valid_2025_jan_feb | anchor_top1 | 56 | 14153.4998 | 8.7837 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_jan_feb | gbr_lift | 56 | 14153.7989 | 8.7837 | 0 | 0.2990 | 2 | 3 | 7 | 1.5179 |
| valid_2025_jan_feb | gbr_true_profit | 56 | 14085.6315 | 7.6575 | 0 | -67.8683 | 4 | 10 | 17 | 2.6429 |
| valid_2025_jan_feb | oracle_top10 | 56 | 14932.5473 | 8.7837 | 0 | 779.0475 | 38 | 0 | 53 | 5.6429 |

## Selected Rank Counts

| method | candidate_rank | days |
| --- | --- | --- |
| anchor_top1 | 1 | 176 |
| gbr_lift | 1 | 169 |
| gbr_lift | 2 | 1 |
| gbr_lift | 3 | 1 |
| gbr_lift | 4 | 1 |
| gbr_lift | 6 | 2 |
| gbr_lift | 7 | 1 |
| gbr_lift | 8 | 1 |
| gbr_true_profit | 1 | 143 |
| gbr_true_profit | 2 | 2 |
| gbr_true_profit | 3 | 1 |
| gbr_true_profit | 4 | 6 |
| gbr_true_profit | 5 | 1 |
| gbr_true_profit | 6 | 7 |
| gbr_true_profit | 7 | 1 |
| gbr_true_profit | 8 | 2 |
| gbr_true_profit | 9 | 7 |
| gbr_true_profit | 10 | 6 |
| oracle_top10 | 1 | 21 |
| oracle_top10 | 2 | 15 |
| oracle_top10 | 3 | 10 |
| oracle_top10 | 4 | 21 |
| oracle_top10 | 5 | 18 |
| oracle_top10 | 6 | 18 |
| oracle_top10 | 7 | 15 |
| oracle_top10 | 8 | 24 |
| oracle_top10 | 9 | 11 |
| oracle_top10 | 10 | 23 |

## Chosen Thresholds

| heldout_fold | method | threshold | feature_count | train_rows | valid_rows |
| --- | --- | --- | --- | --- | --- |
| valid_2025_09 | gbr_true_profit | 750.0000 | 179 | 1480 | 280 |
| valid_2025_09 | gbr_lift | 1000.0000 | 179 | 1480 | 280 |
| valid_2025_10 | gbr_true_profit | 500.0000 | 179 | 1450 | 310 |
| valid_2025_10 | gbr_lift | 1000.0000 | 179 | 1450 | 310 |
| valid_2025_11 | gbr_true_profit | 500.0000 | 179 | 1460 | 300 |
| valid_2025_11 | gbr_lift | 1000.0000 | 179 | 1460 | 300 |
| valid_2025_12 | gbr_true_profit | 1000.0000 | 179 | 1450 | 310 |
| valid_2025_12 | gbr_lift | 1000.0000 | 179 | 1450 | 310 |
| valid_2025_jan_feb | gbr_true_profit | 250.0000 | 179 | 1200 | 560 |
| valid_2025_jan_feb | gbr_lift | 0.0000 | 179 | 1200 | 560 |

## Decision

- The best learned top-K reranker is positive but too small.
- Do not submit yet unless Jan-Feb-like and loss-day behavior are also favorable.

## Artifacts

- `reports/current_prior_topk_pair_reranker_dataset.csv` rows: `1760`
- `reports/current_prior_topk_pair_reranker_selected_daily.csv` rows: `704`
- `reports/current_prior_topk_pair_reranker_by_fold.csv`
- `reports/current_prior_topk_pair_reranker_by_regime.csv`
- `reports/current_prior_topk_pair_reranker_overall.csv`
- `reports/current_prior_topk_pair_reranker_params.csv`
