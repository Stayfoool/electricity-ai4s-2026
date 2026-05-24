# Current Prior Top-K Pair Reranker Fold-Safe NWP Residual

## Scope

- Anchor and candidate set are unchanged from the current prior top-K reranker.
- Candidate set: champion-prior top10 legal charge/discharge pairs per day.
- Difference from the original reranker: NWP month-hour residual means are fitted outside each outer held-out fold.
- Fallback when held-out months are absent from training side: hour-of-day mean, then global training mean.
- This is an outer fold-safe leakage check, not a submit strategy.

## Overall

| regime | method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank | profit_delta_vs_anchor | loss_day_delta_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_5fold | oracle_top10 | 176 | 10689.6177 | -9394.2116 | 2 | 737.5830 | 131 | 0 | 155 | 5.6193 | 737.5830 | -2 |
| all_5fold | anchor_top1 | 176 | 9952.0347 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 | 0.0000 | 0 |
| all_5fold | gbr_lift | 176 | 9947.3503 | -15925.8974 | 4 | -4.6844 | 2 | 6 | 12 | 1.3636 | -4.6844 | 0 |
| all_5fold | gbr_true_profit | 176 | 9840.3288 | -15925.8974 | 4 | -111.7059 | 10 | 22 | 41 | 2.3295 | -111.7059 | 0 |

## Original vs Fold-Safe

| method | original_mean_profit | original_loss_days | original_changed_days | original_delta_vs_anchor | foldsafe_mean_profit | foldsafe_loss_days | foldsafe_changed_days | foldsafe_delta_vs_anchor | mean_profit_change | delta_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_top10 | 10689.6177 | 2 | 155 | 737.5830 | 10689.6177 | 2 | 155 | 737.5830 | -0.0000 | 0.0000 |
| anchor_top1 | 9952.0347 | 4 | 0 | 0.0000 | 9952.0347 | 4 | 0 | 0.0000 | 0.0000 | 0.0000 |
| gbr_lift | 9952.1298 | 4 | 7 | 0.0952 | 9947.3503 | 4 | 12 | -4.6844 | -4.7796 | -4.7796 |
| gbr_true_profit | 9875.5600 | 5 | 33 | -76.4747 | 9840.3288 | 4 | 41 | -111.7059 | -35.2312 | -35.2312 |

## Regime Summary

| regime | method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard_09_12 | anchor_top1 | 120 | 7991.3509 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| standard_09_12 | gbr_lift | 120 | 7991.3509 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| standard_09_12 | gbr_true_profit | 120 | 7882.0110 | -15925.8974 | 4 | -109.3400 | 6 | 14 | 24 | 2.0833 |
| standard_09_12 | oracle_top10 | 120 | 8709.5839 | -9394.2116 | 2 | 718.2329 | 93 | 0 | 102 | 5.6083 |
| jan_feb_like | anchor_top1 | 56 | 14153.4998 | 8.7837 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| jan_feb_like | gbr_lift | 56 | 14138.7774 | 8.7837 | 0 | -14.7224 | 2 | 6 | 12 | 2.1429 |
| jan_feb_like | gbr_true_profit | 56 | 14036.7241 | 8.7837 | 0 | -116.7757 | 4 | 8 | 17 | 2.8571 |
| jan_feb_like | oracle_top10 | 56 | 14932.5473 | 8.7837 | 0 | 779.0475 | 38 | 0 | 53 | 5.6429 |
| winter_11_12_jan_feb | anchor_top1 | 117 | 10170.8882 | -12550.1225 | 1 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| winter_11_12_jan_feb | gbr_lift | 117 | 10163.8415 | -12550.1225 | 1 | -7.0466 | 2 | 6 | 12 | 1.5470 |
| winter_11_12_jan_feb | gbr_true_profit | 117 | 10052.4464 | -12550.1225 | 1 | -118.4417 | 9 | 16 | 33 | 2.5812 |
| winter_11_12_jan_feb | oracle_top10 | 117 | 10758.6165 | -9394.2116 | 1 | 587.7283 | 81 | 0 | 101 | 5.2564 |
| all_5fold | anchor_top1 | 176 | 9952.0347 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| all_5fold | gbr_lift | 176 | 9947.3503 | -15925.8974 | 4 | -4.6844 | 2 | 6 | 12 | 1.3636 |
| all_5fold | gbr_true_profit | 176 | 9840.3288 | -15925.8974 | 4 | -111.7059 | 10 | 22 | 41 | 2.3295 |
| all_5fold | oracle_top10 | 176 | 10689.6177 | -9394.2116 | 2 | 737.5830 | 131 | 0 | 155 | 5.6193 |

## Fold Summary

| fold | method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_anchor | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | anchor_top1 | 28 | 11310.4282 | 0.0000 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_09 | gbr_lift | 28 | 11310.4282 | 0.0000 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_09 | gbr_true_profit | 28 | 11231.9167 | 0.0000 | 0 | -78.5115 | 0 | 1 | 2 | 1.5714 |
| valid_2025_09 | oracle_top10 | 28 | 11887.8514 | 127.2545 | 0 | 577.4232 | 23 | 0 | 26 | 6.0000 |
| valid_2025_10 | anchor_top1 | 31 | 7899.1033 | -15925.8974 | 3 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_10 | gbr_lift | 31 | 7899.1033 | -15925.8974 | 3 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_10 | gbr_true_profit | 31 | 7782.8376 | -15925.8974 | 3 | -116.2656 | 1 | 5 | 6 | 2.0645 |
| valid_2025_10 | oracle_top10 | 31 | 9346.9273 | -9224.9472 | 1 | 1447.8240 | 27 | 0 | 28 | 6.6452 |
| valid_2025_11 | anchor_top1 | 30 | 7120.7667 | 613.3252 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_11 | gbr_lift | 30 | 7120.7667 | 613.3252 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_11 | gbr_true_profit | 30 | 7001.9588 | 453.6717 | 0 | -118.8079 | 4 | 6 | 12 | 2.9667 |
| valid_2025_11 | oracle_top10 | 30 | 7353.7912 | 676.9524 | 0 | 233.0245 | 22 | 0 | 24 | 5.4000 |
| valid_2025_12 | anchor_top1 | 31 | 5928.2233 | -12550.1225 | 1 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_12 | gbr_lift | 31 | 5928.2233 | -12550.1225 | 1 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_12 | gbr_true_profit | 31 | 5807.1263 | -12550.1225 | 1 | -121.0970 | 1 | 2 | 4 | 1.7097 |
| valid_2025_12 | oracle_top10 | 31 | 6513.6048 | -9394.2116 | 1 | 585.3815 | 21 | 0 | 24 | 4.4194 |
| valid_2025_jan_feb | anchor_top1 | 56 | 14153.4998 | 8.7837 | 0 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| valid_2025_jan_feb | gbr_lift | 56 | 14138.7774 | 8.7837 | 0 | -14.7224 | 2 | 6 | 12 | 2.1429 |
| valid_2025_jan_feb | gbr_true_profit | 56 | 14036.7241 | 8.7837 | 0 | -116.7757 | 4 | 8 | 17 | 2.8571 |
| valid_2025_jan_feb | oracle_top10 | 56 | 14932.5473 | 8.7837 | 0 | 779.0475 | 38 | 0 | 53 | 5.6429 |

## Selected Rank Counts

| method | candidate_rank | days |
| --- | --- | --- |
| anchor_top1 | 1 | 176 |
| gbr_lift | 1 | 164 |
| gbr_lift | 2 | 1 |
| gbr_lift | 3 | 2 |
| gbr_lift | 4 | 1 |
| gbr_lift | 6 | 2 |
| gbr_lift | 7 | 1 |
| gbr_lift | 8 | 1 |
| gbr_lift | 9 | 3 |
| gbr_lift | 10 | 1 |
| gbr_true_profit | 1 | 135 |
| gbr_true_profit | 2 | 1 |
| gbr_true_profit | 3 | 5 |
| gbr_true_profit | 4 | 5 |
| gbr_true_profit | 5 | 3 |
| gbr_true_profit | 6 | 7 |
| gbr_true_profit | 8 | 7 |
| gbr_true_profit | 9 | 5 |
| gbr_true_profit | 10 | 8 |
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
| valid_2025_09 | gbr_true_profit | 500.0000 | 179 | 1480 | 280 |
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

- Fold-safe learned rerankers do not beat anchor top1.
- Do not submit a learned reranker from this experiment.

## Artifacts

- `reports/current_prior_topk_pair_reranker_foldsafe_dataset.csv` rows: `1760`
- `reports/current_prior_topk_pair_reranker_foldsafe_selected_daily.csv` rows: `704`
- `reports/current_prior_topk_pair_reranker_foldsafe_by_fold.csv`
- `reports/current_prior_topk_pair_reranker_foldsafe_by_regime.csv`
- `reports/current_prior_topk_pair_reranker_foldsafe_overall.csv`
- `reports/current_prior_topk_pair_reranker_foldsafe_params.csv`
