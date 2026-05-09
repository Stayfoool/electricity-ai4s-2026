# Top-K Reranker Diagnostics

## Scope

Model: `ens_champion_segmented6`. Candidate set: predicted top `10` legal pairs. The reranker is trained only on held-in months and evaluated on held-out months.

## Overall Held-Out Result

| method       |   mean_profit |   worst_fold_profit |   loss_days |   mean_selected_rank |   rank1_days |   lift_vs_top1 |
|:-------------|--------------:|--------------------:|------------:|---------------------:|-------------:|---------------:|
| top1         |     7850.8142 |           5867.3983 |           5 |               1.0000 |          120 |         0.0000 |
| reranker     |     7858.4950 |           5867.3983 |           5 |               1.0667 |          118 |         7.6808 |
| oracle_top10 |     8578.3433 |           6364.4851 |           3 |               5.7667 |           19 |       727.5291 |

## Fold Detail

| heldout_fold   | method       |   valid_days |   mean_profit |   worst_profit |   loss_days |   mean_rank |   rank1_days |
|:---------------|:-------------|-------------:|--------------:|---------------:|------------:|------------:|-------------:|
| valid_2025_09  | top1         |           28 |    10898.1313 |        95.4409 |           0 |      1.0000 |           28 |
| valid_2025_09  | reranker     |           28 |    10909.4933 |        95.4409 |           0 |      1.0357 |           27 |
| valid_2025_09  | oracle_top10 |           28 |    11782.5392 |       318.1362 |           0 |      7.7143 |            0 |
| valid_2025_10  | top1         |           31 |     7961.4092 |    -15992.7060 |           4 |      1.0000 |           31 |
| valid_2025_10  | reranker     |           31 |     7961.4092 |    -15992.7060 |           4 |      1.0000 |           31 |
| valid_2025_10  | oracle_top10 |           31 |     9081.2656 |     -9224.9472 |           2 |      6.0000 |            4 |
| valid_2025_11  | top1         |           30 |     6941.8998 |       890.7813 |           0 |      1.0000 |           30 |
| valid_2025_11  | reranker     |           30 |     6962.0186 |       890.7813 |           0 |      1.2333 |           29 |
| valid_2025_11  | oracle_top10 |           30 |     7355.7273 |       922.5949 |           0 |      4.9000 |            6 |
| valid_2025_12  | top1         |           31 |     5867.3983 |    -11244.2975 |           1 |      1.0000 |           31 |
| valid_2025_12  | reranker     |           31 |     5867.3983 |    -11244.2975 |           1 |      1.0000 |           31 |
| valid_2025_12  | oracle_top10 |           31 |     6364.4851 |     -7500.6109 |           1 |      4.6129 |            9 |

## Reranker Selected Rank Counts

|   rank |   days |
|-------:|-------:|
|      1 |    118 |
|      2 |      1 |
|      8 |      1 |

## Decision

- The held-out reranker improves over predicted top1.
- Next step is to harden it into a submit-time strategy only if fold-level results are stable.
