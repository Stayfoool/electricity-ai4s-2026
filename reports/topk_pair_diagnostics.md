# Top-K Pair Diagnostics

## Scope

For each expert and validation day, rank legal charge/discharge pairs by predicted spread, then score the predicted top-K candidates using hidden validation prices.

## Summary

| model                                 |   top_k |   days |   top1_mean_profit |   topk_best_mean_profit |   topk_lift_mean |   topk_lift_positive_days |   topk_lift_positive_rate |   top1_oracle_hit_rate |   oracle_in_topk_rate |   top1_mean_regret |   topk_best_mean_regret |
|:--------------------------------------|--------:|-------:|-------------------:|------------------------:|-----------------:|--------------------------:|--------------------------:|-----------------------:|----------------------:|-------------------:|------------------------:|
| ens_champion_segmented6               |       1 |    120 |          7850.8142 |               7850.8142 |           0.0000 |                         0 |                    0.0000 |                 0.0000 |                0.0000 |          3716.9294 |               3716.9294 |
| ens_champion_segmented6               |       3 |    120 |          7850.8142 |               8196.1948 |         345.3806 |                        73 |                    0.6083 |                 0.0000 |                0.0333 |          3716.9294 |               3371.5488 |
| ens_champion_segmented6               |       5 |    120 |          7850.8142 |               8373.5833 |         522.7691 |                        91 |                    0.7583 |                 0.0000 |                0.0417 |          3716.9294 |               3194.1603 |
| ens_champion_segmented6               |      10 |    120 |          7850.8142 |               8578.3433 |         727.5291 |                       101 |                    0.8417 |                 0.0000 |                0.0500 |          3716.9294 |               2989.4004 |
| lgb_segmented_6_last_180d             |       1 |    120 |          7739.5498 |               7739.5498 |           0.0000 |                         0 |                    0.0000 |                 0.0000 |                0.0000 |          3828.1938 |               3828.1938 |
| lgb_segmented_6_last_180d             |       3 |    120 |          7739.5498 |               8051.1590 |         311.6091 |                        75 |                    0.6250 |                 0.0000 |                0.0083 |          3828.1938 |               3516.5847 |
| lgb_segmented_6_last_180d             |       5 |    120 |          7739.5498 |               8198.9439 |         459.3941 |                        86 |                    0.7167 |                 0.0000 |                0.0083 |          3828.1938 |               3368.7997 |
| lgb_segmented_6_last_180d             |      10 |    120 |          7739.5498 |               8400.9740 |         661.4241 |                        99 |                    0.8250 |                 0.0000 |                0.0250 |          3828.1938 |               3166.7697 |
| lgb_segmented_6_margin_core_last_180d |       1 |    120 |          7725.8973 |               7725.8973 |           0.0000 |                         0 |                    0.0000 |                 0.0000 |                0.0000 |          3841.8463 |               3841.8463 |
| lgb_segmented_6_margin_core_last_180d |       3 |    120 |          7725.8973 |               8017.9582 |         292.0609 |                        67 |                    0.5583 |                 0.0000 |                0.0167 |          3841.8463 |               3549.7854 |
| lgb_segmented_6_margin_core_last_180d |       5 |    120 |          7725.8973 |               8156.3604 |         430.4630 |                        82 |                    0.6833 |                 0.0000 |                0.0250 |          3841.8463 |               3411.3833 |
| lgb_segmented_6_margin_core_last_180d |      10 |    120 |          7725.8973 |               8436.0045 |         710.1072 |                        98 |                    0.8167 |                 0.0000 |                0.0333 |          3841.8463 |               3131.7391 |

## Top-10 Interpretation

| model                                 |   top1_mean_profit |   topk_best_mean_profit |   topk_lift_mean |   topk_lift_positive_rate |   oracle_in_topk_rate |   top1_mean_regret |   topk_best_mean_regret |
|:--------------------------------------|-------------------:|------------------------:|-----------------:|--------------------------:|----------------------:|-------------------:|------------------------:|
| ens_champion_segmented6               |          7850.8142 |               8578.3433 |         727.5291 |                    0.8417 |                0.0500 |          3716.9294 |               2989.4004 |
| lgb_segmented_6_last_180d             |          7739.5498 |               8400.9740 |         661.4241 |                    0.8250 |                0.0250 |          3828.1938 |               3166.7697 |
| lgb_segmented_6_margin_core_last_180d |          7725.8973 |               8436.0045 |         710.1072 |                    0.8167 |                0.0333 |          3841.8463 |               3131.7391 |

## Decision

- Predicted top-K contains materially better pairs than top1 on average.
- Next step is a conservative reranker diagnostic using only ex-ante features.
