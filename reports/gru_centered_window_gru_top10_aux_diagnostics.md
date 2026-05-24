# GRU Centered / Window GRU Top10 Auxiliary Diagnostics

## Scope

- Candidate pool: current champion top10 legal charge/discharge pairs, with dispatch prior.
- Auxiliary experts: GRU centered point curve and Window GRU 89-window scores.
- This is diagnostic only; no submit file is generated.

## Signal Checks

| metric | value |
| --- | --- |
| gru_global_in_anchor_top10_rate | 0.1167 |
| window_global_in_anchor_top10_rate | 0.1333 |
| gru_window_same_pick_rate_in_anchor_top10 | 0.2667 |
| mean_oracle_top10_lift | 718.2329 |
| positive_oracle_top10_lift_days | 93.0000 |

## Overall Methods

| method | days | mean_profit | min_profit | loss_days | changed_days | positive_changed_days | negative_changed_days | profit_delta_vs_anchor | total_lift_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_top10 | 120 | 8709.5839 | -9394.2116 | 2 | 93 | 93 | 0 | 718.2329 | 86187.9533 |
| anchor_top1 | 120 | 7991.3509 | -15925.8974 | 4 | 0 | 0 | 0 | -0.0000 | 0.0000 |
| consensus_if_same_else_anchor | 120 | 7852.3670 | -15925.8974 | 4 | 31 | 17 | 10 | -138.9840 | -16678.0763 |
| avg_aux_rank | 120 | 7803.8397 | -15992.7060 | 6 | 103 | 47 | 36 | -187.5112 | -22501.3460 |
| window_top10 | 120 | 7756.9632 | -16056.3332 | 7 | 111 | 52 | 40 | -234.3877 | -28126.5282 |
| gru_top10 | 120 | 7679.2451 | -12601.3742 | 7 | 117 | 48 | 38 | -312.1059 | -37452.7063 |

## By Fold

| fold | method | days | mean_profit | min_profit | loss_days | changed_days | positive_changed_days | negative_changed_days | profit_delta_vs_anchor | total_lift_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | anchor_top1 | 28 | 11310.4282 | 0.0000 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 |
| valid_2025_10 | anchor_top1 | 31 | 7899.1033 | -15925.8974 | 3 | 0 | 0 | 0 | -0.0000 | 0.0000 |
| valid_2025_11 | anchor_top1 | 30 | 7120.7667 | 613.3252 | 0 | 0 | 0 | 0 | -0.0000 | 0.0000 |
| valid_2025_12 | anchor_top1 | 31 | 5928.2233 | -12550.1225 | 1 | 0 | 0 | 0 | -0.0000 | 0.0000 |
| valid_2025_09 | avg_aux_rank | 28 | 10594.4387 | -254.5089 | 1 | 23 | 8 | 12 | -715.9895 | -20047.7047 |
| valid_2025_10 | avg_aux_rank | 31 | 7563.4309 | -15992.7060 | 4 | 29 | 15 | 12 | -335.6724 | -10405.8433 |
| valid_2025_11 | avg_aux_rank | 30 | 7161.7844 | 390.0445 | 0 | 26 | 11 | 8 | 41.0177 | 1230.5317 |
| valid_2025_12 | avg_aux_rank | 31 | 6145.0514 | -12359.0149 | 1 | 25 | 13 | 4 | 216.8281 | 6721.6704 |
| valid_2025_09 | consensus_if_same_else_anchor | 28 | 10924.1565 | 0.0000 | 0 | 9 | 7 | 2 | -386.2716 | -10815.6058 |
| valid_2025_10 | consensus_if_same_else_anchor | 31 | 7676.3520 | -15925.8974 | 3 | 14 | 7 | 6 | -222.7513 | -6905.2890 |
| valid_2025_11 | consensus_if_same_else_anchor | 30 | 7120.4189 | 613.3252 | 0 | 4 | 1 | 1 | -0.3478 | -10.4349 |
| valid_2025_12 | consensus_if_same_else_anchor | 31 | 5962.1993 | -12550.1225 | 1 | 4 | 2 | 1 | 33.9759 | 1053.2535 |
| valid_2025_09 | gru_top10 | 28 | 10461.6256 | -286.3226 | 1 | 26 | 9 | 12 | -848.8026 | -23766.4718 |
| valid_2025_10 | gru_top10 | 31 | 7404.3173 | -12601.3742 | 5 | 31 | 16 | 13 | -494.7859 | -15338.3635 |
| valid_2025_11 | gru_top10 | 30 | 6965.6890 | 453.6717 | 0 | 29 | 9 | 8 | -155.0777 | -4652.3313 |
| valid_2025_12 | gru_top10 | 31 | 6131.5930 | -12359.0149 | 1 | 31 | 14 | 5 | 203.3697 | 6304.4602 |
| valid_2025_09 | oracle_top10 | 28 | 11887.8514 | 127.2545 | 0 | 23 | 23 | 0 | 577.4232 | 16167.8495 |
| valid_2025_10 | oracle_top10 | 31 | 9346.9273 | -9224.9472 | 1 | 27 | 27 | 0 | 1447.8240 | 44882.5446 |
| valid_2025_11 | oracle_top10 | 30 | 7353.7912 | 676.9524 | 0 | 22 | 22 | 0 | 233.0245 | 6990.7340 |
| valid_2025_12 | oracle_top10 | 31 | 6513.6048 | -9394.2116 | 1 | 21 | 21 | 0 | 585.3815 | 18146.8251 |
| valid_2025_09 | window_top10 | 28 | 10571.7613 | -159.0681 | 1 | 25 | 12 | 13 | -738.6669 | -20682.6728 |
| valid_2025_10 | window_top10 | 31 | 7407.3862 | -16056.3332 | 5 | 29 | 14 | 13 | -491.7170 | -15243.2280 |
| valid_2025_11 | window_top10 | 30 | 7147.6639 | 378.6075 | 0 | 28 | 12 | 9 | 26.8972 | 806.9174 |
| valid_2025_12 | window_top10 | 31 | 6153.7864 | -11244.2975 | 1 | 29 | 14 | 5 | 225.5631 | 6992.4552 |

## Best Conservative Rules

| method | rule_family | anchor_gap_max | aux_delta_min | mean_profit | min_fold_profit | loss_days | changed_days | positive_changed_days | negative_changed_days | profit_delta_vs_anchor | positive_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg_aux_rank_rule_gap0.2_delta0 | avg_aux_rank_rule | 0.2000 | 0.0000 | 8079.1622 | 6144.3210 | 6 | 81 | 41 | 23 | 87.8112 | 3 |
| gru_rule_gap0.05_delta0.1 | gru_rule | 0.0500 | 0.1000 | 8072.0759 | 6073.7864 | 5 | 45 | 20 | 10 | 80.7250 | 4 |
| avg_aux_rank_rule_gap0.1_delta0 | avg_aux_rank_rule | 0.1000 | 0.0000 | 8067.8942 | 6068.4815 | 6 | 67 | 34 | 18 | 76.5432 | 3 |
| avg_aux_rank_rule_gap0.05_delta0 | avg_aux_rank_rule | 0.0500 | 0.0000 | 8060.5517 | 6057.1928 | 6 | 52 | 26 | 16 | 69.2007 | 3 |
| gru_rule_gap0.03_delta0.1 | gru_rule | 0.0300 | 0.1000 | 8059.6015 | 6019.3245 | 5 | 39 | 19 | 7 | 68.2505 | 4 |
| window_rule_gap0.2_delta0.01 | window_rule | 0.2000 | 0.0100 | 8058.9952 | 6129.0348 | 6 | 75 | 40 | 23 | 67.6442 | 3 |
| gru_rule_gap0.1_delta0.1 | gru_rule | 0.1000 | 0.1000 | 8056.3089 | 6073.7864 | 5 | 59 | 26 | 14 | 64.9580 | 3 |
| window_rule_gap0.05_delta0 | window_rule | 0.0500 | 0.0000 | 8051.6104 | 6054.4705 | 6 | 80 | 36 | 24 | 60.2595 | 3 |
| window_rule_gap0_delta0 | window_rule | 0.0000 | 0.0000 | 8050.7283 | 6050.7618 | 6 | 54 | 28 | 19 | 59.3774 | 3 |
| window_rule_gap0.1_delta0.01 | window_rule | 0.1000 | 0.0100 | 8050.2884 | 6047.0378 | 6 | 63 | 31 | 23 | 58.9375 | 3 |
| gru_rule_gap0.01_delta0.1 | gru_rule | 0.0100 | 0.1000 | 8049.6079 | 6005.9892 | 5 | 28 | 17 | 6 | 58.2569 | 4 |
| gru_rule_gap0.05_delta0.05 | gru_rule | 0.0500 | 0.0500 | 8047.8190 | 6090.2353 | 6 | 65 | 25 | 18 | 56.4680 | 2 |
| gru_rule_gap0.05_delta0.03 | gru_rule | 0.0500 | 0.0300 | 8047.0707 | 6090.2353 | 6 | 74 | 28 | 22 | 55.7198 | 2 |
| window_rule_gap0.1_delta0 | window_rule | 0.1000 | 0.0000 | 8046.4679 | 6060.7389 | 6 | 96 | 43 | 29 | 55.1169 | 3 |
| window_rule_gap0.01_delta0 | window_rule | 0.0100 | 0.0000 | 8046.1437 | 6038.3799 | 6 | 62 | 29 | 20 | 54.7928 | 3 |

## Decision Notes

- Some rules are positive, but current signal is below promotion threshold or risk constraints need review.
