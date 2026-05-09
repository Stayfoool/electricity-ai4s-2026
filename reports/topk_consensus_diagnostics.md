# Top-K Consensus Diagnostics

## Scope

Use the existing predicted top-10 legal charge/discharge pairs from three experts:

- `ens_champion_segmented6`
- `lgb_segmented_6_last_180d`
- `lgb_segmented_6_margin_core_last_180d`

No new price model is trained. Each rule chooses one pair per day using only candidate ranks, predicted spreads, and cross-expert agreement.

## Overall Result

| rule                      |   days |   mean_profit |   worst_profit |   loss_days |   mean_lift_vs_champion |   positive_lift_days |   negative_lift_days |   changed_days |   mean_support_count |   oracle_hit_rate |   mean_miss_vs_oracle_topk |
|:--------------------------|-------:|--------------:|---------------:|------------:|------------------------:|---------------------:|---------------------:|---------------:|---------------------:|------------------:|---------------------------:|
| unanimous_then_rank       |    120 |     7924.3558 |    -14029.8057 |           4 |                 73.5416 |                   26 |                   17 |             53 |               2.6500 |            0.0000 |                   963.7739 |
| champion_anchored_support |    120 |     7865.5126 |    -15992.7060 |           5 |                 14.6984 |                    4 |                    2 |              8 |               2.4417 |            0.0000 |                  1022.6171 |
| champion_top1             |    120 |     7850.8142 |    -15992.7060 |           5 |                  0.0000 |                    0 |                    0 |              0 |               2.3500 |            0.0000 |                  1037.3155 |
| support2_then_rank        |    120 |     7808.6140 |    -14029.8057 |           4 |                -42.2002 |                   29 |                   28 |             69 |               2.7417 |            0.0000 |                  1079.5157 |
| support_then_rank         |    120 |     7783.4277 |    -14029.8057 |           4 |                -67.3865 |                   29 |                   30 |             71 |               2.7417 |            0.0000 |                  1104.7020 |
| rank_score                |    120 |     7700.0353 |    -14029.8057 |           5 |               -150.7788 |                   26 |                   31 |             69 |               2.6667 |            0.0000 |                  1188.0944 |

## Fold Result

| fold          | rule                      |   days |   mean_profit |   worst_profit |   loss_days |   mean_lift_vs_champion |   changed_days |
|:--------------|:--------------------------|-------:|--------------:|---------------:|------------:|------------------------:|---------------:|
| valid_2025_09 | champion_anchored_support |     28 |    10903.8450 |        95.4409 |           0 |                  5.7137 |              4 |
| valid_2025_09 | champion_top1             |     28 |    10898.1313 |        95.4409 |           0 |                  0.0000 |              0 |
| valid_2025_09 | rank_score                |     28 |    10739.5495 |       118.2417 |           0 |               -158.5818 |             24 |
| valid_2025_09 | support2_then_rank        |     28 |    10866.0851 |        95.4409 |           0 |                -32.0462 |             23 |
| valid_2025_09 | support_then_rank         |     28 |    10866.0851 |        95.4409 |           0 |                -32.0462 |             23 |
| valid_2025_09 | unanimous_then_rank       |     28 |    10954.5989 |        95.4409 |           0 |                 56.4676 |             19 |
| valid_2025_10 | champion_anchored_support |     31 |     7987.4758 |    -15992.7060 |           4 |                 26.0666 |              2 |
| valid_2025_10 | champion_top1             |     31 |     7961.4092 |    -15992.7060 |           4 |                  0.0000 |              0 |
| valid_2025_10 | rank_score                |     31 |     7543.1689 |    -14029.8057 |           4 |               -418.2403 |             18 |
| valid_2025_10 | support2_then_rank        |     31 |     7659.9466 |    -14029.8057 |           3 |               -301.4626 |             20 |
| valid_2025_10 | support_then_rank         |     31 |     7659.9466 |    -14029.8057 |           3 |               -301.4626 |             20 |
| valid_2025_10 | unanimous_then_rank       |     31 |     7965.0346 |    -14029.8057 |           3 |                  3.6254 |             17 |
| valid_2025_11 | champion_anchored_support |     30 |     6946.7779 |       890.7813 |           0 |                  4.8781 |              1 |
| valid_2025_11 | champion_top1             |     30 |     6941.8998 |       890.7813 |           0 |                  0.0000 |              0 |
| valid_2025_11 | rank_score                |     30 |     6918.7269 |       763.5268 |           0 |                -23.1729 |             14 |
| valid_2025_11 | support2_then_rank        |     30 |     6920.1494 |       763.5268 |           0 |                -21.7504 |             14 |
| valid_2025_11 | support_then_rank         |     30 |     6920.1494 |       763.5268 |           0 |                -21.7504 |             14 |
| valid_2025_11 | unanimous_then_rank       |     30 |     6982.0953 |       763.5268 |           0 |                 40.1954 |              6 |
| valid_2025_12 | champion_anchored_support |     31 |     5888.3472 |    -11244.2975 |           1 |                 20.9490 |              1 |
| valid_2025_12 | champion_top1             |     31 |     5867.3983 |    -11244.2975 |           1 |                  0.0000 |              0 |
| valid_2025_12 | rank_score                |     31 |     5867.6391 |     -8661.8462 |           1 |                  0.2409 |             13 |
| valid_2025_12 | support2_then_rank        |     31 |     6055.4992 |     -8661.8462 |           1 |                188.1009 |             12 |
| valid_2025_12 | support_then_rank         |     31 |     5958.0036 |     -8661.8462 |           1 |                 90.6054 |             14 |
| valid_2025_12 | unanimous_then_rank       |     31 |     6058.5482 |     -8661.8462 |           1 |                191.1500 |             11 |

## Decision

- At least one consensus rule materially beats champion top1.
- Next step is to validate the rule as a submit-time dispatch option.
