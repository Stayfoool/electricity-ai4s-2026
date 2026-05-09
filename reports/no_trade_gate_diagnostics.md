# No-Trade Gate Diagnostics

## Scope

Default action is the current champion trade. A learned rule may set a whole day to no-trade, giving that day profit `0`.

## Overall Held-Out Result

| method        |   mean_profit |   lift_vs_champion |   worst_day_profit |   loss_days |   no_trade_days |
|:--------------|--------------:|-------------------:|-------------------:|------------:|----------------:|
| champion      |     7850.8142 |             0.0000 |        -15992.7060 |           5 |               0 |
| no_trade_gate |     7415.7460 |          -435.0682 |        -15992.7060 |           5 |               4 |

## Fold Detail

| heldout_fold   |   valid_days | rule_kind   | feature_a           | op_a   |   threshold_a | feature_b                       | op_b   |   threshold_b |   train_lift_vs_champion |   train_no_trade_days |   valid_no_trade_days |   valid_filtered_loss_days |   valid_filtered_profit_days |   champion_mean_profit |   champion_worst_profit |   champion_loss_days |   champion_no_trade_days |   no_trade_gate_mean_profit |   no_trade_gate_worst_profit |   no_trade_gate_loss_days |   no_trade_gate_no_trade_days |   valid_lift_vs_champion |
|:---------------|-------------:|:------------|:--------------------|:-------|--------------:|:--------------------------------|:-------|--------------:|-------------------------:|----------------------:|----------------------:|---------------------------:|-----------------------------:|-----------------------:|------------------------:|---------------------:|-------------------------:|----------------------------:|-----------------------------:|--------------------------:|------------------------------:|-------------------------:|
| valid_2025_09  |           28 | pair        | net_load_range      | <=     |        1.1749 | champion_segmented6_charge_diff | >      |        0.0000 |                 304.2934 |                     4 |                     2 |                          0 |                            2 |             10898.1313 |                 95.4409 |                    0 |                        0 |                   9266.4744 |                       0.0000 |                         0 |                             2 |               -1631.6569 |
| valid_2025_10  |           31 | pair        | load_mean           | >      |        3.0935 | net_load_std                    | <=     |        0.4577 |                 126.3404 |                     1 |                     0 |                          0 |                            0 |              7961.4092 |             -15992.7060 |                    4 |                        0 |                   7961.4092 |                  -15992.7060 |                         4 |                             0 |                   0.0000 |
| valid_2025_11  |           30 | pair        | renewable_ratio_std | <=     |        0.1282 | charge_start_std                | >      |        0.5774 |                 320.9531 |                     3 |                     2 |                          0 |                            2 |              6941.8998 |                890.7813 |                    0 |                        0 |                   6724.5068 |                       0.0000 |                         0 |                             2 |                -217.3931 |
| valid_2025_12  |           31 | pair        | load_mean           | <=     |        2.8947 | renewable_ratio_std             | <=     |        0.1251 |                 198.2189 |                     2 |                     0 |                          0 |                            0 |              5867.3983 |             -11244.2975 |                    1 |                        1 |                   5867.3983 |                  -11244.2975 |                         1 |                             1 |                   0.0000 |

## No-Trade Days

| date       | fold          |   champion_profit |   lift_vs_champion | rule_kind   | feature_a           |   feature_a_value |   threshold_a | feature_b                       |   feature_b_value |   threshold_b |
|:-----------|:--------------|------------------:|-------------------:|:------------|:--------------------|------------------:|--------------:|:--------------------------------|------------------:|--------------:|
| 2025-09-16 | valid_2025_09 |        31713.8509 |        -31713.8509 | pair        | net_load_range      |            1.0259 |        1.1749 | champion_segmented6_charge_diff |            1.0000 |        0.0000 |
| 2025-09-20 | valid_2025_09 |        13972.5412 |        -13972.5412 | pair        | net_load_range      |            1.0754 |        1.1749 | champion_segmented6_charge_diff |            1.0000 |        0.0000 |
| 2025-11-05 | valid_2025_11 |         5631.0105 |         -5631.0105 | pair        | renewable_ratio_std |            0.1088 |        0.1282 | charge_start_std                |           27.7128 |        0.5774 |
| 2025-11-06 | valid_2025_11 |          890.7813 |          -890.7813 | pair        | renewable_ratio_std |            0.0884 |        0.1282 | charge_start_std                |            3.0000 |        0.5774 |

## Decision

- The held-out no-trade gate does not improve mean profit.
- Do not promote no-trade filtering yet.
