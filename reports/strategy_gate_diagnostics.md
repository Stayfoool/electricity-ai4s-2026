# Strategy Gate Diagnostics

## Scope

This diagnostic evaluates a small day-level selector over three existing experts. Hidden true prices are used only to score validation days.

Experts: `ens_champion_segmented6, lgb_segmented_6_last_180d, lgb_segmented_6_margin_core_last_180d`.

## Overall Held-Out Result

| method          |   mean_profit |   lift_vs_champion |   worst_day_profit |   loss_days | selected_counts                                                                                  |
|:----------------|--------------:|-------------------:|-------------------:|------------:|:-------------------------------------------------------------------------------------------------|
| champion        |     7850.8142 |             0.0000 |        -15992.7060 |           5 | ens_champion_segmented6:120                                                                      |
| static          |     7850.8142 |             0.0000 |        -15992.7060 |           5 | ens_champion_segmented6:120                                                                      |
| rule            |     7860.0052 |             9.1910 |        -15992.7060 |           5 | lgb_segmented_6_margin_core_last_180d:75;ens_champion_segmented6:45                              |
| logistic        |     7836.1368 |           -14.6774 |        -15992.7060 |           6 | lgb_segmented_6_margin_core_last_180d:48;ens_champion_segmented6:37;lgb_segmented_6_last_180d:35 |
| oracle_selector |     8319.3451 |           468.5309 |        -12600.3721 |           5 |                                                                                                  |

## Fold Detail

| heldout_fold   |   valid_days | best_static_expert_from_train   | rule_feature                       |   rule_threshold | rule_left_expert                      | rule_right_expert                     |   rule_train_lift_vs_champion | logistic_mode   |   champion_mean_profit |   champion_worst_profit |   champion_oracle_ratio |   champion_loss_days |   static_mean_profit |   static_worst_profit |   static_oracle_ratio |   static_loss_days |   rule_gate_mean_profit |   rule_gate_worst_profit |   rule_gate_oracle_ratio |   rule_gate_loss_days |   logistic_gate_mean_profit |   logistic_gate_worst_profit |   logistic_gate_oracle_ratio |   logistic_gate_loss_days |   oracle_selector_mean_profit |   oracle_selector_worst_profit |   oracle_selector_oracle_ratio |   oracle_selector_loss_days |
|:---------------|-------------:|:--------------------------------|:-----------------------------------|-----------------:|:--------------------------------------|:--------------------------------------|------------------------------:|:----------------|-----------------------:|------------------------:|------------------------:|---------------------:|---------------------:|----------------------:|----------------------:|-------------------:|------------------------:|-------------------------:|-------------------------:|----------------------:|----------------------------:|-----------------------------:|-----------------------------:|--------------------------:|------------------------------:|-------------------------------:|-------------------------------:|----------------------------:|
| valid_2025_09  |           28 | ens_champion_segmented6         | discharge_start_std                |           1.1547 | lgb_segmented_6_margin_core_last_180d | ens_champion_segmented6               |                      219.1915 | logistic        |             10898.1313 |                 95.4409 |                  0.7589 |                    0 |           10898.1313 |               95.4409 |                0.7589 |                  0 |              10813.6472 |                 118.2417 |                   0.7530 |                     0 |                  10975.1396 |                     118.2417 |                       0.7642 |                         0 |                    11290.2442 |                       118.2417 |                         0.7862 |                           0 |
| valid_2025_10  |           31 | ens_champion_segmented6         | load_range                         |           0.2968 | ens_champion_segmented6               | lgb_segmented_6_margin_core_last_180d |                      201.5090 | logistic        |              7961.4092 |             -15992.7060 |                  0.5719 |                    4 |            7961.4092 |           -15992.7060 |                0.5719 |                  4 |               7941.7640 |              -15992.7060 |                   0.5704 |                     4 |                   7608.3579 |                  -15992.7060 |                       0.5465 |                         4 |                     8460.1189 |                    -12600.3721 |                         0.6077 |                           4 |
| valid_2025_11  |           30 | ens_champion_segmented6         | champion_segmented6_discharge_diff |           1.0000 | lgb_segmented_6_margin_core_last_180d | ens_champion_segmented6               |                      189.0586 | logistic        |              6941.8998 |                890.7813 |                  0.8669 |                    0 |            6941.8998 |              890.7813 |                0.8669 |                  0 |               6874.5288 |                   8.4274 |                   0.8585 |                     0 |                   6984.3734 |                     592.4364 |                       0.8722 |                         0 |                     7220.2037 |                       890.7813 |                         0.9016 |                           0 |
| valid_2025_12  |           31 | ens_champion_segmented6         | discharge_start_std                |           1.1547 | lgb_segmented_6_margin_core_last_180d | ens_champion_segmented6               |                      131.4771 | logistic        |              5867.3983 |             -11244.2975 |                  0.5789 |                    1 |            5867.3983 |           -11244.2975 |                0.5789 |                  1 |               6064.1275 |               -6342.6334 |                   0.5983 |                     1 |                   6052.9744 |                   -6342.6334 |                       0.5972 |                         2 |                     6558.8639 |                     -6342.6334 |                         0.6471 |                           1 |

## Oracle Selector Upper Bound

| fold          |   oracle_selector_lift_vs_champion | oracle_selected_counts                                                                          |
|:--------------|-----------------------------------:|:------------------------------------------------------------------------------------------------|
| valid_2025_09 |                           392.1130 | ens_champion_segmented6:14;lgb_segmented_6_last_180d:8;lgb_segmented_6_margin_core_last_180d:6  |
| valid_2025_10 |                           498.7097 | ens_champion_segmented6:17;lgb_segmented_6_last_180d:8;lgb_segmented_6_margin_core_last_180d:6  |
| valid_2025_11 |                           278.3038 | ens_champion_segmented6:15;lgb_segmented_6_margin_core_last_180d:12;lgb_segmented_6_last_180d:3 |
| valid_2025_12 |                           691.4656 | ens_champion_segmented6:18;lgb_segmented_6_margin_core_last_180d:9;lgb_segmented_6_last_180d:4  |

## Decision

- A held-out gate beats the current champion in this diagnostic.
- Next step is to implement a submit-time gate and validate the generated power schedule, not to replace the champion blindly.
