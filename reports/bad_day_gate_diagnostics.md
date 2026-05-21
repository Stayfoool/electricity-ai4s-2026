# Bad-Day Gate Diagnostics

## Scope

This report asks whether current champion-prior bad days can be identified before seeing actual prices. It uses only prediction-stage features for rules, then evaluates against validation labels.

Labels inspected:

- `loss_day`: champion top1 profit < 0.
- `high_regret_top25`: top quartile of regret.
- `high_oracle10_lift_top25`: top quartile of available top-10 improvement.
- `bottom_profit25`: bottom quartile of top1 profit.

## Baseline Bad-Day Counts

| days | loss_days | high_regret_days | bottom_profit_days | mean_top1_profit | mean_oracle10_lift |
| --- | --- | --- | --- | --- | --- |
| 120.0000 | 4.0000 | 30.0000 | 30.0000 | 7991.3509 | 718.2329 |

## Feature Screen Top Signals

| label | feature | auc | auc_distance | spearman | mean_on_label_true | mean_on_label_false | direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bottom_profit25 | top1_weather_bid_spread | 0.2944 | 0.2056 | -0.3083 | 0.6905 | 1.0046 | lower_is_risk |
| bottom_profit25 | ghi_charge_mean | 0.2994 | 0.2006 | -0.3009 | 351.6595 | 445.3698 | lower_is_risk |
| bottom_profit25 | top10_weather_spread_mean | 0.3015 | 0.1985 | -0.2978 | 0.7055 | 1.0071 | lower_is_risk |
| bottom_profit25 | top1_ghi_delta | 0.6956 | 0.1956 | 0.2933 | -286.5512 | -368.0354 | higher_is_risk |
| bottom_profit25 | top10_weather_spread_max | 0.3096 | 0.1904 | -0.2856 | 0.7945 | 1.0829 | lower_is_risk |
| bottom_profit25 | top1_orig_bid_spread | 0.3122 | 0.1878 | -0.2817 | 1.0815 | 1.5193 | lower_is_risk |
| bottom_profit25 | top10_prior_score_min | 0.3167 | 0.1833 | -0.2750 | 4.4929 | 6.3524 | lower_is_risk |
| bottom_profit25 | rank10_prior_score | 0.3167 | 0.1833 | -0.2750 | 4.4929 | 6.3524 | lower_is_risk |
| high_oracle10_lift_top25 | top1_orig_bid_spread | 0.2978 | 0.2022 | -0.3033 | 1.0672 | 1.5241 | lower_is_risk |
| high_oracle10_lift_top25 | top10_weather_spread_mean | 0.3330 | 0.1670 | -0.2506 | 0.7490 | 0.9925 | lower_is_risk |
| high_oracle10_lift_top25 | top1_rank2_pred_gap | 0.3330 | 0.1670 | -0.2506 | -0.0049 | 0.0375 | lower_is_risk |
| high_oracle10_lift_top25 | top1_top2_gap | 0.3330 | 0.1670 | -0.2506 | -0.0049 | 0.0375 | lower_is_risk |
| high_oracle10_lift_top25 | top10_weather_spread_max | 0.3400 | 0.1600 | -0.2400 | 0.8308 | 1.0708 | lower_is_risk |
| high_oracle10_lift_top25 | top1_weather_bid_spread | 0.3419 | 0.1581 | -0.2372 | 0.7558 | 0.9829 | lower_is_risk |
| high_oracle10_lift_top25 | top10_pred_spread_range | 0.6541 | 0.1541 | 0.2311 | 0.4063 | 0.2667 | higher_is_risk |
| high_oracle10_lift_top25 | wind_charge_mean | 0.3470 | 0.1530 | -0.2295 | 4.9248 | 5.5872 | lower_is_risk |
| high_regret_top25 | wind_discharge_mean | 0.2763 | 0.2237 | -0.3356 | 4.8257 | 5.7339 | lower_is_risk |
| high_regret_top25 | top10_pred_spread_std | 0.7111 | 0.2111 | 0.3167 | 0.1790 | 0.1162 | higher_is_risk |
| high_regret_top25 | top1_orig_bid_minus_pred | 0.2900 | 0.2100 | -0.3150 | -8.2971 | -5.9900 | lower_is_risk |
| high_regret_top25 | wind_charge_mean | 0.2911 | 0.2089 | -0.3133 | 4.7073 | 5.6597 | lower_is_risk |
| high_regret_top25 | top10_prior_score_std | 0.6907 | 0.1907 | 0.2861 | 0.1211 | 0.0710 | higher_is_risk |
| high_regret_top25 | top1_weather_minus_pred | 0.3115 | 0.1885 | -0.2828 | -8.6012 | -6.5337 | lower_is_risk |
| high_regret_top25 | top10_pred_spread_range | 0.6881 | 0.1881 | 0.2822 | 0.4263 | 0.2600 | higher_is_risk |
| high_regret_top25 | top1_rank10_prior_gap | 0.6848 | 0.1848 | 0.2772 | 0.3403 | 0.2096 | higher_is_risk |
| loss_day | top10_weather_spread_max | 0.0453 | 0.4547 | -0.2828 | 0.3842 | 1.0324 | lower_is_risk |
| loss_day | top10_weather_spread_mean | 0.0647 | 0.4353 | -0.2707 | 0.2867 | 0.9539 | lower_is_risk |
| loss_day | top1_weather_bid_spread | 0.0668 | 0.4332 | -0.2694 | 0.2868 | 0.9481 | lower_is_risk |
| loss_day | top1_orig_bid_spread | 0.0690 | 0.4310 | -0.2680 | 0.3734 | 1.4456 | lower_is_risk |
| loss_day | top1_ghi_delta | 0.8815 | 0.3815 | 0.2372 | -128.3738 | -355.2261 | higher_is_risk |
| loss_day | top10_prior_score_std | 0.8017 | 0.3017 | 0.1876 | 0.1622 | 0.0808 | higher_is_risk |
| loss_day | top1_rank10_prior_gap | 0.8017 | 0.3017 | 0.1876 | 0.4495 | 0.2352 | higher_is_risk |
| loss_day | top10_prior_score_range | 0.8017 | 0.3017 | 0.1876 | 0.4495 | 0.2352 | higher_is_risk |

## Cross-Fold Gate Result

| method | days | mean_profit | worst_profit | loss_days | lift_vs_top1 |
| --- | --- | --- | --- | --- | --- |
| top1 | 120 | 7991.3509 | -15925.8974 | 4 | 0.0000 |
| bad_day_gate_cv | 120 | 7944.7791 | -15925.8974 | 4 | -46.5718 |

## Learned Rules By Held-Out Fold

| heldout_fold | action | feature | op | threshold | train_mean_profit | train_lift_vs_top1 | train_loss_days | train_mask_days | valid_mask_days | valid_filtered_loss_days | valid_filtered_profit_days | top1_mean_profit | top1_worst_profit | top1_loss_days | gate_mean_profit | gate_worst_profit | gate_loss_days | valid_lift_vs_top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | rank10 | top10_weather_spread_mean | <= | 0.3400 | 7097.0140 | 115.8170 | 2 | 10 | 2 | 0 | 2 | 11310.4282 | 0.0000 | 0 | 11327.5504 | 0.0000 | 0 | 17.1222 |
| valid_2025_10 | rank10 | top1_wind_delta | <= | -0.2887 | 8074.1856 | 50.7035 | 1 | 18 | 5 | 1 | 4 | 7899.1033 | -15925.8974 | 3 | 7703.3600 | -15925.8974 | 3 | -195.7432 |
| valid_2025_11 | rank10 | top10_weather_spread_mean | <= | 0.3166 | 8367.0610 | 85.5153 | 2 | 9 | 0 | 0 | 0 | 7120.7667 | 613.3252 | 0 | 7120.7667 | 613.3252 | 0 | 0.0000 |
| valid_2025_12 | rank10 | top10_weather_spread_mean | <= | 0.3127 | 8796.4446 | 86.4761 | 1 | 9 | 0 | 0 | 0 | 5928.2233 | -12550.1225 | 1 | 5928.2233 | -12550.1225 | 1 | 0.0000 |

## Triggered Days

| date | fold | action | feature | op | threshold | feature_value | top1_profit | gate_profit | lift_vs_top1 | top1_regret | oracle10_lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-10-01 | valid_2025_10 | rank10 | top1_wind_delta | <= | -0.2887 | -0.3274 | 6737.9685 | -512.1006 | -7250.0691 | 4215.4603 | 4103.4096 |
| 2025-10-29 | valid_2025_10 | rank10 | top1_wind_delta | <= | -0.2887 | -1.7896 | 18469.7461 | 18042.8710 | -426.8751 | 0.0000 | 0.0000 |
| 2025-10-04 | valid_2025_10 | rank10 | top1_wind_delta | <= | -0.2887 | -1.1050 | 10029.5359 | 9800.4778 | -229.0581 | 4314.6838 | 1043.4867 |
| 2025-09-25 | valid_2025_09 | rank10 | top10_weather_spread_mean | <= | 0.3400 | 0.1059 | 11160.3573 | 11288.0731 | 127.7158 | 2898.6819 | 127.7158 |
| 2025-10-16 | valid_2025_10 | rank10 | top1_wind_delta | <= | -0.2887 | -0.5112 | 407.0743 | 627.9722 | 220.8979 | 4161.4917 | 698.1021 |
| 2025-09-21 | valid_2025_09 | rank10 | top10_weather_spread_mean | <= | 0.3400 | 0.1271 | 459.2582 | 810.9641 | 351.7059 | 3354.6633 | 351.7059 |
| 2025-10-15 | valid_2025_10 | rank10 | top1_wind_delta | <= | -0.2887 | -0.5409 | -132.5324 | 1484.5316 | 1617.0640 | 14618.6535 | 1617.0640 |

## Decision

- The current simple gate is not strong enough to replace champion dispatch.
- Keep the feature screen as evidence for the next targeted bad-day model.

## Artifacts

- `reports/bad_day_gate_dataset.csv`
- `reports/bad_day_gate_feature_screen.csv`
- `reports/bad_day_gate_rules.csv`
- `reports/bad_day_gate_daily.csv`
- `reports/bad_day_gate_summary.csv`
