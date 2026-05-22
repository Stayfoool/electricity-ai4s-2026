# Robust Loss Experiment

## Purpose

External discussions suggested replacing MSE with robust losses such as Huber or MAE because electricity prices contain spikes. Our champion already uses LightGBM `huber`, so this experiment is a loss-function ablation and blend test, not a first-time Huber trial.

## Configs

| name | config |
| --- | --- |
| champion | `configs/ensemble_champion_segmented6_prior.yaml` |
| l2 | `configs/ensemble_champion_segmented6_prior_l2.yaml` |
| l1 | `configs/ensemble_champion_segmented6_prior_l1.yaml` |
| huber_alpha05 | `configs/ensemble_champion_segmented6_prior_huber_alpha05.yaml` |
| huber_alpha07 | `configs/ensemble_champion_segmented6_prior_huber_alpha07.yaml` |
| huber_alpha12 | `configs/ensemble_champion_segmented6_prior_huber_alpha12.yaml` |
| blend_l1_20 | `configs/ensemble_champion_segmented6_prior_blend_l1_20.yaml` |
| blend_l2_20 | `configs/ensemble_champion_segmented6_prior_blend_l2_20.yaml` |
| blend_l1_l2_15 | `configs/ensemble_champion_segmented6_prior_blend_l1_l2_15.yaml` |

## Summary

| model | fold_mean_profit | day_weighted_profit | worst_fold_profit | loss_days | oracle_ratio_mean | mae_mean | rmse_mean | curve_z_rmse_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| champion | 8064.630 | 7991.351 | 5928.223 | 5 | 0.707 | 0.438 | 0.693 | 0.850 |
| blend_l1_20 | 8045.858 | 7972.219 | 5946.095 | 5 | 0.705 | 0.435 | 0.698 | 0.848 |
| blend_l1_l2_15 | 8032.476 | 7958.863 | 5867.008 | 5 | 0.705 | 0.435 | 0.694 | 0.847 |
| blend_l2_20 | 7992.909 | 7918.044 | 5896.284 | 5 | 0.702 | 0.437 | 0.690 | 0.848 |
| huber_alpha07 | 7809.762 | 7741.943 | 5987.434 | 5 | 0.683 | 0.440 | 0.699 | 0.857 |
| huber_alpha05 | 7796.474 | 7735.469 | 5922.993 | 4 | 0.688 | 0.437 | 0.703 | 0.853 |
| l2 | 7773.304 | 7709.244 | 5851.304 | 5 | 0.686 | 0.437 | 0.682 | 0.844 |
| huber_alpha12 | 7757.200 | 7693.103 | 5908.493 | 4 | 0.681 | 0.440 | 0.691 | 0.851 |
| l1 | 7638.432 | 7592.474 | 5898.656 | 6 | 0.676 | 0.434 | 0.732 | 0.859 |

## Fold Delta vs Champion

| fold | l2 | l1 | huber_alpha05 | huber_alpha07 | huber_alpha12 | blend_l1_20 | blend_l2_20 | blend_l1_l2_15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid_2025_09 | -747.671 | -1624.985 | -838.160 | -393.502 | -676.283 | +4.177 | -32.195 | -36.533 |
| valid_2025_10 | -312.138 | +58.141 | -198.150 | -190.973 | -231.034 | -52.702 | -222.750 | -51.892 |
| valid_2025_11 | -28.576 | -108.381 | -31.086 | -494.210 | -302.672 | -44.437 | +0.000 | +21.024 |
| valid_2025_12 | -76.920 | -29.567 | -5.231 | +59.210 | -19.731 | +17.872 | -31.940 | -61.216 |

## Conclusions

- Keep the current champion. No robust-loss variant or fixed blend beats it on day-weighted profit.
- Plain `l2` improves RMSE but loses profit, confirming that lower point error alone is not enough for this competition.
- Plain `l1` improves MAE but is worse on profit and loss days. It can help some October days, but hurts September heavily.
- Huber alpha tuning does not improve the global result. Default Huber remains the best single loss setting.
- Fixed blending is close but still below champion. `blend_l1_20` is the closest candidate and can remain as a future selector/gating component, but should not replace champion.

## Next Direction

The useful signal is not a better global loss. It is conditional complementarity: some loss variants pick better windows on specific days. The next worthwhile experiment is a leak-free selector/gating model that decides when to trust champion vs a conservative blend using only prediction-time features and prediction-confidence features.
