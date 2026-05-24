# Forecast Error Augmentation Experiment

## Purpose

Test whether historical boundary forecast residuals can be used to create robust training rows and reduce wrong-window / loss-day risk.

Residual definition: `forecast - actual`.

Validation is fold-safe: each fold samples residuals only from that fold's training frame. Validation rows are never perturbed.

## Implemented Variants

| label | channels | scope | shrink | augmented weight | intent |
|---|---|---|---:|---:|---|
| `feaug_full_v1` | wind, solar, non-market | all hours | 0.50 | 1.00 | first strong robustness probe |
| `feaug_ws025_w05` | wind, solar | all hours | 0.25 | 0.50 | lighter perturbation, lower augmented-row influence |
| `feaug_ws_late050_w05` | wind, solar | hours 10-18 | 0.50 | 0.50 | targeted renewable-window perturbation |

`风光总加预测值` is recomputed from `风电预测值 + 光伏预测值` after perturbation to preserve the physical identity present in the raw data.

## Overall Results

See `reports/forecast_error_augmentation_experiment_summary.csv` for the full table.

| label | day_weighted_profit | delta_vs_champion | loss_days | delta_loss_days | p90_regret | delta_p90_regret |
|---|---:|---:|---:|---:|---:|---:|
| champion | 9952.035 | 0.000 | 5 | 0 | 9990.909 | 0.000 |
| feaug_full_v1 | 9548.048 | -403.986 | 3 | -2 | 11256.118 | +1265.209 |
| feaug_ws025_w05 | 9650.494 | -301.540 | 5 | 0 | 10774.646 | +783.736 |
| feaug_ws_late050_w05 | 9591.477 | -360.558 | 5 | 0 | 11922.731 | +1931.822 |

## Readout

- Strong full augmentation reduced loss days from 5 to 3, so robustness augmentation can change bad-day behavior.
- All variants reduced average profit and worsened high-regret tail behavior.
- The lighter global variant slightly improves mean RMSE, but still loses profit, which confirms that lower point error does not guarantee better storage-window selection.
- The late-hours targeted variant helps September but loses too much in October, November, and Jan-Feb-like.

## Decision

Do not promote forecast-error augmentation as a global replacement for champion.

Keep the implementation as a reusable tool, but pause broad augmentation experiments. If reused, use it as:

1. a secondary expert for conservative bad-day selection;
2. a diagnostic stress-test for candidate stability;
3. a targeted month/hour experiment only when a specific failure mode is identified.

## Artifacts

- `src/electricity/features/forecast_error_augmentation.py`
- `configs/ensemble_champion_segmented6_prior_forecast_error_aug_5fold.yaml`
- `configs/ensemble_champion_segmented6_prior_feaug_ws025_w05_5fold.yaml`
- `configs/ensemble_champion_segmented6_prior_feaug_ws_late050_w05_5fold.yaml`
- `reports/forecast_error_augmentation_experiment_summary.csv`
- `reports/forecast_error_augmentation_experiment_by_fold.csv`
- `reports/backtest_ens_champion_segmented6_prior_forecast_error_aug_5fold.csv`
- `reports/backtest_ens_champion_segmented6_prior_feaug_ws025_w05_5fold.csv`
- `reports/backtest_ens_champion_segmented6_prior_feaug_ws_late050_w05_5fold.csv`
