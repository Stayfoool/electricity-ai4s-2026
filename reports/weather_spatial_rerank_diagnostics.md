# Spatial Weather Rerank Diagnostics

## Scope

Keep champion price predictions unchanged. Enumerate champion top-10 legal pairs, then use spatially corrected bid-space spread only as a conservative rerank signal.

Parameters are selected cross-fold: each held-out month uses parameters chosen from the other validation months only.

## Overall Result

| method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_top1 | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| champion_prior_top1 | 120 | 7991.3509 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| spatial_weather_rerank_cv | 120 | 7989.4837 | -15925.8974 | 4 | -1.8672 | 1 | 1 | 4 | 1.0333 |
| oracle_top10_upper_bound | 120 | 8709.5839 | -9394.2116 | 2 | 718.2329 | 93 | 0 | 93 | 5.3000 |

## Fold Result

| method | fold | days | mean_profit | loss_days | changed_days | mean_lift_vs_top1 |
| --- | --- | --- | --- | --- | --- | --- |
| spatial_weather_rerank_cv | valid_2025_09 | 28 | 11310.4282 | 0 | 0 | 0.0000 |
| spatial_weather_rerank_cv | valid_2025_10 | 31 | 7899.1033 | 3 | 0 | 0.0000 |
| spatial_weather_rerank_cv | valid_2025_11 | 30 | 7120.7667 | 0 | 0 | 0.0000 |
| spatial_weather_rerank_cv | valid_2025_12 | 31 | 5920.9953 | 1 | 4 | -7.2281 |
| champion_prior_top1 | valid_2025_09 | 28 | 11310.4282 | 0 | 0 | 0.0000 |
| champion_prior_top1 | valid_2025_10 | 31 | 7899.1033 | 3 | 0 | 0.0000 |
| champion_prior_top1 | valid_2025_11 | 30 | 7120.7667 | 0 | 0 | 0.0000 |
| champion_prior_top1 | valid_2025_12 | 31 | 5928.2233 | 1 | 0 | 0.0000 |

## Selected Parameters By Held-Out Fold

| heldout_fold | signal | top_k | gamma | threshold | train_mean_profit | train_loss_days | train_changed_days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | spatial_raw | 3 | 0.0200 | 0.0000 | 6987.0756 | 4 | 1 |
| valid_2025_10 | spatial_raw | 3 | 0.0200 | 0.0000 | 8029.5589 | 1 | 1 |
| valid_2025_11 | spatial_raw | 3 | 0.0200 | 0.0000 | 8287.5549 | 4 | 1 |
| valid_2025_12 | spatial_resid | 3 | 0.1000 | 0.0000 | 8710.6833 | 3 | 1 |

## Decision

- Spatial weather rerank does not materially beat current champion-prior top1.
- Do not use this as final dispatch unless held-out lift is stable by fold.

## Artifacts

- `reports/weather_spatial_rerank_candidates.csv`
- `reports/weather_spatial_rerank_selected_daily.csv`
- `reports/weather_spatial_rerank_summary.csv`
- `reports/weather_spatial_rerank_by_fold.csv`
- `reports/weather_spatial_rerank_params.csv`
