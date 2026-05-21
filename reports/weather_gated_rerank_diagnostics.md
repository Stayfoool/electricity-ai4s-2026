# Weather Gated Rerank Diagnostics

## Scope

Keep current champion-prior price predictions unchanged. Enumerate the predicted top-10 legal charge/discharge pairs using the same dispatch prior as the champion. Then test whether weather-corrected bid-space can conservatively rerank candidates inside the top-K set.

The parameter choice is cross-fold: for each held-out validation month, `top_k`, `gamma`, and `threshold` are selected using the other validation months only.

## Overall Result

| method | days | mean_profit | worst_profit | loss_days | mean_lift_vs_top1 | positive_lift_days | negative_lift_days | changed_days | mean_selected_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| champion_prior_top1 | 120 | 7991.3509 | -15925.8974 | 4 | 0.0000 | 0 | 0 | 0 | 1.0000 |
| weather_gated_rerank_cv | 120 | 7989.4837 | -15925.8974 | 4 | -1.8672 | 1 | 1 | 3 | 1.0250 |
| oracle_top10_upper_bound | 120 | 8709.5839 | -9394.2116 | 2 | 718.2329 | 93 | 0 | 93 | 5.3000 |

## Fold Result

| method | fold | days | mean_profit | loss_days | changed_days | mean_lift_vs_top1 |
| --- | --- | --- | --- | --- | --- | --- |
| weather_gated_rerank_cv | valid_2025_09 | 28 | 11310.4282 | 0 | 0 | 0.0000 |
| weather_gated_rerank_cv | valid_2025_10 | 31 | 7899.1033 | 3 | 0 | 0.0000 |
| weather_gated_rerank_cv | valid_2025_11 | 30 | 7120.7667 | 0 | 0 | 0.0000 |
| weather_gated_rerank_cv | valid_2025_12 | 31 | 5920.9953 | 1 | 3 | -7.2281 |
| champion_prior_top1 | valid_2025_09 | 28 | 11310.4282 | 0 | 0 | 0.0000 |
| champion_prior_top1 | valid_2025_10 | 31 | 7899.1033 | 3 | 0 | 0.0000 |
| champion_prior_top1 | valid_2025_11 | 30 | 7120.7667 | 0 | 0 | 0.0000 |
| champion_prior_top1 | valid_2025_12 | 31 | 5928.2233 | 1 | 0 | 0.0000 |

## Selected Parameters By Held-Out Fold

| heldout_fold | top_k | gamma | threshold | train_mean_profit | train_loss_days | train_changed_days |
| --- | --- | --- | --- | --- | --- | --- |
| valid_2025_09 | 3 | 0.0500 | 0.0000 | 6987.0756 | 4 | 2 |
| valid_2025_10 | 3 | 0.0500 | 0.0000 | 8029.5589 | 1 | 2 |
| valid_2025_11 | 3 | 0.0500 | 0.0000 | 8287.5549 | 4 | 2 |
| valid_2025_12 | 3 | 0.1000 | 0.0000 | 8710.6833 | 3 | 1 |

## Decision

- Weather gated rerank does not materially beat current champion-prior top1.
- Keep it as diagnostic evidence; do not change champion dispatch yet.

## Artifacts

- `reports/weather_gated_rerank_candidates.csv`
- `reports/weather_gated_rerank_selected_daily.csv`
- `reports/weather_gated_rerank_summary.csv`
- `reports/weather_gated_rerank_by_fold.csv`
- `reports/weather_gated_rerank_params.csv`
