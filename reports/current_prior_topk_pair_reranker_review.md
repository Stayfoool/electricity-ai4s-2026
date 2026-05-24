# Current Prior Top-K Pair Reranker Review

Date: 2026-05-22

## Conclusion

Do not use the current learned top-K reranker for submission.

The experiment is useful as a diagnostic: the champion top1 pair is not always the best pair inside its own top10 candidates, so there is theoretical room for local correction. However, the learned reranker does not reliably capture that room yet. The best learned variant, `gbr_lift`, improves total 5-fold validation profit by only `16.75` across `176` days, equal to `+0.095/day`. This is far below the noise level of our backtest and not enough to justify changing submit behavior.

## What Was Tested

Anchor:

- `ens_champion_segmented6_prior`

Candidate set:

- For each day, keep the anchor model's top10 legal charge/discharge window pairs.
- The default action remains the anchor top1 pair.
- The reranker may switch from top1 to another pair only if predicted lift exceeds a selected threshold.

Features:

- Candidate timing: charge start, discharge start, gap, rank.
- Anchor confidence: top1-top2 gap, top5 spread statistics.
- Business signals: bid_space, net_load, renewable_ratio, wind/solar/tie-line/non-market ratios.
- Weather residual signals: selected NWP month-hour residual features.
- Agreement signals from existing expert models.

Validation:

- Outer held-out fold evaluates the final decision.
- Inner out-of-fold predictions select the switching threshold.
- This is better than selecting thresholds directly on the outer validation fold.

## Result Summary

| method | mean_profit | total_profit | loss_days | changed_days | delta_vs_anchor |
| --- | ---: | ---: | ---: | ---: | ---: |
| `oracle_top10` | `10689.62` | `1881372.72` | `2` | `155` | `+737.58/day` |
| `gbr_lift` | `9952.13` | `1751574.85` | `4` | `7` | `+0.095/day` |
| `anchor_top1` | `9952.03` | `1751558.10` | `4` | `0` | `0.00/day` |
| `gbr_true_profit` | `9875.56` | `1738098.56` | `5` | `33` | `-76.47/day` |

Important interpretation:

- `oracle_top10` shows an upper bound: if we magically knew the true best pair inside the champion top10, we could gain about `+737.58/day`.
- `gbr_lift` shows the current learned selector: it captures almost none of that upside.
- `gbr_true_profit` is worse than anchor and increases loss days, so it is rejected.

## Changed-Day Audit

`gbr_lift` changed only 7 days:

| date | fold | selected_rank | lift_vs_anchor |
| --- | --- | ---: | ---: |
| 2025-01-11 | `valid_2025_jan_feb` | 8 | `-52.06` |
| 2025-01-26 | `valid_2025_jan_feb` | 6 | `0.00` |
| 2025-02-02 | `valid_2025_jan_feb` | 3 | `0.00` |
| 2025-02-03 | `valid_2025_jan_feb` | 2 | `+31.81` |
| 2025-02-09 | `valid_2025_jan_feb` | 6 | `-31.81` |
| 2025-02-12 | `valid_2025_jan_feb` | 7 | `+68.83` |
| 2025-02-24 | `valid_2025_jan_feb` | 4 | `-0.02` |

Net lift on changed days: `+16.75`.

This is not a robust pattern. It is a small number of switches, all in the Jan-Feb-like fold, with both positive and negative changes.

## Leakage Review

No direct target leakage was found in the reranker feature columns.

Explicitly blocked from features:

- `true_profit`
- `oracle_profit`
- `regret_vs_oracle`
- `is_oracle_pair`
- `top1_true_profit`
- `target_lift`
- `target`

Relevant code:

- `scripts/analyze_current_prior_topk_pair_reranker.py`
- `feature_columns()` excludes the label and oracle columns.
- `crossfold_evaluate()` uses outer held-out folds.
- `_inner_oof_predictions()` creates inner OOF predictions for threshold selection.

One residual risk remains:

- NWP residual features are computed as `value - month_hour_mean`.
- The current implementation computes month-hour means on the full frame passed into `build_feature_frame()`.
- In validation backtests, this can use validation-period feature distribution when normalizing validation rows.
- This is not label leakage because it does not use true price, but it is a mild transductive feature-normalization leak.
- Because the measured reranker lift is only `+0.095/day`, even this mild risk is enough to treat the result as non-actionable.

Relevant code:

- `src/electricity/features/time_features.py`
- `add_nwp_residual_features()`
- `groupby(["month", "hour"]).transform("mean")`

## Decision

Do not generate a submit file from this reranker.

Keep the artifacts as diagnostics:

- `reports/current_prior_topk_pair_reranker_diagnostics.md`
- `reports/current_prior_topk_pair_reranker_overall.csv`
- `reports/current_prior_topk_pair_reranker_by_fold.csv`
- `reports/current_prior_topk_pair_reranker_by_regime.csv`
- `reports/current_prior_topk_pair_reranker_dataset.csv`
- `reports/current_prior_topk_pair_reranker_selected_daily.csv`
- `reports/current_prior_topk_pair_reranker_params.csv`

## Recommended Next Experiment

Do this before trying another high-capacity reranker:

1. Make NWP residual generation fold-safe.
2. Rerun the current top-K reranker.
3. If the tiny lift disappears, discard this direction for now.
4. If the top10 oracle gap remains large, test a conservative gate instead of a full reranker.

Conservative gate design:

- Default to champion top1.
- Switch only when multiple independent signals agree.
- Require a large predicted lift threshold.
- Penalize changes on days where top1/top2 predicted spread is close.
- Evaluate primarily on Jan-Feb-like fold and loss-day behavior, not just all-fold mean profit.

The next practical target is not "select the best pair every day". The next target should be "avoid a small number of obviously bad champion top1 choices without damaging normal days".
