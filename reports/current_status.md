# Current Status

Last updated: 2026-05-21 Asia/Shanghai.


## Latest Update: Current Prior Top-K Consensus

Completed on 2026-05-22.

Artifacts:

- `scripts/analyze_current_prior_topk_consensus.py`
- `reports/current_prior_topk_consensus_diagnostics.md`
- `reports/current_prior_topk_consensus_summary.csv`
- `reports/current_prior_topk_consensus_by_regime.csv`
- `reports/current_prior_topk_pair_summary.csv`

Setup:

- Anchor model: `configs/ensemble_champion_segmented6_prior_5fold.yaml`.
- Candidate set is restricted to the current champion-prior model's own top-10 legal charge/discharge pairs.
- Other experts only provide support/rank signals; they cannot introduce pairs outside anchor top10.
- Fold-specific dispatch priors are rebuilt exactly as in the champion-prior backtest.

Main findings:

| rule | all_5fold mean profit | lift vs anchor | loss days |
|---|---:|---:|---:|
| anchor_top1 | `9952.035` | `0.000` | 4 |
| support2_then_anchor_rank | `9876.130` | `-75.905` | 4 |
| anchor_rank_le5_support2 | `9855.947` | `-96.088` | 4 |
| support_score | `9808.230` | `-143.804` | 3 |

Decision:

- Current champion-prior top10 contains upside: oracle top10 upper bound is `+737.58` over anchor top1.
- The tested support/consensus rules do not capture that upside and underperform anchor top1 in all major regimes, including Jan-Feb-like.
- Do not generate or submit a current-prior topK consensus file.
- If revisiting topK, use a stronger pairwise/rerank learner with stricter out-of-fold validation, not simple expert-support rules.

## Latest Update: Online Submission Feedback

Completed on 2026-05-22.

Submitted candidates:

| candidate | config | online score |
|---|---|---:|
| champion | `configs/ensemble_champion_segmented6_prior.yaml` | 5482 |
| holiday_only | `configs/ensemble_champion_segmented6_prior_holiday_only.yaml` | 5370 |

Decision:

- `champion` is the current online best.
- `holiday_only` is rejected for now: it beat champion on Jan-Feb-like local validation but lost online by `112`.
- This weakens confidence in using Jan-Feb-like validation alone for model promotion.
- Next promotions should require broad-validation stability plus a diagnostic reason, not just winter-fold lift.

## Latest Update: Bid Space Diagnostics

Completed on 2026-05-21.

Artifacts:

- `scripts/analyze_bid_space_diagnostics.py`
- `reports/bid_space_diagnostics.md`
- `reports/bid_space_point_stats.csv`
- `reports/bid_space_price_bins.csv`
- `reports/bid_space_window_stats.csv`
- `reports/bid_space_daily_dispatch.csv`
- `reports/bid_space_daily_summary.csv`
- `reports/bid_space_vs_price_bins.png`
- `reports/bid_space_error_by_hour.png`
- `reports/bid_space_window_rank_vs_profit.png`

Setup:

- Uses 2025 labelled training data only.
- `bid_space_fct = 系统负荷预测值 - 风光总加预测值 - 联络线预测值 - 水电预测值 - 非市场化机组预测值`.
- `bid_space_act = 系统负荷实际值 - 风光总加实际值 - 联络线实际值 - 水电实际值 - 非市场化机组实际值`.
- Window diagnostics use 8-slot / 2-hour rolling means, matching the storage block constraint.

Main findings:

| metric | forecast bid space | actual bid space | interpretation |
|---|---:|---:|---|
| point Spearman vs price | `0.7606` | `0.8360` | actual bid space is a stronger price signal |
| window Spearman vs price | `0.6478` | `0.7404` | forecast error hurts window ranking |
| direct pair profit mean | `7684.70` | `8653.16` | actual bid space improves dispatch but is still below oracle |
| window oracle profit mean | `12619.53` | `12619.53` | price-window oracle remains far higher |
| loss days from pair | `35` | `23` | actual bid space reduces bad days but does not solve them |

Decision:

- Keep `bid_space` as a diagnostic / auxiliary business feature, not as a direct dispatch replacement.
- The gap from forecast to actual bid space confirms boundary forecast error is material.
- The remaining gap from actual bid space to price-window oracle means bid space alone misses price drivers; the price model remains necessary.
- Next useful direction is targeted feature/diagnostic work around months with weak window ranking, especially May-August and October, rather than promoting bid-space-only models.

## Current Champion

- Run ID: `ens_champion_segmented6_prior`
- Alias: `ens_champion_segmented6_prior`
- Config: `configs/ensemble_champion_segmented6_prior.yaml`
- Candidate submission: `outputs/output_ens_champion_segmented6_prior.csv`
- Promote to `outputs/output.csv` once submission validation passes (see Promotion below).

The champion is the same weighted ensemble as before, with dispatch slot
prior reranking added:

- `0.25 * lgb_baseline`
- `0.25 * lgb_baseline_last_180d`
- `0.5 * lgb_segmented_6_last_180d`
- dispatch prior: `lambda_charge=0.18`, `lambda_discharge=0.40`, `alpha=0.5`,
  estimated from oracle slot frequencies on training labels strictly before
  the fold boundary.

## Champion Backtest Metrics

Source: `reports/backtest_ens_champion_segmented6_prior.csv`

- Mean validation profit: `8064.630364`
- Worst fold profit: `5928.223332`
- Mean oracle ratio: `0.706768`
- Minimum oracle ratio: `0.567383`
- Total validation loss days: `5`
- Mean curve z-RMSE: identical to baseline (price predictions unchanged)
- Mean regret: `3541.984428`

Lift versus prior-disabled baseline
(`reports/backtest_ens_champion_segmented6.csv`, mean profit
`7917.209643`):

| fold | base profit | prior profit | delta |
|---|---:|---:|---:|
| `valid_2025_09` | `10898.13` | `11310.43` | `+412.30` |
| `valid_2025_10` |  `7961.41` |  `7899.10` |  `-62.31` |
| `valid_2025_11` |  `6941.90` |  `7120.77` | `+178.87` |
| `valid_2025_12` |  `5867.40` |  `5928.22` |  `+60.83` |
| **mean** | **`7917.21`** | **`8064.63`** | **`+147.42`** |

Loss days unchanged at 5 (one redistributed across folds).

## Submission Validation

Validated files:

- `outputs/output_ens_champion_segmented6.csv` (previous champion)
- `outputs/output_ens_champion_segmented6_prior.csv` (new champion, prior-enabled)
- `outputs/output.csv`

Validation result:

- shape: `5664 x 3`
- columns: `times, 实时价格, power`
- rows: `59 days * 96 slots`
- allowed power values only: `-1000, 0, 1000`
- all 59 days trade exactly one charge block and one discharge block
- charge/discharge blocks are consecutive 8-slot blocks
- discharge starts after charge

## Quality Gates

Remote AutoDL environment (2026-05-17 run):

- `make lint`: passed
- `make test`: passed, `40 passed` (includes 8 new dispatch-prior tests)

Local machine note:

- local `python3` exists, but `pytest` and `ruff` are not installed in that interpreter.
- formal gates should continue to run on AutoDL env unless we intentionally build a local Python env.

## Experiments Completed

Main completed comparisons:

- original baseline: mean profit `7454.028`, oracle ratio `0.661`, loss days `9`.
- `business` features: mean profit `7617.559`, improves mean but month stability is mixed.
- `baseline_last_180d`: mean profit `7667.670`, best single model so far.
- 90-day windows: worse than 180-day and all-past variants.
- centered/zscore/rank/deviation targets: did not beat absolute target.
- tau no-trade threshold: did not help; current dominant error is wrong charge/discharge window selection, not merely low-spread days.
- `0.25 baseline + 0.25 baseline_last_180d + 0.5 segmented_6` is the current champion.

## Latest Experiment: Dispatch Slot Prior

Completed on 2026-05-17.

Source:

- `reports/backtest_ens_champion_segmented6_prior.csv`
- `reports/backtest_ens_champion_segmented6_prior_daily.csv`
- `reports/dispatch_prior_grid.csv`
- `reports/dispatch_prior_best_daily.csv`
- code: `src/electricity/dispatch/priors.py`,
  `src/electricity/dispatch/optimizer.py`,
  `src/electricity/eval/backtest.py`,
  `src/electricity/submit.py`

Setup:

- price predictions are produced by the existing champion ensemble
  (`ens_champion_segmented6`, three LightGBM members with weights
  `0.25 / 0.25 / 0.5`); no model retraining.
- dispatch optimizer reranks `(charge_start, discharge_start)` candidates
  with `score = predicted_spread + lambda_charge * log P(t_c | history)
  + lambda_discharge * log P(t_d | history)`, where `P(t_c)` and `P(t_d)`
  are oracle slot frequencies estimated on training labels strictly
  before each fold boundary, with Laplace smoothing `alpha=0.5`.
- best lambdas chosen by grid search on cached predictions:
  `lambda_charge=0.18`, `lambda_discharge=0.40`.
- prior is disabled by default; existing configs without
  `dispatch.prior.enabled: true` reproduce the previous results
  bit-for-bit (verified on AutoDL: baseline run reproduces
  `7917.21` exactly).

Result:

| fold | base profit | prior profit | delta |
|---|---:|---:|---:|
| `valid_2025_09` | `10898.13` | `11310.43` | `+412.30` |
| `valid_2025_10` |  `7961.41` |  `7899.10` |  `-62.31` |
| `valid_2025_11` |  `6941.90` |  `7120.77` | `+178.87` |
| `valid_2025_12` |  `5867.40` |  `5928.22` |  `+60.83` |
| mean | `7917.21` | `8064.63` | `+147.42` |
| worst fold | `5867.40` | `5928.22` | `+60.83` |
| mean oracle ratio | `0.694` | `0.707` | `+0.013` |
| total loss days | `5` | `5` | `0` |

Promotion gate (`TEAM_WORKFLOW.md`):

- mean profit: `+147.42` (passes).
- loss days: unchanged at `5` (passes).
- fold stability: 3/4 folds positive; the only regression is
  `valid_2025_10` at `-62.31`, which is small relative to that fold's
  profit (`< 0.8%`) and is offset by `valid_2025_10`'s loss-day
  improvement (`4 -> 3`).
- mean oracle ratio: `0.694 -> 0.707` (closer to oracle).
- explainability: the prior corrects the optimizer's tendency to pick
  systematically biased slots; verified by the per-fold lift pattern.
- submit generation: yes, `outputs/output_ens_champion_segmented6_prior.csv`
  is generated and validates against the submit schema.

Decision:

- promote `ens_champion_segmented6_prior` to current champion.
- regenerate `outputs/output.csv` from
  `outputs/output_ens_champion_segmented6_prior.csv` after a final
  spot check on the submission file.

## Latest Experiment: Selective Business On Segments

Completed on 2026-05-05.

Source:

- `reports/backtest_lgb_segmented_6_selective_business_last_180d.csv`
- `reports/backtest_lgb_segmented_6_selective_business_last_180d_daily.csv`
- `reports/backtest_lgb_segmented_6_selective_business_last_180d_segments.csv`

Setup:

- base model: `lgb_segmented_6_last_180d`
- only two segments received `business` features:
  - `40_56`
  - `72_88`

Result:

- mean profit: `7682.243`
- worst fold profit: `5847.864`
- mean oracle ratio: `0.662604`
- loss days: `6`
- mean curve z-RMSE: `0.879702`
- mean regret: `3924.372`

Comparison:

- worse than `lgb_segmented_6_last_180d` on mean profit, worst fold, oracle ratio, curve z-RMSE, and regret.
- also worse than current champion `ens_champion_segmented6`.
- only small positive signal is loss days `7 -> 6`, which is not enough to justify promotion.

Decision:

- reject `lgb_segmented_6_selective_business_last_180d` as a promotion candidate.
- keep current champion and default submission unchanged.

## Latest Experiment: Margin-Focused Business Features

Completed on 2026-05-05.

Source:

- `reports/backtest_lgb_margin_net_load_last_180d.csv`
- `reports/backtest_lgb_margin_core_last_180d.csv`
- `reports/backtest_lgb_segmented_6_margin_core_last_180d.csv`

Setup:

- goal: translate business understanding into a narrower feature family centered on marginal-price drivers.
- tested three variants:
  - `lgb_margin_net_load_last_180d`: only `net_load`
  - `lgb_margin_core_last_180d`: `net_load`, `renewable_ratio`, `non_market_ratio`, `tie_line_ratio`
  - `lgb_segmented_6_margin_core_last_180d`: segmented model with time-of-day specific subsets of those margin-oriented features

Result summary:

| model | mean profit | worst fold | oracle ratio mean | loss days | curve z-RMSE | avg regret |
|---|---:|---:|---:|---:|---:|---:|
| `lgb_margin_net_load_last_180d` | `7726.193` | `5975.994` | `0.678443` | `7` | `0.863716` | `3880.422` |
| `lgb_margin_core_last_180d` | `7523.491` | `5995.881` | `0.662508` | `9` | `0.862076` | `4083.123` |
| `lgb_segmented_6_margin_core_last_180d` | `7790.025` | `6251.791` | `0.684462` | `5` | `0.848046` | `3816.589` |
| current champion `ens_champion_segmented6` | `7917.210` | `5867.398` | `0.694125` | `5` | `0.850138` | `3689.405` |

Interpretation:

- `net_load` alone is informative, but not sufficient to beat the stronger baselines or champion.
- the non-segmented `margin_core` compression loses too much information and is clearly rejected.
- `segmented_6_margin_core` is the only promising result in this batch:
  - it is still below `lgb_segmented_6_last_180d` on mean profit.
  - it materially improves worst fold versus `lgb_segmented_6_last_180d`: `5891.334 -> 6251.791`.
  - it reduces loss days from `7 -> 5`.
  - it improves curve z-RMSE from `0.868496 -> 0.848046`.
  - but it still does not beat the champion on mean profit or regret.

Decision:

- reject `lgb_margin_net_load_last_180d` and `lgb_margin_core_last_180d`.
- keep `lgb_segmented_6_margin_core_last_180d` as a stability candidate, not a promotion candidate.
- keep current champion and default submission unchanged.

## Latest Experiment: Strategy Gate Diagnostics

Completed on 2026-05-06. Updated later the same day after adding dispatch confidence features.

Source:

- `reports/strategy_gate_diagnostics.md`
- `reports/strategy_gate_overall.csv`
- `reports/strategy_gate_crossfold_summary.csv`
- `reports/strategy_gate_crossfold_daily.csv`
- `reports/strategy_gate_dataset.csv`

Setup:

- evaluated a day-level selector over three existing experts:
  - `ens_champion_segmented6`
  - `lgb_segmented_6_last_180d`
  - `lgb_segmented_6_margin_core_last_180d`
- validation method: held-out month cross-fold.
- tested:
  - best static expert from training months
  - single-feature threshold rule
  - logistic-regression gate
  - oracle selector upper bound

Result:

| method | mean profit | lift vs champion | loss days | decision |
|---|---:|---:|---:|---|
| `champion` | `7850.814` | `0.000` | `5` | baseline |
| `static` | `7850.814` | `0.000` | `5` | same as champion |
| `rule` | `7860.005` | `9.191` | `5` | tiny positive diagnostic only |
| `logistic` | `7836.137` | `-14.677` | `6` | rejected |
| `oracle_selector` | `8319.345` | `468.531` | `5` | upper bound only |

Interpretation:

- there is real daily complementarity among the three experts: oracle selector lift is about `+468.5`.
- the learnable held-out signal is weak so far: the best simple rule improves mean profit by only about `+9.2`.
- added dispatch confidence features to daily reports:
  - `top2_spread`
  - `top5_spread_mean`
  - `top5_spread_std`
  - `top1_top2_gap`
  - `top1_top5_mean_gap`
  - `top_candidate_count`
- the confidence features improved logistic gate from `-58.5` to `-14.7` vs champion, but still did not beat champion.

Decision:

- do not promote a strategy gate or change `outputs/output.csv`.
- keep this result as evidence that expert complementarity exists, but gate features/model are not yet reliable enough for submission.

## Latest Experiment: No-Trade Gate Diagnostics

Completed on 2026-05-07.

Source:

- `reports/no_trade_gate_diagnostics.md`
- `reports/no_trade_gate_overall.csv`
- `reports/no_trade_gate_crossfold_summary.csv`
- `reports/no_trade_gate_crossfold_daily.csv`

Setup:

- default action: use current champion `ens_champion_segmented6`.
- learned action: set an entire held-out day to no-trade, making that day's profit `0`.
- validation method: held-out month cross-fold.
- searched conservative single-feature and two-feature threshold rules using ex-ante daily features, expert disagreement, and dispatch confidence features.

Result:

| method | mean profit | lift vs champion | worst day profit | loss days | no-trade days |
|---|---:|---:|---:|---:|---:|
| `champion` | `7850.814` | `0.000` | `-15992.706` | `5` | `0` |
| `no_trade_gate` | `7415.746` | `-435.068` | `-15992.706` | `5` | `4` |

Interpretation:

- the no-trade gate did not filter any held-out loss days.
- it filtered four held-out profitable days instead.
- two filtered September days were highly profitable champion days: `31713.851` and `13972.541`.
- this confirms the earlier concern: avoiding bad days is hard because validation loss days are few and high-risk-looking days can also be high-opportunity days.

Decision:

- reject this no-trade gate.
- do not change `outputs/output.csv`.
- if no-trade is revisited, it should use a much stricter rule such as only filtering days where multiple experts agree on low confidence and expected spread is also low.

## Latest Experiment: Top-K Pair Diagnostics

Completed on 2026-05-07.

Source:

- `reports/topk_pair_diagnostics.md`
- `reports/topk_pair_summary.csv`
- `reports/topk_pair_daily.csv`
- `reports/topk_pair_candidates.csv`

Setup:

- for each expert and validation day, rank all `3321` legal charge/discharge pairs by predicted spread.
- keep predicted top-K pairs for `K in {1, 3, 5, 10}`.
- score those candidates using hidden validation prices.
- this is diagnostic only; it uses true prices after candidate generation, so `topk_best` is an upper bound inside predicted candidates.

Key result:

| model | top1 mean profit | top10 best-in-candidates mean profit | top10 lift | positive lift days | oracle in top10 |
|---|---:|---:|---:|---:|---:|
| `ens_champion_segmented6` | `7850.814` | `8578.343` | `727.529` | `101/120` | `5.0%` |
| `lgb_segmented_6_last_180d` | `7739.550` | `8400.974` | `661.424` | `99/120` | `2.5%` |
| `lgb_segmented_6_margin_core_last_180d` | `7725.897` | `8436.005` | `710.107` | `98/120` | `3.3%` |

Interpretation:

- predicted top-K often contains a better pair than predicted top1.
- for the current champion, predicted top10 contains a better-than-top1 pair on `84.2%` of validation days.
- however, the true oracle pair is rarely exactly inside predicted top10, only `5.0%` for the champion.
- this means there is meaningful reranking potential inside predicted candidates, but not enough to solve the full prediction problem by candidate search alone.

Decision:

- keep 96-point prediction as mainline.
- next promising diagnostic is a conservative top-K reranker that chooses among predicted top10 pairs using only ex-ante features, not hidden prices.

## Reproduction Commands

Run on AutoDL under `/root/autodl-tmp/electricity`:

```bash
make lint
make test
make ensemble CONFIG=configs/ensemble_champion_and_segmented_6.yaml
make submit CONFIG=configs/ensemble_champion_and_segmented_6.yaml
```

Generic commands now support a config override:

```bash
make backtest CONFIG=configs/lgb_baseline_last_180d.yaml
make ensemble CONFIG=configs/ensemble_champion_and_segmented_6.yaml
make submit CONFIG=configs/ensemble_champion_and_segmented_6.yaml
```

## Next Experiment Boundary

Do not run 0.6/0.4 or 0.7/0.3 ensemble weight search unless explicitly re-approved.

Most useful next directions:

- compare another tree model family, especially CatBoost or XGBoost, under the same fold and dispatch harness.
- add small, controlled NWP summary features from `all_nc`, starting with hourly/daily aggregate weather variables rather than large neural models.
- add diagnostics for month/day types where `business` helps versus hurts, then decide whether a gated ensemble is justified.
- keep GPU off for now; current experiments are CPU-bound tabular training.

## Model Family Comparison: XGBoost and CatBoost

Completed on 2026-05-04.

Source: `reports/model_family_comparison.csv`

| model | mean profit | worst fold | oracle ratio mean | loss days | decision |
|---|---:|---:|---:|---:|---|
| `ens_baseline_baseline_last_180d` | `7758.127` | `5847.162` | `0.684640` | `10` | current champion |
| `lgb_baseline_last_180d` | `7667.670` | `5859.852` | `0.675228` | `7` | strongest single model |
| `cat_baseline_last_180d` | `7556.908` | `5729.337` | `0.671176` | `7` | candidate only, not promoted |
| `cat_baseline` | `7509.300` | `6026.640` | `0.668496` | `8` | candidate only, not promoted |
| `lgb_baseline` | `7454.028` | `5861.933` | `0.661241` | `9` | original baseline |
| `xgb_baseline_last_180d` | `7426.235` | `6045.302` | `0.656938` | `8` | rejected |
| `xgb_baseline` | `7400.550` | `5909.587` | `0.656997` | `9` | rejected |

Conclusion:

- CatBoost is the best non-LightGBM model family so far, but it does not beat LightGBM 180-day or the current champion.
- XGBoost does not show a promotion signal under the same features, folds, and dispatch harness.
- No model-family candidate replaces `ens_baseline_baseline_last_180d`.
- Do not continue XGBoost tuning unless we later introduce new features that materially change the feature distribution.
- CatBoost may be reconsidered only as a diversity member after we inspect daily complementarity, not as a direct replacement.

Dependency provenance note:

- `xgboost` was already installed in the AutoDL environment: version `3.2.0`.
- `catboost` was missing and was installed as version `1.2.10`.
- Attempted official PyPI install with `--index-url https://pypi.org/simple`, but the download timed out and stalled. The usable installation came from the AutoDL environment's configured pip mirror. This should be treated as an environment provenance exception, not a competition data/model-weight source.

## Daily Diagnostics and CatBoost Complementarity

Completed on 2026-05-04.

Generated files:

- `reports/daily_diagnostics.md`
- `reports/daily_model_summary.csv`
- `reports/daily_model_complementarity.csv`
- `reports/daily_model_comparison.csv`

Key diagnostic findings:

- The champion has 10 validation loss days under daily aggregation.
- High-regret days are dominated by large charge/discharge start gaps, not by invalid dispatch.
- CatBoost 180d beats the current champion on 58 of 120 validation days, but loses on 40 days, with 22 ties.
- A perfect daily selector between champion and CatBoost 180d would lift daily mean profit by about `391`, but this is an oracle selector and is not directly usable in test.

Fixed equal-weight CatBoost ensemble checks:

| model | mean profit | worst fold | oracle ratio mean | loss days | decision |
|---|---:|---:|---:|---:|---|
| `ens_baseline_baseline_last_180d` | `7758.127` | `5847.162` | `0.684640` | `10` | current champion |
| `ens_base_base180_cat180` | `7668.601` | `5887.218` | `0.678945` | `6` | not promoted; lower mean, more stable |
| `ens_lgb180_cat180` | `7657.853` | `5927.563` | `0.679343` | `9` | not promoted |

Conclusion:

- Simple equal-weight CatBoost ensembling does not convert daily complementarity into higher average validation profit.
- The three-member ensemble is useful as a robustness reference because it reduces loss days from 10 to 6, but it gives up about 90 mean profit versus champion.
- The current champion remains `ens_baseline_baseline_last_180d`.
- Next highest-value direction remains adding controlled NWP/weather summary features rather than further CatBoost/XGBoost tuning.

## NWP Grid Feature Attempt 1

Completed on 2026-05-04.

Generated artifacts:

- `artifacts/features/nwp_grid_15min.csv`
- `artifacts/features/nwp_grid_15min_manifest.md`
- `configs/lgb_nwp_last_180d.yaml`
- `reports/backtest_lgb_nwp_last_180d.csv`
- `reports/backtest_lgb_nwp_last_180d_daily.csv`

Feature construction:

- Source: competition-provided `all_nc/*.nc` files.
- Alignment: file date `D` is issue date; weather features align to Beijing target day `D+1`.
- Aggregation: full grid mean/std/min/max per hour for `ghi, sp, t2m, tcc, tp, u100, v100`.
- Derived feature: `wind_speed100 = sqrt(u100^2 + v100^2)` with grid mean/std/min/max.
- Hourly values are repeated to the four 15-minute rows in each hour.
- Output shape: `40704 x 33`; 32 NWP features plus `times`.
- Test set NWP coverage: complete.
- Train set missing NWP only for 2025-01-01 because no 2024-12-31 issue file exists.

Backtest result:

| model | mean profit | worst fold | oracle ratio mean | loss days | curve z-RMSE | decision |
|---|---:|---:|---:|---:|---:|---|
| `ens_baseline_baseline_last_180d` | `7758.127` | `5847.162` | `0.684640` | `10` | `0.862050` | current champion |
| `lgb_baseline_last_180d` | `7667.670` | `5859.852` | `0.675228` | `7` | `0.870785` | strongest single model |
| `lgb_nwp_last_180d` | `7044.125` | `5615.329` | `0.628540` | `11` | `0.896809` | rejected |

Conclusion:

- Naively adding full-grid NWP summary features materially worsened validation profit.
- This does not mean NWP is useless; it means this first coarse full-grid aggregation is too noisy or not targeted enough.
- Do not ensemble or submit this NWP model.
- If continuing NWP work, use more selective/weather-aware features: smaller region, fewer variables, daily/hourly deltas, or interactions with official wind/solar forecasts.

## Targeted NWP Attempts

Completed on 2026-05-04.

Compared variants:

- `lgb_nwp_core_last_180d`: only 9 weather columns directly tied to wind/solar: `ghi_mean/std/max`, `tcc_mean/std/max`, `wind_speed100_mean/std/max`.
- `lgb_nwp_core_inter_last_180d`: the same 9 weather columns plus 5 interactions with official wind/solar forecasts.

Backtest result:

| model | mean profit | worst fold | oracle ratio mean | loss days | curve z-RMSE | decision |
|---|---:|---:|---:|---:|---:|---|
| `ens_baseline_baseline_last_180d` | `7758.127` | `5847.162` | `0.684640` | `10` | `0.862050` | current champion |
| `lgb_baseline_last_180d` | `7667.670` | `5859.852` | `0.675228` | `7` | `0.870785` | strongest single model |
| `lgb_nwp_core_last_180d` | `7371.067` | `5698.410` | `0.653018` | `10` | `0.870472` | rejected |
| `lgb_nwp_core_inter_last_180d` | `7001.667` | `5827.860` | `0.625128` | `7` | `0.872807` | rejected |
| `lgb_nwp_last_180d` | `7044.125` | `5615.329` | `0.628540` | `11` | `0.896809` | rejected |

Conclusion:

- Targeted NWP core features improve over the naive full-grid NWP attempt but still underperform the no-NWP LightGBM 180-day model by about 297 mean profit.
- NWP interactions with official wind/solar forecasts made performance worse, especially in October.
- Current evidence says NWP should not be added to the main champion pipeline yet.
- If NWP is revisited, the next attempt should not be more columns; it should be better spatial selection or weather-error features, not coarse full-grid aggregation.

## Window Mean Model Attempt 1

Completed on 2026-05-04.

Implemented files:

- `src/electricity/features/window_features.py`
- `src/electricity/eval/window_backtest.py`
- `configs/lgb_window_mean_last_180d.yaml`
- `reports/backtest_lgb_window_mean_last_180d.csv`
- `reports/backtest_lgb_window_mean_last_180d_daily.csv`

Model design:

- Convert each complete day into 89 candidate 8-slot windows.
- Target: true mean price of each 8-slot window.
- Features: mean/std/min/max of official base features inside each window plus window start/end/center time features.
- Dispatch: predict all 89 window means, then enumerate feasible charge/discharge window pairs.

Backtest result:

| model | mean profit | worst fold | oracle ratio mean | loss days | charge gap | discharge gap | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `ens_baseline_baseline_last_180d` | `7758.127` | `5847.162` | `0.684640` | `10` | `12.664` | `10.112` | current champion |
| `lgb_baseline_last_180d` | `7667.670` | `5859.852` | `0.675228` | `7` | `12.611` | `10.393` | strongest single model |
| `lgb_window_mean_last_180d` | `7238.530` | `5708.910` | `0.639640` | `10` | `14.227` | `10.755` | rejected |

Conclusion:

- First window-mean regression model underperforms the point-price LGB180 baseline.
- It worsens charge-window timing, which is the current main failure mode.
- If continuing the window route, do not just refine single-window mean regression; try pairwise window spread/ranking features that directly compare charge and discharge windows.

## Pair Window Spread Attempt 1

Completed on 2026-05-04.

Implemented files:

- `src/electricity/features/pair_window_features.py`
- `src/electricity/eval/pair_window_backtest.py`
- `configs/lgb_pair_spread_last_180d.yaml`
- `tests/test_pair_window_features.py`
- `reports/backtest_lgb_pair_spread_last_180d.csv`
- `reports/backtest_lgb_pair_spread_last_180d_daily.csv`

Model design:

- Convert each complete day into all legal charge/discharge 8-slot window pairs.
- Legal pair count per day: `3321`.
- Target: `sum(true_price[discharge_window]) - sum(true_price[charge_window])`.
- First full feature version used mean/std/min/max window aggregates and was killed by memory during LightGBM training on the no-card AutoDL instance.
- Final runnable version uses only window mean aggregates plus pair time/gap features to reduce memory.
- Dispatch: choose the legal pair with highest predicted spread, `tau=0`.

Backtest result:

| model | mean profit | worst fold | oracle ratio mean | loss days | charge gap | discharge gap | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `ens_baseline_baseline_last_180d` | `7758.127` | `5847.162` | `0.684640` | `10` | `12.664` | `10.112` | current champion |
| `lgb_baseline_last_180d` | `7667.670` | `5859.852` | `0.675228` | `7` | `12.611` | `10.393` | strongest single model |
| `lgb_window_mean_last_180d` | `7238.530` | `5708.910` | `0.639640` | `10` | `14.227` | `10.755` | rejected |
| `lgb_pair_spread_last_180d` | `6912.828` | `5652.976` | `0.619168` | `9` | `14.352` | `10.444` | rejected |

Conclusion:

- Direct pair-spread regression did not improve the storage decision despite matching the dispatch objective more directly.
- It especially underperformed in October and December; charge-window timing remained worse than point-price LGB180.
- Do not promote this model and do not replace `outputs/output.csv`.
- The window route should pause unless we redesign it as a ranking/classification problem or use pair predictions only as a secondary ensemble signal.

## Gated Diagnostics Attempt 1

Completed on 2026-05-04.

Implemented file:

- `scripts/analyze_gated_diagnostics.py`

Generated reports:

- `reports/gated_daily_dataset.csv`
- `reports/gated_month_summary.csv`
- `reports/gated_feature_bins.csv`
- `reports/gated_crossfold_rules.csv`
- `reports/gated_crossfold_daily.csv`
- `reports/gated_diagnostics.md`

Goal:

- Diagnose whether same-day visible features can predict which existing point model should be trusted.
- Candidate models: current champion, `lgb_baseline_last_180d`, `lgb_business_last_180d`, `cat_baseline_last_180d`, and `ens_base_base180_cat180`.
- Rules are deliberately simple one-feature threshold gates to reduce overfitting risk.
- Hidden true prices are used only for validation, not for rule inputs.

Cross-fold result:

| held-out fold | learned gate feature | train lift vs champion | valid lift vs champion |
|---|---|---:|---:|
| `valid_2025_09` | `renewable_ratio_mean` | `+276.535` | `-148.171` |
| `valid_2025_10` | `champion_predicted_spread` | `+182.612` | `-132.564` |
| `valid_2025_11` | `net_load_mean` | `+287.200` | `-243.480` |
| `valid_2025_12` | `wind_ratio_mean` | `+261.444` | `-6.157` |

Aggregate held-out result:

- Champion mean: `7758.127428`
- Gated selector mean: `7625.534312`
- Gated lift vs champion: `-132.593116`
- Best-static-model-from-train mean: `7528.596633`

Month diagnostic:

- September: champion is best.
- October: `lgb_business_last_180d` is best by `+270.093` vs champion.
- November: `cat_baseline_last_180d` is best by `+104.874` vs champion.
- December: `ens_business_business_last_180d` is best by `+54.796` vs champion.

Decision:

- Do not promote a gated selector yet.
- The apparent month/day-type signal is not stable enough under leave-one-month-out validation.
- Feature bucket results are useful diagnostics, but should not be used directly as submit-time rules.
- Current champion remains `ens_baseline_baseline_last_180d`.

## Segmented Point Model Attempt 1

Completed on 2026-05-04.

Implemented files:

- `src/electricity/eval/segmented_backtest.py`
- `configs/lgb_segmented_6_last_180d.yaml`
- `tests/test_segmented_backtest.py`

Model design:

- Split one day into six slot segments: `[0,24)`, `[24,40)`, `[40,56)`, `[56,72)`, `[72,88)`, `[88,96)`.
- Train one LightGBM per segment inside each fold.
- Use the same base official features plus time features as `lgb_baseline_last_180d`.
- Use the same 180-day rolling training window.
- Concatenate segment predictions back into 96-point daily curves, then use the standard dispatch optimizer.

Backtest result:

| model | mean profit | worst fold | oracle ratio mean | loss days | curve z-RMSE | charge gap | discharge gap | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ens_baseline_baseline_last_180d` | `7758.127` | `5847.162` | `0.684640` | `10` | `0.862050` | `12.664` | `10.112` | current champion |
| `lgb_baseline_last_180d` | `7667.670` | `5859.852` | `0.675228` | `7` | `0.870785` | `12.611` | `10.393` | strongest previous single model |
| `lgb_segmented_6_last_180d` | `7806.961` | `5891.334` | `0.685423` | `7` | `0.868496` | `13.674` | `10.402` | candidate |

Decision:

- This is the first non-ensemble point-model candidate that beats the current champion on mean profit, worst fold, and loss days.
- Do not replace `outputs/output.csv` yet because submit-time segmented inference is not implemented and charge-window gap is worse.
- Next step: implement segmented submit or test an ensemble between champion and segmented model under the same backtest harness.

## Segmented Submit and Champion+Segmented Ensemble

Completed on 2026-05-05.

Implemented updates:

- `src/electricity/submit.py` now supports segmented full-model inference.
- `src/electricity/eval/backtest.py` now supports segmented members inside the ensemble harness.
- `src/electricity/eval/segmented_utils.py` contains shared segmented helper logic.
- `configs/ensemble_champion_and_segmented_6.yaml`

Generated artifact:

- `outputs/output_lgb_segmented_6_last_180d.csv`

Segmented submit validation:

- shape: `5664 x 3`
- columns: `times, 实时价格, power`
- allowed power values only: `-1000, 0, 1000`
- trade days: `59`

Champion plus segmented ensemble result:

| model | mean profit | worst fold | oracle ratio mean | loss days | curve z-RMSE | charge gap | discharge gap | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ens_baseline_baseline_last_180d` | `7758.127` | `5847.162` | `0.684640` | `10` | `0.862050` | `12.664` | `10.112` | previous champion |
| `lgb_segmented_6_last_180d` | `7806.961` | `5891.334` | `0.685423` | `7` | `0.868496` | `13.674` | `10.402` | strong single-model candidate |
| `ens_champion_segmented6` | `7917.210` | `5867.398` | `0.694125` | `5` | `0.850138` | `12.857` | `9.807` | strongest backtest so far |

Interpretation:

- This ensemble is currently the best validation result in the project.
- It improves mean profit by about `+159.082` over the previous champion.
- It reduces loss days from `10` to `5`.
- It improves discharge gap, oracle ratio, and curve z-RMSE.
- Worst fold is slightly below the segmented single model but still above the previous champion.

Current decision:

- `ens_champion_segmented6` is the strongest backtest candidate and should be treated as the new promotion target.
- `outputs/output.csv` has not been replaced yet because the ensemble submit file has not been generated in this step.
- Immediate next step is to generate and validate the ensemble submit file, then promote it if no submission-format issue appears.

## Segmented Business Feature Attempt 1

Completed on 2026-05-05.

Config:

- `configs/lgb_segmented_6_business_last_180d.yaml`

Model design:

- Start from `lgb_segmented_6_last_180d`.
- Add the `business` derived feature family globally to every segment.
- Keep the same six time segments and 180-day rolling window.

Backtest result:

| model | mean profit | worst fold | oracle ratio mean | loss days | curve z-RMSE | decision |
|---|---:|---:|---:|---:|---:|---|
| `lgb_segmented_6_last_180d` | `7806.961` | `5891.334` | `0.685423` | `7` | `0.868496` | strong single-model candidate |
| `lgb_segmented_6_business_last_180d` | `7508.059` | `5665.289` | `0.649100` | `4` | `0.851251` | rejected |
| `ens_champion_segmented6` | `7917.210` | `5867.398` | `0.694125` | `5` | `0.850138` | current champion |

Interpretation:

- Adding `business` features directly to all six segments reduces loss days and slightly improves curve-shape metrics, but it materially harms average profit.
- The degradation is dominated by a large November drop, so this is not a stable improvement.
- The correct next inference is not “business features are useless”, but “business features should not be globally added to every segmented expert”.
- If revisiting this direction, the next step should be segment-specific feature selection rather than whole-model feature expansion.

## Top-K Pair Reranker Diagnostic

Completed on 2026-05-07.

Artifacts:

- `reports/topk_reranker_diagnostics.md`
- `reports/topk_reranker_overall.csv`
- `reports/topk_reranker_crossfold_summary.csv`
- `reports/topk_reranker_selected_daily.csv`
- `reports/topk_reranker_dataset.csv`

Setup:

- Base model: `ens_champion_segmented6`.
- Candidate set: the predicted top 10 legal charge/discharge window pairs per day.
- Reranker: held-in-month `GradientBoostingRegressor` predicting candidate true profit, evaluated by held-out month.
- Purpose: test whether the large top-10 oracle gap can be captured by an ex-ante candidate selector.

Held-out result:

| method | mean profit | worst fold | loss days | selected rank mean | rank-1 days | lift vs top1 |
|---|---:|---:|---:|---:|---:|---:|
| predicted top1 | `7850.814` | `5867.398` | `5` | `1.000` | `120` | `0.000` |
| top-K reranker | `7858.495` | `5867.398` | `5` | `1.067` | `118` | `+7.681` |
| oracle within top10 | `8578.343` | `6364.485` | `3` | `5.767` | `19` | `+727.529` |

Interpretation:

- The top-10 candidate set contains substantial unrealized upside, but the current learned reranker captures almost none of it.
- The reranker changed only 2 out of 120 validation days, selecting rank 2 once and rank 8 once.
- The result is directionally positive but too small and too sparse to promote.
- Do not replace `outputs/output.csv` or change the submit strategy based on this diagnostic.

Recommended next step:

- Try a stronger but still conservative candidate-selection diagnostic, preferably using cross-expert top-K consensus or pairwise preference learning instead of direct profit regression on champion-only top10 candidates.

## Top-K Cross-Expert Consensus

Completed on 2026-05-07.

Artifacts:

- `scripts/analyze_topk_consensus.py`
- `scripts/generate_topk_consensus_submit.py`
- `reports/topk_consensus_diagnostics.md`
- `reports/topk_consensus_summary.csv`
- `reports/topk_consensus_by_fold.csv`
- `reports/topk_consensus_daily.csv`
- `reports/topk_consensus_candidates.csv`
- `outputs/output_topk_consensus_unanimous_then_rank.csv`

Setup:

- Experts:
  - `ens_champion_segmented6`
  - `lgb_segmented_6_last_180d`
  - `lgb_segmented_6_margin_core_last_180d`
- Candidate set: each expert's predicted top 10 legal charge/discharge window pairs.
- Best rule: `unanimous_then_rank`.
- Rule behavior: if a pair appears in all three experts' top 10, choose the unanimous pair with the best mean candidate rank; otherwise fall back to champion top1.

Validation result:

| rule | mean profit | worst day profit | loss days | mean lift vs champion top1 | changed days |
|---|---:|---:|---:|---:|---:|
| `champion_top1` | `7850.814` | `-15992.706` | `5` | `0.000` | `0` |
| `unanimous_then_rank` | `7924.356` | `-14029.806` | `4` | `+73.542` | `53` |

Fold lift for `unanimous_then_rank`:

| fold | mean lift vs champion top1 |
|---|---:|
| `valid_2025_09` | `+56.468` |
| `valid_2025_10` | `+3.625` |
| `valid_2025_11` | `+40.195` |
| `valid_2025_12` | `+191.150` |

Submit candidate:

- Generated: `outputs/output_topk_consensus_unanimous_then_rank.csv`
- Shape: `5664 x 3`
- Trade days: `59`
- Power counts: `472` charge slots, `472` discharge slots, `4720` zero slots.
- Difference vs current `outputs/output.csv`: `20` days and `100` slots differ.

Decision:

- This is the first candidate-selection rule with positive lift in every validation fold.
- It is a credible submit candidate, but should not automatically replace `outputs/output.csv` yet.
- Next step before promotion: compare this candidate against the current default submit assumptions and decide whether to submit both if the platform allows multiple daily attempts.
