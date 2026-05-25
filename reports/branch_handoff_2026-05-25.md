# Branch Handoff: `codex/pair-e2e-bidspace-weather`

Date: 2026-05-25
Branch head: `88c3a25 Sync experiment records and sequence baselines`

## Purpose

This branch consolidates the recent experiment records, online submissions, sequence/deep baselines, and online/local validation gap analysis for the electricity storage arbitrage competition.

Use this document to catch up another Codex thread before continuing experiments.

## Current Best

Current online champion remains:

- Model: `ens_champion_segmented6_prior`
- Config: `configs/ensemble_champion_segmented6_prior.yaml`
- Submit file: `outputs/output_ens_champion_segmented6_prior.csv`
- Online score: `5482`
- Dashboard model id: `ens_champion_segmented6_prior_5fold`

Champion local validation:

| Metric | Value |
| --- | ---: |
| Standard 2025-09..12 mean daily profit | 7991.35 |
| Jan-Feb-like mean daily profit | 14153.50 |
| All 5-fold weighted mean daily profit | 9952.03 |
| Simple fold mean daily profit | 9282.40 |
| All 5-fold loss days | 4 in dashboard / 5 by raw fold sum |
| Online score | 5482 |

Raw champion fold report: `reports/backtest_ens_champion_segmented6_prior_5fold.csv`.

## Online Submission Ledger

| Model | Output | Online Score | Conclusion |
| --- | --- | ---: | --- |
| `ens_champion_segmented6_prior` | `outputs/output_ens_champion_segmented6_prior.csv` | 5482 | Current best |
| `ens_champion_segmented6_prior_holiday_only` | `outputs/output_ens_champion_segmented6_prior_holiday_only.csv` | 5370 | Worse despite Jan-Feb-like local gain |
| `sequence_gru_centered_5fold` | `outputs/output_sequence_gru_centered.csv` | 4693 | Standalone GRU fails online |
| `sequence_gru_direct_price_5fold` | `outputs/output_sequence_gru_direct_price.csv` | 4686 | Standalone GRU fails online |
| `sequence_window_gru_5fold` | `outputs/output_sequence_window_gru.csv` | 4477 | Standalone window GRU fails online |

Canonical ledger files:

- `reports/online_submission_scores.csv`
- `reports/online_score_ledger.csv`
- `reports/online_risk_dashboard.csv`
- `reports/online_validation_gap_analysis.md`
- `reports/online_validation_gap_known_scores.csv`

## Main Conclusion From Online/Local Gap

Do not use Jan-Feb-like validation alone as the promotion metric.

Observed online failures:

- `holiday_only` gained Jan-Feb-like locally but lost online by `-112`.
- `GRU direct` gained Jan-Feb-like locally by `+386/day` but lost online by `-796`.
- `GRU centered` was close to champion on Jan-Feb-like but lost online by `-789`.
- `window GRU` had the smallest local all-5fold loss among deep/window models, but was worst online: `4477`, `-1005` vs champion.

Metrics that matched online direction for known submitted non-champion models:

| Metric | Direction Match Rate |
| --- | ---: |
| Standard 2025-09..12 mean daily profit | 100% |
| All 5-fold mean daily profit | 100% |
| Test-like weighted profit | 100% |
| All 5-fold loss days | 100% |
| All 5-fold p90 regret | 100% |

Implication: prioritize broad validation stability, test-like weighted profit, loss days, and p90 regret over Jan-Feb-like uplift.

## Deep / Sequence Experiments

Standalone deep/sequence models were tested as substitutes for champion dispatch inputs. They do not beat champion locally and fail badly online.

| Model | Local Mean Profit | Delta vs Champion | Loss Days | Online Score | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `sequence_gru_direct_price_5fold` | 8652.07 | -630.33/day | 10 | 4686 | Reject |
| `sequence_gru_centered_5fold` | 8834.37 | -448.03/day | 5 | 4693 | Reject |
| `sequence_gru_zscore_5fold` | 8421.38 | -861.02/day | 10 | not submitted | Reject |
| `sequence_tcn_direct_price_5fold` | 8778.87 | -503.53/day | 9 | not submitted | Reject |
| `sequence_tcn_centered_5fold` | 8684.51 | -597.89/day | 9 | not submitted | Reject |
| `sequence_tftlike_direct_price_5fold` | 8563.32 | -719.08/day | 10 | not submitted | Reject |
| `sequence_tftlike_centered_5fold` | 8427.26 | -855.15/day | 9 | not submitted | Reject |
| `sequence_window_gru_5fold` | 9098.30 | -184.11/day | 6 | 4477 | Reject |
| `sequence_window_tcn_5fold` | 8318.80 | -963.61/day | 9 | not submitted | Reject |
| `sequence_window_tftlike_5fold` | 8429.28 | -853.12/day | 10 | not submitted | Reject |

Residual/TFT-like blend note:

- `sequence_tftlike_residual_5fold` with blend weight `0.2` had local `+2.00/day` vs anchor but was later rejected by risk analysis because it increased loss-day risk in gated/residual checks.

Deep model takeaway:

- Stop standalone GRU/TCN/TFT submissions.
- If deep models are used later, use them only as auxiliary risk/rerank signals, not as direct replacements for champion.

## Top-K / Reranker / Bad-Day Direction

We confirmed there is theoretical room inside champion top-K pairs, but learned/rule rerankers were not stable enough.

Important artifacts:

- `reports/current_prior_topk_pair_reranker_review.md`
- `reports/current_prior_topk_pair_reranker_foldsafe_diagnostics.md`
- `reports/current_prior_topk_conservative_switch_diagnostics.md`
- `reports/bad_day_action_not_no_trade_diagnostics.md`
- `reports/weather_bidspace_topk_aux_diagnostics.md`
- `reports/gru_centered_window_gru_top10_aux_diagnostics.md`

Key points:

- Champion top-10 oracle has meaningful upside, but selectors did not reliably capture it.
- Conservative rules sometimes improve average profit modestly but increase loss days or fail fold stability.
- Current best local rerank/rule signals are diagnostic only, not submit-ready.
- Future reranker work must be fold-safe and must pass stricter promotion gates.

## Feature / Business Experiments

Tried axes include:

- `bid_space` / competitive space features.
- Capacity normalization / capacity-only variants.
- Holiday and Spring Festival features.
- Weekly relative / lag-like forecast-input features.
- Forecast-error augmentation.
- Weather / NWP correction and weather as top-K auxiliary signal.
- Feature importance and feature drop diagnostics.
- Shallow LightGBM regularization variants.
- CatBoost/XGBoost family comparisons.

Broad conclusion:

- Many intuitive business features improved some regimes but hurt broad/test-like validation.
- Weather and bid_space are useful for diagnostics and auxiliary risk signals, but global replacement/addition has not beaten champion.
- Holiday-only and winter-like tuning are high overfit risk unless broad validation is preserved.

Important files:

- `reports/champion_feature_importance.md`
- `reports/forecast_error_augmentation_experiment.md`
- `reports/constrained_winter_blend_experiment.md`
- `reports/experiment_map.md`
- `reports/experiment_mindmap.md`
- `reports/experiment_lineage_graph_v2.svg`

## Promotion Rules Going Forward

Do not submit a candidate unless it satisfies most of the following:

- `all_5fold_delta >= -50/day`
- `test_like_delta >= -50/day`
- no extra all-5fold loss days
- no material p90 regret increase
- does not rely only on Jan-Feb-like improvement
- changes few days if it is a reranker/gate
- has fold-safe validation, not in-sample selector validation

Current watch-only candidates from online gap analysis are not strong enough to submit directly.

## Current Git / AutoDL State

This branch was synced through the intended flow:

`local -> GitHub -> AutoDL`

Verified state:

- Local HEAD: `88c3a25`
- GitHub branch `codex/pair-e2e-bidspace-weather`: `88c3a25`
- AutoDL HEAD: `88c3a25`
- AutoDL tests: `32 passed`

AutoDL kept a safety stash before syncing:

- `stash@{0}: pre-sync backup before 88c3a25`

Data policy:

- Competition data is not committed.
- Large prediction caches and search datasets are ignored.
- Compact reports, configs, submit files, and ledgers are committed.

## Recommended Next Steps

1. Continue from champion, not from standalone deep models.
2. Focus on conservative local correction of champion decisions rather than global model replacement.
3. Revisit top-K reranking only with strict fold-safe gates and online-risk filters.
4. Use weather/bid_space/deep outputs as auxiliary features for risk/rerank, not direct price model replacements.
5. Update `reports/online_submission_scores.csv` immediately after every online submission, then rebuild `online_risk_dashboard`.
6. Before submitting any new candidate, compare against `reports/online_validation_gap_candidate_filter.csv` and reject candidates with broad/test-like weakness.

## Quick Pointers For Main Conversation

Start by reading:

1. `reports/branch_handoff_2026-05-25.md`
2. `reports/online_validation_gap_analysis.md`
3. `reports/experiment_map.md`
4. `reports/current_prior_topk_pair_reranker_review.md`
5. `reports/champion_feature_importance.md`

Then inspect current best config:

- `configs/ensemble_champion_segmented6_prior.yaml`
- `configs/ensemble_champion_segmented6_prior_5fold.yaml`
