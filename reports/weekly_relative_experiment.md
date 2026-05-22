# Weekly Relative Feature Experiment

## Purpose

Test a safer version of the shared `672-point weekly lag` idea. Instead of using historical true prices or actual boundary values, this experiment only uses official forecast inputs from prior same-slot days.

## Feature Design

Base columns:

- `系统负荷预测值`
- `风光总加预测值`
- `风电预测值`
- `光伏预测值`
- `bid_space`

Windows:

- 7 same-slot historical days
- 14 same-slot historical days
- 28 same-slot historical days

Variants:

| variant | config | features |
|---|---|---|
| weekly_delta | `configs/ensemble_champion_segmented6_prior_weekly_delta_5fold.yaml` | same-slot delta and rolling same-slot mean deviation |
| weekly_percentile | `configs/ensemble_champion_segmented6_prior_weekly_percentile_5fold.yaml` | same-slot rolling percentile only |
| weekly_delta_percentile | `configs/ensemble_champion_segmented6_prior_weekly_delta_percentile_5fold.yaml` | delta + mean deviation + percentile |

The implementation is test-time safe because it only uses earlier official forecast inputs available in the train+test feature timeline. It does not use historical true prices or actual boundary values.

## Validation-Regime Summary

| model | standard_09_12 | jan_feb_like | winter_11_12_jan_feb | late_winter_12_jan_feb | all_5fold |
|---|---:|---:|---:|---:|---:|
| champion | 7991.351 | 14153.500 | 10170.888 | 11222.654 | 9952.035 |
| weekly_delta | 7390.403 | 14242.104 | 9940.910 | 11227.332 | 9570.490 |
| weekly_percentile | 7423.415 | 13687.295 | 9880.417 | 10876.747 | 9416.468 |
| weekly_delta_percentile | 7491.406 | 14164.877 | 10157.776 | 11389.943 | 9614.784 |

## Fold Profit

| fold | champion | weekly_delta | weekly_percentile | weekly_delta_percentile |
|---|---:|---:|---:|---:|
| valid_2025_09 | 11310.428 | 10369.084 | 10217.257 | 10446.032 |
| valid_2025_10 | 7899.103 | 7451.142 | 6942.142 | 6814.620 |
| valid_2025_11 | 7120.767 | 6210.285 | 6991.060 | 6584.492 |
| valid_2025_12 | 5928.223 | 5781.292 | 5799.627 | 6377.158 |
| valid_2025_jan_feb | 14153.500 | 14242.104 | 13687.295 | 14164.877 |

## Conclusions

- Do not promote weekly-relative features as a global champion replacement.
- `weekly_delta` improves `jan_feb_like` by about `+88.6`, but hurts the standard September-December regime by about `-601.0`.
- `weekly_delta_percentile` improves December and the stricter `late_winter_12_jan_feb` regime by about `+167.3`, but still hurts standard rolling substantially.
- `weekly_percentile` is rejected: it is worse in all key regimes.
- The result supports the validation lesson: weekly relative features may be winter useful, but they are not robust enough to add globally.

## Decision

Keep the code and configs as winter-candidate components. Do not use them in the current champion submit path yet.

The next useful step is not to add more weekly features. It is to test a conservative winter-only blend or gate, for example:

- use champion by default;
- allow `weekly_delta_percentile` only when the target period is winter-like or when a winter validation rule justifies it;
- compare against `holiday_only`, which also looked stronger under Jan-Feb-like validation.
