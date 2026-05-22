# Shallow LightGBM Champion Experiment

## Purpose

External competition discussions suggested stronger overfitting control for LightGBM, especially shallow trees (`max_depth` around 3-6), smaller leaf counts, and larger leaf-size constraints.

This experiment keeps the current champion structure fixed and changes only LightGBM tree complexity:

- same feature set
- same three-member ensemble weights `0.25 / 0.25 / 0.5`
- same 5fold validation design
- same dispatch prior `lambda_charge=0.18`, `lambda_discharge=0.40`

## Variants

| variant | max_depth | num_leaves | min_data_in_leaf | lambda_l2 |
|---|---:|---:|---:|---:|
| champion | default | 63 | 100 | 1.0 |
| shallow_d3_l7_leaf200 | 3 | 7 | 200 | 2.0 |
| shallow_d4_l15_leaf200 | 4 | 15 | 200 | 2.0 |
| shallow_d5_l31_leaf150 | 5 | 31 | 150 | 1.5 |

## Mean Profit By Regime

| model | standard_09_12 | jan_feb_like | winter_11_12_jan_feb | late_winter_12_jan_feb | all_5fold |
| --- | ---: | ---: | ---: | ---: | ---: |
| champion | 7991.351 | 14153.500 | 10170.888 | 11222.654 | 9952.035 |
| shallow_d3_l7_leaf200 | 7592.432 | 13966.266 | 10147.148 | 11156.146 | 9620.470 |
| shallow_d4_l15_leaf200 | 7757.352 | 13916.702 | 10089.789 | 11079.009 | 9717.146 |
| shallow_d5_l31_leaf150 | 7736.800 | 13939.276 | 10083.910 | 11094.066 | 9710.315 |

## Delta Vs Champion

| model | standard_09_12 | jan_feb_like | winter_11_12_jan_feb | late_winter_12_jan_feb | all_5fold |
| --- | ---: | ---: | ---: | ---: | ---: |
| champion | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| shallow_d3_l7_leaf200 | -398.919 | -187.234 | -23.741 | -66.508 | -331.565 |
| shallow_d4_l15_leaf200 | -233.999 | -236.797 | -81.100 | -143.646 | -234.889 |
| shallow_d5_l31_leaf150 | -254.551 | -214.224 | -86.978 | -128.588 | -241.720 |

## Loss Days By Regime

| model | standard_09_12 | jan_feb_like | winter_11_12_jan_feb | late_winter_12_jan_feb | all_5fold |
| --- | ---: | ---: | ---: | ---: | ---: |
| champion | 4 | 0 | 1 | 1 | 4 |
| shallow_d3_l7_leaf200 | 5 | 1 | 2 | 2 | 6 |
| shallow_d4_l15_leaf200 | 4 | 1 | 2 | 2 | 5 |
| shallow_d5_l31_leaf150 | 4 | 0 | 1 | 1 | 4 |

## Interpretation

- All shallow variants underperform champion on `all_5fold` and `jan_feb_like`.
- `shallow_d3_l7_leaf200` and `shallow_d4_l15_leaf200` improve November but lose too much in September, October, and Jan-Feb-like.
- `shallow_d5_l31_leaf150` is closest to champion but still loses about `-147.56` on `all_5fold` and about `-214.22` on `jan_feb_like`.
- The result suggests current champion is not mainly suffering from excessive tree depth; simple shallow-tree regularization is not a promotion path.

## Decision

- Reject these shallow variants as standalone submit candidates.
- Do not expand the shallow grid unless combined with a different feature or E2E objective.
- Next higher-value direction is weather feature/time-alignment audit and pair-level/E2E window-profit modeling.
