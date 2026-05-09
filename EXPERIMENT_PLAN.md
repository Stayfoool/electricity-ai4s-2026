# Electricity Price Forecasting & Storage Dispatch Experiment Plan

## 1. Goal and Constraints

Task: predict day-ahead 96-point real-time price sequence (15-minute resolution), then output daily charge/discharge plan to maximize profit under fixed storage constraints.

Official constraints we must enforce in all offline evaluation and online inference:

- No external data.
- Test phase does not provide true historical prices for recent days.
- One complete charge block and one complete discharge block at most per day:
  - charge block: `-1000` for 8 consecutive intervals.
  - discharge block: `+1000` for 8 consecutive intervals.
  - `0 <= t_c <= 80`, `t_d >= t_c + 8`, `t_d <= 88`.
  - or no operation for that day (`power=0` all day).
- Score is average daily profit across test days.

## 2. What We Learned from the Papers

Local papers reviewed:

- [Forecasting day-ahead electricity prices- A review of state-of-the-art  algorithms, best practices and an open-access benchmar.pdf](/Users/yw/ai/electricity/papers/Forecasting%20day-ahead%20electricity%20prices-%20A%20review%20of%20state-of-the-art%20%20algorithms,%20best%20practices%20and%20an%20open-access%20benchmar.pdf)
- [Recent advances in electricity price forecasting- A review of probabilistic  forecasting.pdf](/Users/yw/ai/electricity/papers/Recent%20advances%20in%20electricity%20price%20forecasting-%20A%20review%20of%20probabilistic%20%20forecasting.pdf)
- [Energy Forecasting- A Review and Outlook Hongetal2020.pdf](/Users/yw/ai/electricity/papers/Energy%20Forecasting-%20A%20Review%20and%20Outlook%20Hongetal2020.pdf)
- [Estimating the Value of Electricity Storage in PJM- Arbitrage and Some Welfare  Effects.pdf](/Users/yw/ai/electricity/papers/Estimating%20the%20Value%20of%20Electricity%20Storage%20in%20PJM-%20Arbitrage%20and%20Some%20Welfare%20%20Effects.pdf)

Key takeaways to apply directly:

- Strong simple baselines matter. Many papers overclaim improvements without fair benchmark and strict validation.
- Rolling / time-ordered evaluation is mandatory. Random split is misleading.
- Point forecast metrics (MAE/RMSE) do not fully reflect trading/dispatch value.
- Probabilistic thinking helps in volatile/negative-price regimes; uncertainty-aware dispatch can outperform pure point forecast ranking.
- Perfect-foresight storage value is an upper bound. Real forecast pipelines should track the gap to this bound.

## 2.1 Business and Policy Add-ons (China Market Context)

Additional authoritative references reviewed:

- [电力现货市场基本规则（试行）通知（发改能源规〔2023〕1217号）](https://zfxxgk.nea.gov.cn/2023-09/07/c_1310741791.htm)
- [《电力现货市场基本规则（试行）》答记者问](https://www.ndrc.gov.cn/xxgk/jd/jd/202309/t20230918_1360663.html)
- [关于进一步推动新型储能参与电力市场和调度运用（发改办运行〔2022〕475号）](https://www.gov.cn/zhengce/zhengceku/2022-06/07/content_5694423.htm)
- [关于加快推动新型储能发展的指导意见（发改能源规〔2021〕1051号）](https://zfxxgk.nea.gov.cn/2021-07/15/c_1310079331.htm)
- [EPFtoolbox benchmark repository](https://github.com/jeslago/epftoolbox)
- [Comparing Predictive Accuracy (Diebold-Mariano)](https://www.nber.org/papers/t0169)

Competition-relevant implications:

- Price signals are expected to reflect time-varying supply-demand tightness and flexibility scarcity. For this task, we should prioritize features that proxy supply stack pressure and renewable variability.
- Storage value in policy and market practice is multi-source, but this competition score captures only energy arbitrage. Model selection must therefore optimize arbitrage profit under the given dispatch constraints, not broader system value.
- The market-policy context supports treating negative prices as legitimate operating states rather than pure data errors.
- Evaluation rigor from EPF literature must be retained: rolling time splits, fair benchmarking, and formal forecast comparison tests before claiming model superiority.

## 3. Dataset Facts (Our Local Data)

Based on local files in [eletricmaterial](/Users/yw/ai/electricity/eletricmaterial):

- Train feature file: 2025-01-01 00:15:00 to 2025-12-31 23:45:00, 35039 rows.
- Train label file (`A`): 34922 rows (some timestamps missing vs feature file).
- Test feature file: 2026-01-01 00:00:00 to 2026-02-28 23:45:00, 5664 rows = 59 days * 96.
- Label distribution:
  - negative price ratio: about `9.56%`.
  - p50: `0.891`, p95: `3.185`, p99: `4.772`.
- Preliminary correlations (train, timestamp-aligned):
  - strong negative: wind+solar total forecast, wind forecast, non-market generation forecast.
  - weak: load forecast, tie-line forecast.

Implication:

- Negative price is a real regime, not outlier noise to blindly remove.
- Renewable-related predicted supply features are likely primary drivers for short-term price level and spread.

## 4. Experiment Design Principles

- No leakage:
  - never use unavailable test-time information.
  - do not use true historical price lags in the final online model.
- Two-layer pipeline:
  - Layer A: forecast 96-point price path.
  - Layer B: exact optimization of discrete dispatch from predicted prices.
- Evaluation must include both forecast error and dispatch profit.
- Keep a strict reproducible experiment log per run.

## 5. Modeling Roadmap

## Phase 0: Reproducible Baseline (Day 1)

Model:

- LightGBM point forecast for `A`.
- Features:
  - seven forecast boundary features from official test schema.
  - time features: hour, quarter-hour, day-of-week, month, weekend.
- Train target:
  - direct `A`.
- Validation:
  - rolling split by date.

Dispatch:

- exact daily enumeration over `(t_c, t_d)` feasible pairs.
- if best predicted spread <= 0, choose no operation.

## Phase 1: Strong Tabular Models (Days 2-4)

Models:

- LightGBM (Huber/L2 variants).
- XGBoost (`reg:squarederror` and quantile if stable).
- Optional linear regularized baseline (ElasticNet/LASSO) for sanity.

Feature engineering:

- cyclic encodings for hour/quarter.
- interaction features:
  - net supply proxy: `风光总加预测值 - 系统负荷预测值`.
  - renewable penetration proxy: `风光总加预测值 / (系统负荷预测值 + eps)`.
- per-day normalization candidates (careful: only same-day known features).

Ensembling:

- simple average of top 2-3 models by validation profit.
- optional weighted average by fold profit.

## Phase 2: NWP Integration (Days 5-7)

Use `all_nc/*.nc`:

- derive low-dimensional daily/hourly weather summaries over fixed ROI:
  - means/std/max for `u100,v100,t2m,tp,tcc,sp,ghi` by hour.
  - derived wind speed magnitude from `u100,v100`.
- map publish date `D` to forecast target date `D+1`.
- align to 15-minute grid by repeat/interpolate within day.

Target:

- improve high/low segment ranking rather than only global MAE.

## Phase 3: Uncertainty-aware Dispatch (Days 8-10)

Approach:

- train quantile models (`q10/q50/q90`) or bootstrap ensemble.
- compute robust spread score:
  - conservative spread = `E[q10(discharge block)] - E[q90(charge block)]`.
- dispatch only if conservative spread > threshold.

Expected value:

- lower variance, potentially higher average score if noisy days are frequent.

## 6. Sample Construction and Validation

Timestamp alignment:

- inner join train feature and label on `times`.
- remove days with too many missing points for training folds.

Fold design (recommended):

- Fold 1: train 2025-01 to 2025-08, valid 2025-09
- Fold 2: train 2025-01 to 2025-09, valid 2025-10
- Fold 3: train 2025-01 to 2025-10, valid 2025-11
- Fold 4: train 2025-01 to 2025-11, valid 2025-12

Per fold outputs:

- MAE, RMSE.
- average daily profit.
- normalized profit ratio vs oracle (see below).

Oracle benchmark:

- On validation with true prices, compute daily perfect-foresight best feasible profit.
- Track `model_profit / oracle_profit`.
- This ratio is the core quality KPI for dispatch value.

## 7. Hyperparameter Starting Points

LightGBM start:

- `objective`: `huber` (and compare with `regression`).
- `learning_rate`: `0.03`.
- `num_leaves`: `63`.
- `max_depth`: `-1`.
- `feature_fraction`: `0.8`.
- `bagging_fraction`: `0.8`.
- `bagging_freq`: `1`.
- `min_data_in_leaf`: `100`.
- `lambda_l1`: `0.0`, `lambda_l2`: `1.0`.
- `num_boost_round`: up to `3000` with early stopping `100`.

XGBoost start:

- `objective`: `reg:squarederror`.
- `eta`: `0.03`.
- `max_depth`: `8`.
- `subsample`: `0.8`.
- `colsample_bytree`: `0.8`.
- `min_child_weight`: `5`.
- `reg_lambda`: `1.0`.
- `n_estimators`: `2000` with early stopping.

Dispatch threshold tuning:

- tune no-trade threshold on validation:
  - trade if predicted best spread > `tau`.
  - search `tau` in `[0, 0.05, 0.1, ..., 1.0]` price units.

## 8. Hardware Plan

Current project stage:

- Tabular models + feature engineering + n-fold backtest are CPU-friendly.
- Recommended local: Apple Silicon CPU is enough for first two phases.

When to rent AutoDL:

- only if we run heavy sequence deep models (Transformer/TCN with large sweep) or very large NWP feature expansion.

AutoDL suggestion if needed:

- GPU: `RTX 4090` (24GB) or `A5000` class.
- CPU RAM: `>=32GB`.
- Disk: `>=100GB` (data + experiments + artifacts).

For this competition, expected ROI:

- High for better validation automation and parallel search.
- Low for naive deep model brute-force without strict leakage-safe validation.

## 9. Deliverables and File Layout

Recommended structure:

- `src/data/` data loading and alignment
- `src/features/` feature builders (base + nwp)
- `src/models/` lgb/xgb trainers
- `src/dispatch/` exact optimizer for `(t_c, t_d)`
- `src/eval/` metrics and rolling backtest
- `configs/` model and fold configs
- `outputs/` predictions and submission csv
- `reports/` experiment logs

Run log must include:

- data version hash / file timestamp
- fold definition
- feature set id
- model params
- MAE/RMSE/profit/oracle ratio
- submission file path

## 10. Immediate Execution Checklist

Completed now:

- local paper corpus identified and converted to text.
- isolated virtual env created at [`.venv`](/Users/yw/ai/electricity/.venv).
- core packages installed (numpy/pandas/sklearn/lightgbm/xgboost/xarray/netCDF4/jupyter).

Next execution steps:

1. Implement data alignment + rolling backtest skeleton.
2. Implement exact dispatch optimizer and oracle benchmark.
3. Run Phase 0 baseline and produce first score report.
4. Iterate model/features by `profit` and `oracle ratio`, not RMSE alone.

## 11. High-Risk Pitfalls for This Competition

- Leakage by design:
  - using true price lag features that are unavailable in test.
  - validating with random split instead of rolling time split.
- Optimizing wrong target:
  - selecting models only by RMSE/MAE while profit deteriorates.
  - ignoring no-trade threshold tuning.
- Mishandling negative prices:
  - clipping/removing negative prices without validation evidence.
- Ignoring timestamp consistency:
  - train starts at `00:15`, test starts at `00:00`; daily reconstruction must be robust to this mismatch.
- Over-investing in complex DL too early:
  - deep models before establishing strong tabular + dispatch baseline usually delays score gains.

Practical guardrails:

- Every experiment report must include both error metrics and profit metrics.
- Keep an immutable leaderboard of offline runs with feature/model/threshold identifiers.
- Promote a model only if it improves mean profit and does not increase fold instability sharply.
