# Literature-Guided Plan for the Electricity Storage Competition

## Scope

This note translates electricity price forecasting (EPF) and storage-arbitrage literature into competition-specific experiment rules. The competition constraint is stricter than many EPF papers: final inference cannot use recent true historical prices, so methods that depend on price lags are not directly admissible for submission.

## Sources Reviewed

| ID | Source | Status | Competition relevance |
|---|---|---:|---|
| lago_2021_epf_review | Lago, Marcjasz, De Schutter, Weron, *Forecasting day-ahead electricity prices: a review of state-of-the-art algorithms, best practices and an open-access benchmark* | local PDF/text reviewed | Strong benchmark discipline, exogenous variables, rolling evaluation, calibration windows, ensembles, feature selection |
| nowotarski_weron_prob_review | Nowotarski/Weron, *Recent advances in electricity price forecasting: A review of probabilistic forecasting* | local PDF/text reviewed | GEFCom2014 price track, probabilistic/quantile ideas, external predictors, load forecasts, benchmark lessons |
| hong_2020_energy_forecasting | Hong et al., *Energy Forecasting: A Review and Outlook* | local PDF/text reviewed | General forecasting workflow, reproducibility, energy forecasting task framing |
| sioshansi_storage_pjm | Sioshansi et al., *Estimating the Value of Electricity Storage in PJM: Arbitrage and Some Welfare Effects* | local PDF/text reviewed | Storage arbitrage upper bound, perfect foresight benchmark, price/load relationship |
| gefcom2014_overview | Hong, Pinson, Fan et al., *Global Energy Forecasting Competition 2014 and beyond* | public source screened | Price track used day-ahead load forecasts and unified evaluation; high participation validates benchmark lessons |
| gefcom2014_price_methods | Gaillard/Goude/Nedellec and other GEFCom2014 price-track method discussions | public/source snippets screened | Robust aggregation, gradient boosting/GAM/QRA style ideas; must filter for price-lag dependency |
| epftoolbox | EPFtoolbox benchmark repository by Lago et al. | public repo screened | Benchmark methodology and reproducible model comparison |

## Competition-Specific Constraints

Allowed at final inference:

- Official test boundary-condition forecasts.
- Official test NWP files in `all_nc`.
- Calendar/time features derived from timestamps.
- Models trained on historical aligned train data.

Not allowed or not available for final inference:

- Recent true historical prices during the 2026-01 to 2026-02 test period.
- External data not provided by the competition.
- Any feature whose construction requires test labels or hidden actual values.

Implication: literature methods based on price lags such as `P(t-24)`, `P(t-168)`, same-hour last-week price, ARX with true recent prices, or naive price benchmarks are useful as conceptual baselines but cannot be final submission features unless replaced by available proxies.

## What the Literature Suggests for This Task

### 1. Keep strong simple baselines and rolling validation

EPF reviews emphasize that many proposed complex models are over-claimed because they are not compared against strong simple baselines or evaluated on long, realistic test periods. Our response:

- Keep `lgb_baseline` as the current reference.
- Use rolling month folds, not random split.
- Promote only models improving profit/oracle ratio without increasing instability.

### 2. Use exogenous supply-demand predictors, not unavailable price lags

EPF benchmark datasets commonly include day-ahead load forecasts and renewable generation forecasts as exogenous variables. Our competition provides richer official exogenous variables. Priority feature families:

- Supply-demand tightness: `load - renewable - hydro`, ratios to load, non-market generation pressure.
- Renewable pressure: wind/solar/hydro levels and penetration ratios.
- Interchange/market boundary: tie-line ratio and interaction with net load.
- Calendar/time structure: hour, quarter, weekday, month, cyclic encodings.
- NWP summaries: wind speed, irradiance, cloud cover, temperature, precipitation summaries over the target region.

### 3. Feature selection is necessary

The EPF review highlights feature selection/regularization and warns that higher-dimensional models do not automatically improve out-of-sample accuracy. Our first full derived-feature run supports this: loss days improved but average profit fell. Therefore:

- Do feature-family ablation before adding more features.
- Do not keep all derived features just because they are business-plausible.
- Use feature importance only as diagnostic; final decision is rolling profit.

### 4. Calibration windows and ensembles are worth testing

EPF literature discusses averaging forecasts from different calibration windows because short windows adapt faster while long windows fit more data. Our data has only one year, but we can still test:

- Expanding window: current fold design.
- Recent window: train only last 90/120/180 days before validation.
- Ensemble across expanding and recent-window models.

This is directly relevant to our observed month instability: 10/12 differ from 9/11.

### 5. Probabilistic thinking can help, but only after point baselines stabilize

Probabilistic EPF literature and GEFCom2014 price-track methods motivate quantile forecasts and robust aggregation. For this competition, uncertainty is useful if it improves dispatch under noisy price curves. But the first tau-threshold experiment failed because predicted spread did not separate loss days. Next uncertainty work should be model-ensemble based, not simple spread threshold only:

- Train multiple models/feature sets.
- Dispatch on mean prediction.
- Use disagreement as a confidence signal only after validation proves it filters bad days.

### 6. Storage literature supports oracle benchmarking

Storage arbitrage papers use perfect foresight as an upper bound. We should continue reporting:

- Oracle profit.
- Model profit/oracle ratio.
- Regret = oracle profit - model profit.
- Charge/discharge start gap.

These are more aligned with the competition than RMSE alone.

## Methods to Deprioritize or Reject for Now

| Method | Reason |
|---|---|
| Price-lag AR/LEAR/naive features | Strong in EPF literature but unavailable in final test because true recent test prices are not provided |
| Random train/valid split | Leakage-like for time series and rejected by EPF best practices |
| Deep models before feature/validation discipline | Literature warns complex models often fail against strong baselines without fair comparison; current data size is small |
| Blind all-feature expansion | First full derived-feature run reduced average profit |
| Pure RMSE model selection | Final score is storage profit; RMSE can miss window-selection errors |

## Next Experiments, Ordered by Literature Support and Competition Fit

### Experiment A: Derived feature-family ablation

Purpose: identify which exogenous feature family helps without overfitting.

Configs to create:

- `lgb_business`: net load and ratio features only.
- `lgb_rank`: day-within rank features only.
- `lgb_deviation`: day-within deviation-from-mean features only.

Promotion criterion:

- Improve average profit vs `lgb_baseline`, or at least improve worst-fold profit and loss days without large average loss.

### Experiment B: Calibration-window comparison

Purpose: test whether recent market regimes matter.

Configs:

- expanding window: current baseline.
- 90-day recent window.
- 180-day recent window.

Promotion criterion:

- Improve 10/12 folds without damaging 9/11 too much.

### Experiment C: Forecast ensemble

Purpose: use literature-supported model averaging to stabilize fold-specific behavior.

Candidates:

- average predictions from baseline + best feature-family model.
- average expanding-window + recent-window predictions.

Promotion criterion:

- Improve mean oracle ratio and reduce loss days.

### Experiment D: NWP lightweight summaries

Purpose: use official weather predictors without heavy deep models.

First version:

- Per target day/hour, aggregate `ghi`, `t2m`, `tcc`, `tp`, `sp`, `u100`, `v100` over the grid.
- Add wind speed magnitude `sqrt(u100^2 + v100^2)`.
- Repeat hourly NWP features to 15-minute rows.

Promotion criterion:

- Improve 10/12 difficult folds or reduce window gaps.

## Current Decision

Do not immediately run more broad feature additions. The next execution step is Experiment A: feature-family ablation under `absolute` target and the existing rolling backtest.
