# Promotion Gate

Current mode: warn.

## Current Champion

- Run ID: `ens_baseline_baseline_last_180d`
- Alias: `ens_base_base180`
- Config: `configs/ensemble_baseline_and_baseline_last_180d.yaml`
- Backtest summary: `reports/backtest_ens_baseline_baseline_last_180d.csv`
- Backtest daily detail: `reports/backtest_ens_baseline_baseline_last_180d_daily.csv`
- Candidate submission: `outputs/output_ens_baseline_baseline_last_180d.csv`
- Submission validation: passed

## Champion Metrics

- Mean validation profit: `7758.127428`
- Worst fold profit: `5847.162027`
- Mean oracle ratio: `0.684640`
- Minimum oracle ratio: `0.536254`
- Total validation loss days: `10`

## Promotion Criteria

A future candidate can replace the champion when:

- data contract checks pass.
- submission format checks pass.
- mean validation profit improves over current champion, or worst-fold profit improves with only small mean-profit loss.
- oracle ratio does not degrade materially.
- fold instability is reviewed.
- candidate config, summary report, daily report, and output CSV are recorded in `reports/experiment_index.csv`.

In warn mode, failures are recorded but do not block exploratory baseline creation.
