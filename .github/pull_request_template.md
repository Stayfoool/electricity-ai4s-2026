## Experiment Hypothesis

Describe what this PR is trying to improve and why it should help.

## Changed Files

- 

## AutoDL Command For Owner

```bash
make gate
make backtest CONFIG=configs/<experiment>.yaml
```

## Expected Metrics To Check

- `avg_profit`
- `loss_days`
- fold stability
- `oracle_ratio` / regret if available
- submit compatibility if this affects dispatch or submit generation

## Local Checks

- [ ] `make lint PY=python3`
- [ ] `make test PY=python3`
- [ ] Data layout unchanged, or `make data-check PY=python3` passed after data changes

## Risk Notes

Mention possible overfitting, single-month improvements, runtime cost, or submit risk.
