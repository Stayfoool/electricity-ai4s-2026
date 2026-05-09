# Team Workflow

This project uses GitHub for code collaboration and AutoDL only for official
training/evaluation runs. Teammates do not need AutoDL access.

## Roles

- Owner: reviews PRs, runs official gates and backtests on AutoDL, promotes candidates.
- Teammate: implements code/config changes locally, runs lightweight checks, opens PRs.
- GitHub: source of truth for code, configs, tests, and experiment records.
- AutoDL: source of truth for official results using one environment and one data copy.

## What Goes Into Git

Commit:

- `src/`
- `scripts/`
- `configs/`
- `tests/`
- `reports/current_status.md`
- `reports/experiment_index.csv`
- lightweight reports needed to understand decisions

Do not commit:

- official competition data under `eletricmaterial/`
- official zip files and extracted raw data
- model weights or serialized models such as `.pt`, `.pkl`, `.joblib`, `.cbm`
- `mlruns/`, caches, local environments

## Data Handling

The official data is shared outside Git. The repository keeps only
`data_manifest.json`, which records the expected data files and hashes.

Expected local project layout:

```bash
eletricmaterial/to_sais_new/train/mengxi_boundary_anon_filtered.csv
eletricmaterial/to_sais_new/train/mengxi_node_price_selected.csv
eletricmaterial/to_sais_new/test/test_in_feature_ori.csv
eletricmaterial/to_sais_new/all_nc/
```

Run the data check after rebuilding AutoDL data, switching machines, syncing data,
or investigating suspicious results:

```bash
make data-check PY=python3
```

On AutoDL, use the project Python:

```bash
make data-check
```

## Branch and PR Flow

1. Agree on an experiment hypothesis before coding.
2. Create a branch named `exp/<short-name>` or `fix/<short-name>`.
3. Add or modify a config in `configs/` for the experiment.
4. Run local lightweight checks before opening a PR:

```bash
make lint PY=python3
make test PY=python3
```

5. Open a PR with the experiment hypothesis, files changed, and the exact command
   the owner should run on AutoDL.
6. The owner pulls the PR branch on AutoDL and runs the official gate. Add
   `make data-check` only after rebuilding or changing the data copy:

```bash
make gate
make backtest CONFIG=configs/<experiment>.yaml
```

7. If the result is useful, update:

```bash
reports/experiment_index.csv
reports/current_status.md
```

8. Merge only after the result is reproducible and the decision is recorded.

## Promotion Rules

A candidate is not promoted only because its mean profit is higher. Check:

- mean profit
- loss days
- fold stability
- oracle ratio or regret when available
- whether the change affects submit generation
- whether the result is explainable enough to trust

Do not replace `outputs/output.csv` unless the promotion decision is explicit.
