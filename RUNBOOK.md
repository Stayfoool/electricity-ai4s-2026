# Runbook

Remote project root:

```bash
cd /root/autodl-tmp/electricity
```

Environment:

```bash
/root/miniconda3/envs/electricity/bin/python --version
make verify
```

Default workflow:

```bash
make test
make backtest
```

Team PR workflow:

```bash
make lint PY=python3
make test PY=python3
```

Official AutoDL gate:

```bash
make gate
make backtest CONFIG=configs/<experiment>.yaml
```

Data check is manual. Run it after rebuilding AutoDL data, switching machines, or
debugging suspicious results:

```bash
make data-check
```

Harness mode:

- `warn`: record issues without blocking early baseline work.
- `enforce`: block candidate promotion when contracts or gates fail.
