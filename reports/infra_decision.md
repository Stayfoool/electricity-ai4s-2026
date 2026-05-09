# Infrastructure Decision

## Context

The competition data is moderate in row count but includes a 4GB zip and many NWP NetCDF files. The initial winning path is expected to rely on tabular models, rolling backtests, exact dispatch, and repeated experiments.

## Decision

Use AutoDL for the primary environment. Start in no-card mode for setup; switch to GPU/card mode only when needed for heavier model runs.

## Current Target

- Project path: `/root/autodl-tmp/electricity`
- Environment: conda env `electricity`
- Disk: user-selected 200GB data disk target; monitor free space before large feature caches.

## Upgrade Triggers

- Single rolling backtest exceeds 1 hour.
- Feature cache exceeds 150GB.
- Deep sequence model experiments become active.
- Daily experiment queue cannot finish overnight.

