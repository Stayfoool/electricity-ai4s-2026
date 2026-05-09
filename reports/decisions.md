# Decision Log

## 2026-05-03: Use AutoDL as primary experiment environment

Context: Local Mac mini M4 has 16GB RAM, enough for first baseline but less comfortable for repeated backtests and larger NWP features.

Decision: Use AutoDL project root `/root/autodl-tmp/electricity` as the authority for experiment results. Keep local machine for editing and review.

Status: accepted.

## 2026-05-03: Start in warn-mode harness

Context: Full toolchain should be configured now, but early baseline should not be blocked by strict type/data gates.

Decision: Install and configure all tools, but keep promotion/data governance in warn mode until baseline is reproducible.

Status: accepted.

