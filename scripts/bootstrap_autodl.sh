#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/root/autodl-tmp/electricity}"
CONDA="/root/miniconda3/bin/conda"
PY="/root/miniconda3/envs/electricity/bin/python"

mkdir -p "$PROJECT_ROOT"
if ! "$CONDA" env list | grep -q '^electricity '; then
  "$CONDA" create -n electricity python=3.10 -y
fi

"$PY" -m pip install -U pip
"$PY" -m pip install -r "$PROJECT_ROOT/requirements.txt"
"$PY" scripts/verify_env.py

