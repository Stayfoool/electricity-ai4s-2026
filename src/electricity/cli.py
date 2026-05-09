from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from electricity.eval.backtest import run_backtest, run_ensemble_backtest, run_tau_search
from electricity.eval.pair_window_backtest import run_pair_window_backtest
from electricity.eval.segmented_backtest import run_segmented_backtest
from electricity.eval.window_backtest import run_window_backtest
from electricity.submit import run_ensemble_submit, run_submit


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(prog="electricity")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in [
        "backtest",
        "ensemble",
        "tune",
        "submit",
        "window-backtest",
        "pair-window-backtest",
        "segmented-backtest",
    ]:
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "backtest":
        run_backtest(cfg, config_path=args.config)
        return
    if args.command == "tune":
        run_tau_search(cfg, config_path=args.config)
        return
    if args.command == "ensemble":
        run_ensemble_backtest(cfg, config_path=args.config)
        return
    if args.command == "window-backtest":
        run_window_backtest(cfg, config_path=args.config)
        return
    if args.command == "pair-window-backtest":
        run_pair_window_backtest(cfg, config_path=args.config)
        return
    if args.command == "segmented-backtest":
        run_segmented_backtest(cfg, config_path=args.config)
        return
    if args.command == "submit":
        if "ensemble" in cfg:
            run_ensemble_submit(cfg, config_path=args.config)
            return
        run_submit(cfg, config_path=args.config)
        return
    print(f"command={args.command}")
    print(f"config={args.config}")
    print(f"project={cfg['project']['name']}")
    print("status=stub_ready")


if __name__ == "__main__":
    main()
