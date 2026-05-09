from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["warn", "enforce"], default="warn")
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    candidate = Path(args.candidate)
    exists = candidate.exists()
    message = f"candidate_exists={exists} path={candidate}"

    if args.mode == "enforce" and not exists:
        raise SystemExit(message)

    print(message)
    print(f"promotion_mode={args.mode}")


if __name__ == "__main__":
    main()

