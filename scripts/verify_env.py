from __future__ import annotations

import importlib
import shutil
from pathlib import Path

REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "sklearn",
    "lightgbm",
    "xgboost",
    "xarray",
    "netCDF4",
    "mlflow",
    "optuna",
    "dvc",
    "hydra",
    "omegaconf",
    "pytest",
    "ruff",
    "mypy",
]


def main() -> None:
    root = Path.cwd()
    missing = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            missing.append((module, str(exc)))

    print(f"project_root={root}")
    print("python_ok=True")
    print(f"nvidia_smi={shutil.which('nvidia-smi') or 'not-found'}")
    print(f"data_zip_exists={(root / 'eletricmaterial' / 'to_sais_new.zip').exists()}")

    if missing:
        for module, error in missing:
            print(f"missing_or_broken={module}: {error}")
        raise SystemExit(1)

    print("env_ok=True")


if __name__ == "__main__":
    main()
