from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def read_csv_from_zip(zip_path: Path, member: str, *, parse_dates: list[str]) -> pd.DataFrame:
    with ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            return pd.read_csv(f, parse_dates=parse_dates)


def load_train_frame(cfg: dict) -> pd.DataFrame:
    root = Path.cwd()
    zip_path = root / cfg["paths"]["data_zip"]
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]

    features = read_csv_from_zip(
        zip_path,
        cfg["data"]["train_feature"],
        parse_dates=[time_col],
    )
    labels = read_csv_from_zip(
        zip_path,
        cfg["data"]["train_label"],
        parse_dates=[time_col],
    )

    train = features.merge(labels[[time_col, target_col]], on=time_col, how="inner")
    train = train.sort_values(time_col).reset_index(drop=True)
    return train


def load_test_frame(cfg: dict) -> pd.DataFrame:
    root = Path.cwd()
    zip_path = root / cfg["paths"]["data_zip"]
    time_col = cfg["data"]["time_col"]

    test = read_csv_from_zip(
        zip_path,
        cfg["data"]["test_feature"],
        parse_dates=[time_col],
    )
    return test.sort_values(time_col).reset_index(drop=True)
