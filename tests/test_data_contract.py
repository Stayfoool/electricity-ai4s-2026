from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile


def test_required_local_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "eletricmaterial" / "to_sais_new.zip").exists()
    assert (root / "eletricmaterial" / "output_demo.csv").exists()


def test_zip_contains_required_members() -> None:
    root = Path(__file__).resolve().parents[1]
    zip_path = root / "eletricmaterial" / "to_sais_new.zip"
    expected = {
        "to_sais_new/train/mengxi_boundary_anon_filtered.csv",
        "to_sais_new/train/mengxi_node_price_selected.csv",
        "to_sais_new/test/test_in_feature_ori.csv",
    }
    with ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert expected <= names
