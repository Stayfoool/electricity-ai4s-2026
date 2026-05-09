from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_signature(path: Path) -> tuple[int, int, str]:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    total_size = sum(p.stat().st_size for p in files)
    digest = hashlib.sha256()
    for file_path in files:
        rel = file_path.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(file_path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
    return len(files), total_size, digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_manifest(manifest_path: Path) -> int:
    manifest = load_manifest(manifest_path)
    root = Path(manifest["root"])
    errors: list[str] = []
    warnings: list[str] = []

    for item in manifest.get("files", []):
        path = root / item["path"]
        if not path.exists():
            if item.get("required", True):
                errors.append(f"missing file: {path}")
            continue
        size = path.stat().st_size
        expected_size = item.get("size_bytes")
        if expected_size is not None and size != expected_size:
            errors.append(f"size mismatch: {path} expected={expected_size} actual={size}")
        elif expected_size is None:
            warnings.append(f"size not pinned: {path} actual={size}")

        expected_hash = item.get("sha256")
        if expected_hash is not None:
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                errors.append(
                    f"sha256 mismatch: {path} expected={expected_hash} actual={actual_hash}"
                )
        else:
            warnings.append(f"sha256 not pinned: {path}")

    for item in manifest.get("directories", []):
        path = root / item["path"]
        if not path.exists():
            if item.get("required", True):
                errors.append(f"missing directory: {path}")
            continue
        count, total_size, signature = directory_signature(path)
        expected_count = item.get("file_count")
        expected_size = item.get("total_size_bytes")
        expected_signature = item.get("file_list_sha256")
        if expected_count is not None and count != expected_count:
            errors.append(f"file count mismatch: {path} expected={expected_count} actual={count}")
        elif expected_count is None:
            warnings.append(f"file count not pinned: {path} actual={count}")
        if expected_size is not None and total_size != expected_size:
            errors.append(
                f"directory size mismatch: {path} expected={expected_size} actual={total_size}"
            )
        elif expected_size is None:
            warnings.append(f"directory size not pinned: {path} actual={total_size}")
        if expected_signature is not None and signature != expected_signature:
            errors.append(
                f"file list signature mismatch: {path} "
                f"expected={expected_signature} actual={signature}"
            )
        elif expected_signature is None:
            warnings.append(f"file list signature not pinned: {path} actual={signature}")

    for warning in warnings:
        print(f"WARN {warning}")
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    print("data_manifest_ok=true")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data_manifest.json")
    args = parser.parse_args()
    raise SystemExit(check_manifest(Path(args.manifest)))


if __name__ == "__main__":
    main()
