from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

STATS = ["mean", "std", "min", "max"]


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"nwp_{prefix}_mean": float(np.nanmean(values)),
        f"nwp_{prefix}_std": float(np.nanstd(values)),
        f"nwp_{prefix}_min": float(np.nanmin(values)),
        f"nwp_{prefix}_max": float(np.nanmax(values)),
    }


def extract_file(path: Path) -> list[dict[str, float | str]]:
    issue_date = pd.to_datetime(path.stem, format="%Y%m%d")
    target_date = issue_date + pd.Timedelta(days=1)

    rows: list[dict[str, float | str]] = []
    with xr.open_dataset(path) as ds:
        channels = [str(v) for v in ds["channel"].values]
        data = ds["data"].isel(time=0)
        u100 = data.sel(channel="u100").to_numpy()
        v100 = data.sel(channel="v100").to_numpy()
        wind_speed = np.sqrt(u100**2 + v100**2)

        for lead_hour in range(24):
            row: dict[str, float | str] = {
                "issue_date": issue_date.date().isoformat(),
                "target_date": target_date.date().isoformat(),
                "hour": lead_hour,
            }
            for channel in channels:
                values = data.sel(channel=channel).isel(lead_time=lead_hour).to_numpy()
                row.update(_stats(channel, values))
            row.update(_stats("wind_speed100", wind_speed[lead_hour]))
            rows.append(row)
    return rows


def expand_to_15min(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    feature_cols = [c for c in hourly.columns if c not in {"issue_date", "target_date", "hour"}]
    for minute in [0, 15, 30, 45]:
        part = hourly[["target_date", "hour", *feature_cols]].copy()
        part["times"] = pd.to_datetime(part["target_date"]) + pd.to_timedelta(
            part["hour"], unit="h"
        ) + pd.Timedelta(minutes=minute)
        rows.append(part[["times", *feature_cols]])
    out = pd.concat(rows, ignore_index=True).sort_values("times").reset_index(drop=True)
    out["times"] = out["times"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="eletricmaterial/to_sais_new/all_nc")
    parser.add_argument("--output", default="artifacts/features/nwp_grid_15min.csv")
    parser.add_argument("--manifest", default="artifacts/features/nwp_grid_15min_manifest.md")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    manifest = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.nc"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"no nc files found in {input_dir}")

    all_rows: list[dict[str, float | str]] = []
    for i, path in enumerate(paths, start=1):
        print(f"extracting {i}/{len(paths)} {path.name}", flush=True)
        all_rows.extend(extract_file(path))

    hourly = pd.DataFrame(all_rows).sort_values(["target_date", "hour"]).reset_index(drop=True)
    expanded = expand_to_15min(hourly)
    expanded.to_csv(output, index=False)

    feature_cols = [c for c in expanded.columns if c != "times"]
    manifest.write_text(
        "# NWP Grid 15-Minute Features\n\n"
        f"- source_dir: `{input_dir}`\n"
        "- source_type: competition-provided NWP NetCDF files under `all_nc`.\n"
        "- alignment: file date D is treated as issue date; features align to Beijing "
        "target day D+1.\n"
        "- aggregation: full grid lat/lon mean/std/min/max per variable and per hour; "
        "hourly values are repeated to 15-minute rows.\n"
        "- derived: `nwp_wind_speed100_* = sqrt(u100^2 + v100^2)` grid statistics.\n"
        f"- nc_files: `{len(paths)}`\n"
        f"- output_rows: `{len(expanded)}`\n"
        f"- feature_count: `{len(feature_cols)}`\n"
        f"- time_min: `{expanded['times'].min()}`\n"
        f"- time_max: `{expanded['times'].max()}`\n\n"
        "## Columns\n\n"
        + "\n".join(f"- `{col}`" for col in ["times", *feature_cols])
        + "\n",
        encoding="utf-8",
    )
    print(f"output={output}")
    print(f"manifest={manifest}")
    print(f"shape={expanded.shape}")


if __name__ == "__main__":
    main()
