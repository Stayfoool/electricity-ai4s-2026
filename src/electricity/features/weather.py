from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from electricity.data import load_train_frame
from electricity.features.bid_space import ACTUAL_COLS, PRED_COLS

REPORTS_DIR = Path("reports")
WEATHER_CACHE_PATH = REPORTS_DIR / "weather_hourly_spatial_features.csv"

CHANNEL_UNITS = {
    "u100": "m/s",
    "v100": "m/s",
    "t2m": "K",
    "tp": "m",
    "tcc": "0-1",
    "sp": "Pa",
    "ghi": "W/m^2",
    "wind_speed": "m/s",
}

CORE_WEATHER_FEATURES = [
    "ghi_mean",
    "ghi_max",
    "tcc_mean",
    "tcc_max",
    "tp_mean",
    "wind_speed_mean",
    "wind_speed_max",
    "u100_mean",
    "v100_mean",
    "t2m_mean",
    "sp_mean",
]


def nc_dir(cfg: dict) -> Path:
    root = Path.cwd()
    material_dir = root / cfg["paths"]["material_dir"]
    path = material_dir / "to_sais_new" / "all_nc"
    if not path.exists():
        raise FileNotFoundError(f"NWP directory not found: {path}")
    return path


def bjt_valid_times(ds: xr.Dataset) -> pd.DatetimeIndex:
    start_utc = pd.Timestamp(ds["time"].values[0])
    lead_hours = pd.to_timedelta(ds["lead_time"].values.astype(int), unit="h")
    return pd.DatetimeIndex(start_utc + lead_hours + pd.Timedelta(hours=8))


def summarize_nc_file(path: Path) -> pd.DataFrame:
    release_date = pd.to_datetime(path.stem, format="%Y%m%d")
    target_date = release_date + pd.Timedelta(days=1)
    with xr.open_dataset(path) as ds:
        valid_times = bjt_valid_times(ds)
        data = ds["data"].isel(time=0)
        rows: dict[str, object] = {
            "release_date": release_date.date().isoformat(),
            "target_date": target_date.date().isoformat(),
            "target_hour": valid_times,
        }
        for channel in ds["channel"].values.tolist():
            da = data.sel(channel=channel)
            rows[f"{channel}_mean"] = da.mean(dim=("lat", "lon")).values
            rows[f"{channel}_std"] = da.std(dim=("lat", "lon")).values
            rows[f"{channel}_min"] = da.min(dim=("lat", "lon")).values
            rows[f"{channel}_max"] = da.max(dim=("lat", "lon")).values

        u = data.sel(channel="u100")
        v = data.sel(channel="v100")
        wind_speed = np.sqrt(u**2 + v**2)
        rows["wind_speed_mean"] = wind_speed.mean(dim=("lat", "lon")).values
        rows["wind_speed_std"] = wind_speed.std(dim=("lat", "lon")).values
        rows["wind_speed_min"] = wind_speed.min(dim=("lat", "lon")).values
        rows["wind_speed_max"] = wind_speed.max(dim=("lat", "lon")).values
        rows["ghi_positive_frac"] = (data.sel(channel="ghi") > 50).mean(
            dim=("lat", "lon")
        ).values
        rows["high_cloud_frac"] = (data.sel(channel="tcc") > 0.8).mean(
            dim=("lat", "lon")
        ).values
    out = pd.DataFrame(rows)
    out["valid_date"] = out["target_hour"].dt.normalize()
    out["hour"] = out["target_hour"].dt.hour
    out["month"] = out["target_hour"].dt.month
    return out


def load_weather_features(cfg: dict, *, refresh: bool = False) -> pd.DataFrame:
    if WEATHER_CACHE_PATH.exists() and not refresh:
        return pd.read_csv(WEATHER_CACHE_PATH, parse_dates=["target_hour", "valid_date"])
    frames = [summarize_nc_file(path) for path in sorted(nc_dir(cfg).glob("*.nc"))]
    weather = pd.concat(frames, ignore_index=True).sort_values("target_hour")
    REPORTS_DIR.mkdir(exist_ok=True)
    weather.to_csv(WEATHER_CACHE_PATH, index=False)
    return weather


def hourly_renewable_frame(cfg: dict) -> pd.DataFrame:
    df = load_train_frame(cfg).copy()
    df["target_hour"] = df["times"].dt.floor("h")
    df["valid_date"] = df["target_hour"].dt.normalize()
    df["hour"] = df["target_hour"].dt.hour
    df["month"] = df["target_hour"].dt.month
    df["segment"] = pd.cut(
        df["hour"],
        bins=[0, 6, 10, 14, 18, 22, 24],
        labels=["00_06", "06_10", "10_14", "14_18", "18_22", "22_24"],
        right=False,
        include_lowest=True,
    )
    df["renewable_actual"] = df[ACTUAL_COLS["renewable"]]
    df["renewable_forecast"] = df[PRED_COLS["renewable"]]
    df["renewable_error"] = df["renewable_forecast"] - df["renewable_actual"]
    df["wind_actual"] = df[ACTUAL_COLS["wind"]]
    df["wind_forecast"] = df[PRED_COLS["wind"]]
    df["wind_error"] = df["wind_forecast"] - df["wind_actual"]
    df["solar_actual"] = df[ACTUAL_COLS["solar"]]
    df["solar_forecast"] = df[PRED_COLS["solar"]]
    df["solar_error"] = df["solar_forecast"] - df["solar_actual"]
    boundary_cols = [
        PRED_COLS["load"],
        ACTUAL_COLS["load"],
        PRED_COLS["tie_line"],
        ACTUAL_COLS["tie_line"],
        PRED_COLS["hydro"],
        ACTUAL_COLS["hydro"],
        PRED_COLS["non_market"],
        ACTUAL_COLS["non_market"],
    ]

    agg = {
        "rows_15min": ("times", "count"),
        "renewable_actual": ("renewable_actual", "mean"),
        "renewable_forecast": ("renewable_forecast", "mean"),
        "renewable_error": ("renewable_error", "mean"),
        "renewable_abs_error": ("renewable_error", lambda s: s.abs().mean()),
        "wind_actual": ("wind_actual", "mean"),
        "wind_forecast": ("wind_forecast", "mean"),
        "wind_error": ("wind_error", "mean"),
        "solar_actual": ("solar_actual", "mean"),
        "solar_forecast": ("solar_forecast", "mean"),
        "solar_error": ("solar_error", "mean"),
    }
    for col in boundary_cols:
        agg[col] = (col, "mean")
    return df.groupby(
        ["target_hour", "valid_date", "month", "hour", "segment"], observed=True
    ).agg(**agg).reset_index()


def merge_weather_renewable(cfg: dict) -> pd.DataFrame:
    renewable = hourly_renewable_frame(cfg)
    weather = load_weather_features(cfg)
    merged = renewable.merge(weather, on=["target_hour", "valid_date", "month", "hour"])
    merged["daylight"] = merged["ghi_mean"] > 20
    merged["midday"] = merged["hour"].between(10, 17)
    return merged
