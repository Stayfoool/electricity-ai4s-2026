from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from electricity.features.bid_space import markdown_table
from electricity.features.weather import bjt_valid_times, hourly_renewable_frame, nc_dir

CONFIG_PATH = Path("configs/ensemble_champion_segmented6_prior.yaml")
REPORTS_DIR = Path("reports")
OUT_TOP = REPORTS_DIR / "weather_spatial_top_cells.csv"
OUT_STABILITY = REPORTS_DIR / "weather_spatial_stability.csv"
OUT_MD = REPORTS_DIR / "weather_spatial_selection.md"
OUT_PNG = REPORTS_DIR / "weather_spatial_top_cells.png"

CORE_CHANNELS = ["ghi", "tcc", "u100", "v100", "wind_speed"]
BASE_TARGETS = ["renewable_error", "wind_error", "solar_error"]
TOP_K = 30

SCOPES = {
    "all": lambda ts: True,
    "jan_feb": lambda ts: ts.month in {1, 2},
    "winter_like_01_02_11_12": lambda ts: ts.month in {1, 2, 11, 12},
    "daylight_07_18": lambda ts: 7 <= ts.hour <= 18,
    "midday_10_17": lambda ts: 10 <= ts.hour <= 17,
    "winter_daylight": lambda ts: ts.month in {1, 2, 11, 12} and 7 <= ts.hour <= 18,
}


@dataclass
class CorrAccum:
    n: int
    sum_y: float
    sum_y2: float
    sum_x: np.ndarray
    sum_x2: np.ndarray
    sum_xy: np.ndarray

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> CorrAccum:
        return cls(
            n=0,
            sum_y=0.0,
            sum_y2=0.0,
            sum_x=np.zeros(shape, dtype=np.float64),
            sum_x2=np.zeros(shape, dtype=np.float64),
            sum_xy=np.zeros(shape, dtype=np.float64),
        )

    def update(self, x: np.ndarray, y: float) -> None:
        if not np.isfinite(y):
            return
        values = np.asarray(x, dtype=np.float64)
        if np.isnan(values).any():
            values = np.nan_to_num(values, nan=0.0)
        self.n += 1
        self.sum_y += y
        self.sum_y2 += y * y
        self.sum_x += values
        self.sum_x2 += values * values
        self.sum_xy += values * y

    def corr(self) -> np.ndarray:
        if self.n < 20:
            return np.full_like(self.sum_x, np.nan, dtype=np.float64)
        n = float(self.n)
        num = n * self.sum_xy - self.sum_x * self.sum_y
        den_x = n * self.sum_x2 - self.sum_x * self.sum_x
        den_y = n * self.sum_y2 - self.sum_y * self.sum_y
        den = np.sqrt(np.maximum(den_x, 0.0) * max(den_y, 0.0))
        out = np.full_like(self.sum_x, np.nan, dtype=np.float64)
        np.divide(num, den, out=out, where=den > 1e-12)
        return out


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fold_defs(cfg: dict) -> list[dict[str, object]]:
    folds = []
    for fold in cfg["folds"]:
        folds.append(
            {
                "name": fold["name"],
                "train_end": pd.Timestamp(fold["train_end"]),
            }
        )
    return folds


def add_fold_target_residuals(
    hourly: pd.DataFrame,
    folds: list[dict[str, object]],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for fold in folds:
        name = str(fold["name"])
        train = hourly[hourly["target_hour"] <= fold["train_end"]].copy()
        enriched = hourly.copy()
        for target in BASE_TARGETS:
            global_mean = float(train[target].mean())
            mh = train.groupby(["month", "hour"], observed=True)[target].mean()
            month = train.groupby("month", observed=True)[target].mean()
            vals = []
            for row in enriched[["month", "hour"]].itertuples(index=False):
                key = (row.month, row.hour)
                if key in mh.index:
                    vals.append(float(mh.loc[key]))
                elif row.month in month.index:
                    vals.append(float(month.loc[row.month]))
                else:
                    vals.append(global_mean)
            enriched[f"{target}_mh_resid"] = (
                enriched[target].to_numpy(dtype=float) - np.asarray(vals)
            )
        out[name] = enriched.set_index("target_hour")
    return out


def read_core_arrays(ds: xr.Dataset, lead_idx: int) -> dict[str, np.ndarray]:
    data = ds["data"].isel(time=0, lead_time=lead_idx)
    u100 = data.sel(channel="u100").to_numpy()
    v100 = data.sel(channel="v100").to_numpy()
    return {
        "ghi": data.sel(channel="ghi").to_numpy(),
        "tcc": data.sel(channel="tcc").to_numpy(),
        "u100": u100,
        "v100": v100,
        "wind_speed": np.sqrt(u100**2 + v100**2),
    }


def active_scopes(ts: pd.Timestamp) -> list[str]:
    return [name for name, fn in SCOPES.items() if fn(ts)]


def top_cells_from_accums(
    accums: dict[tuple[str, str, str, str], CorrAccum],
    *,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
) -> pd.DataFrame:
    lat_grid, lon_grid = np.meshgrid(lat_values, lon_values, indexing="ij")
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    rows: list[dict[str, object]] = []
    for (fold, scope, target, channel), acc in sorted(accums.items()):
        corr = acc.corr().ravel()
        valid = np.where(np.isfinite(corr))[0]
        if len(valid) == 0:
            continue
        order = valid[np.argsort(np.abs(corr[valid]))[::-1]][:TOP_K]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "fold": fold,
                    "scope": scope,
                    "target": target,
                    "weather_var": channel,
                    "rank": rank,
                    "n_hours": acc.n,
                    "pearson": float(corr[idx]),
                    "abs_pearson": float(abs(corr[idx])),
                    "lat": float(lat_flat[idx]),
                    "lon": float(lon_flat[idx]),
                    "cell_id": f"lat={lat_flat[idx]:.2f},lon={lon_flat[idx]:.2f}",
                }
            )
    return pd.DataFrame(rows)


def stability_table(top: pd.DataFrame) -> pd.DataFrame:
    if top.empty:
        return pd.DataFrame()
    top10 = top[top["rank"] <= 10].copy()
    grouped = top10.groupby(["scope", "target", "weather_var", "cell_id"], observed=True)
    out = grouped.agg(
        folds_seen=("fold", "nunique"),
        mean_abs_pearson=("abs_pearson", "mean"),
        max_abs_pearson=("abs_pearson", "max"),
        mean_rank=("rank", "mean"),
        lat=("lat", "first"),
        lon=("lon", "first"),
    ).reset_index()
    return out.sort_values(
        ["scope", "target", "weather_var", "folds_seen", "mean_abs_pearson"],
        ascending=[True, True, True, False, False],
    )


def plot_top_cells(top: pd.DataFrame) -> None:
    subset = top[
        (top["rank"] <= 5)
        & (
            top["target"].isin(
                ["renewable_error", "renewable_error_mh_resid", "solar_error", "wind_error"]
            )
        )
        & (top["scope"].isin(["all", "jan_feb", "winter_like_01_02_11_12", "winter_daylight"]))
    ].copy()
    if subset.empty:
        return
    subset["label"] = subset["scope"] + "\n" + subset["target"]
    labels = list(dict.fromkeys(subset["label"].tolist()))[:8]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, label in zip(axes_flat, labels, strict=False):
        data = subset[subset["label"] == label]
        sc = ax.scatter(
            data["lon"],
            data["lat"],
            c=data["pearson"],
            s=80,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.grid(alpha=0.2)
    for ax in axes_flat[len(labels) :]:
        ax.axis("off")
    fig.colorbar(sc, ax=axes_flat.tolist(), shrink=0.8, label="Pearson correlation")
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)


def write_report(top: pd.DataFrame, stability: pd.DataFrame, total_files: int) -> None:
    focus = top[
        (top["rank"] <= 5)
        & (
            top["target"].isin(
                ["renewable_error_mh_resid", "wind_error_mh_resid", "solar_error_mh_resid"]
            )
        )
        & (top["scope"].isin(["jan_feb", "winter_like_01_02_11_12", "winter_daylight", "all"]))
    ].copy()
    focus = focus.sort_values(["scope", "target", "weather_var", "fold", "rank"])
    stable = stability[stability["folds_seen"] >= 2].head(40) if not stability.empty else stability
    lines = [
        "# Weather Spatial Selection Diagnostics",
        "",
        "## Goal",
        "",
        "Find whether specific NWP grid cells, using only core weather variables, "
        "explain renewable forecast errors better than coarse full-grid averages.",
        "",
        "## Method",
        "",
        f"- NetCDF files scanned: `{total_files}`.",
        "- Core weather variables: `ghi`, `tcc`, `u100`, `v100`, `wind_speed`.",
        "- Targets: `renewable_error`, `wind_error`, `solar_error`, plus month-hour "
        "residual versions.",
        "- Each fold uses only hours before that fold's `train_end` for spatial ranking.",
        "- Correlation is computed per grid cell with streaming accumulators, avoiding "
        "a huge expanded table.",
        "- Month-hour residual targets reduce pure seasonal/daytime effects in the target side.",
        "",
        "## Top Focus Cells",
        "",
        markdown_table(focus.head(80), floatfmt=".4f") if not focus.empty else "No focus rows.",
        "",
        "## Stable Top-10 Cells Across Folds",
        "",
        (
            markdown_table(stable, floatfmt=".4f")
            if not stable.empty
            else "No cells appeared in top-10 for at least two folds."
        ),
        "",
        "## Readout",
        "",
        "- Stable cells across folds are safer candidates for correction features than "
        "one-fold top cells.",
        "- If winter-like scopes show stronger and more stable cells than `all`, "
        "weather should be treated as a winter-specific signal.",
        "- This report does not yet prove profit lift; it only decides whether spatial "
        "weather correction is worth backtesting.",
        "",
        "## Artifacts",
        "",
        f"- `{OUT_TOP}`",
        f"- `{OUT_STABILITY}`",
        f"- `{OUT_PNG}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    folds = fold_defs(cfg)
    hourly = hourly_renewable_frame(cfg)
    fold_frames = add_fold_target_residuals(hourly, folds)
    target_names = BASE_TARGETS + [f"{t}_mh_resid" for t in BASE_TARGETS]

    paths = sorted(nc_dir(cfg).glob("*.nc"))
    if not paths:
        raise FileNotFoundError("no NWP nc files found")

    accums: dict[tuple[str, str, str, str], CorrAccum] = {}
    lat_values = lon_values = None
    total = len(paths)
    for i, path in enumerate(paths, start=1):
        if i == 1 or i % 25 == 0 or i == total:
            print(f"scanning {i}/{total} {path.name}", flush=True)
        with xr.open_dataset(path) as ds:
            if lat_values is None:
                lat_values = ds["lat"].values
                lon_values = ds["lon"].values
            valid_times = bjt_valid_times(ds)
            for lead_idx, ts in enumerate(valid_times):
                ts = pd.Timestamp(ts)
                scopes = active_scopes(ts)
                if not scopes:
                    continue
                active_folds = [f for f in folds if ts <= f["train_end"]]
                if not active_folds:
                    continue
                arrays = read_core_arrays(ds, lead_idx)
                shape = next(iter(arrays.values())).shape
                for fold in active_folds:
                    fold_name = str(fold["name"])
                    frame = fold_frames[fold_name]
                    if ts not in frame.index:
                        continue
                    row = frame.loc[ts]
                    for scope in scopes:
                        for target in target_names:
                            y = float(row[target])
                            for channel, arr in arrays.items():
                                key = (fold_name, scope, target, channel)
                                if key not in accums:
                                    accums[key] = CorrAccum.zeros(shape)
                                accums[key].update(arr, y)

    if lat_values is None or lon_values is None:
        raise RuntimeError("failed to read lat/lon")
    top = top_cells_from_accums(accums, lat_values=lat_values, lon_values=lon_values)
    stability = stability_table(top)
    top.to_csv(OUT_TOP, index=False)
    stability.to_csv(OUT_STABILITY, index=False)
    plot_top_cells(top)
    write_report(top, stability, total_files=total)
    print(f"top_path={OUT_TOP}")
    print(f"stability_path={OUT_STABILITY}")
    print(f"markdown_path={OUT_MD}")
    print(f"png_path={OUT_PNG}")


if __name__ == "__main__":
    main()
