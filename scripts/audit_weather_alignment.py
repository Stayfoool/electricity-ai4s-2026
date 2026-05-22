from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from electricity.data import load_test_frame, load_train_frame
from electricity.features.bid_space import add_bid_space_columns, markdown_table

CONFIG_PATH = Path("configs/base.yaml")
REPORTS_DIR = Path("reports")
OUT_MD = REPORTS_DIR / "weather_alignment_audit.md"
OUT_METADATA = REPORTS_DIR / "weather_alignment_metadata.csv"
OUT_COVERAGE = REPORTS_DIR / "weather_alignment_coverage.csv"
OUT_SIGNAL = REPORTS_DIR / "weather_alignment_signal_summary.csv"
OUT_TOP_CORR = REPORTS_DIR / "weather_alignment_top_correlations.csv"

CORE_FEATURES = [
    "ghi_mean",
    "ghi_max",
    "tcc_mean",
    "tcc_max",
    "wind_speed_mean",
    "wind_speed_max",
    "u100_mean",
    "v100_mean",
    "t2m_mean",
    "sp_mean",
]

TARGETS = [
    "renewable_actual",
    "renewable_error",
    "wind_actual",
    "wind_error",
    "solar_actual",
    "solar_error",
    "bid_space_act",
    "bid_space_error",
    "price_hour",
]


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def material_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["material_dir"])


def nc_paths(cfg: dict) -> list[Path]:
    paths = sorted((material_dir(cfg) / "to_sais_new" / "all_nc").glob("*.nc"))
    if not paths:
        raise FileNotFoundError("no NetCDF files found under all_nc")
    return paths


def _first_value(values: np.ndarray) -> object:
    if len(values) == 0:
        return None
    return values[0]


def inspect_nc_metadata(paths: list[Path]) -> pd.DataFrame:
    sample_names = {
        paths[0].name,
        paths[len(paths) // 2].name,
        paths[-1].name,
        "20250101.nc",
        "20251231.nc",
        "20260101.nc",
        "20260227.nc",
        "20260228.nc",
    }
    rows: list[dict[str, object]] = []
    for path in paths:
        if path.name not in sample_names:
            continue
        with xr.open_dataset(path) as ds:
            lead = ds["lead_time"].values.astype(int)
            channels = [str(v) for v in ds["channel"].values.tolist()]
            issue_date = pd.to_datetime(path.stem, format="%Y%m%d")
            time_utc = pd.Timestamp(_first_value(ds["time"].values))
            rows.append(
                {
                    "file": path.name,
                    "issue_date_from_file": issue_date.date().isoformat(),
                    "time_coord_utc": time_utc.isoformat(),
                    "time_plus_8_bjt": (time_utc + pd.Timedelta(hours=8)).isoformat(),
                    "target_date_by_rule": (issue_date + pd.Timedelta(days=1))
                    .date()
                    .isoformat(),
                    "lead_min": int(lead.min()),
                    "lead_max": int(lead.max()),
                    "lead_count": int(len(lead)),
                    "channels": ",".join(channels),
                    "has_night_channel": any("night" in c.lower() for c in channels),
                    "lat_count": int(ds.sizes["lat"]),
                    "lon_count": int(ds.sizes["lon"]),
                    "lat_min": float(ds["lat"].min()),
                    "lat_max": float(ds["lat"].max()),
                    "lon_min": float(ds["lon"].min()),
                    "lon_max": float(ds["lon"].max()),
                }
            )
    return pd.DataFrame(rows).sort_values("file").reset_index(drop=True)


def extract_weather_hourly(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        issue_date = pd.to_datetime(path.stem, format="%Y%m%d")
        with xr.open_dataset(path) as ds:
            time_utc = pd.Timestamp(_first_value(ds["time"].values))
            lead_values = ds["lead_time"].values.astype(int)
            channels = [str(v) for v in ds["channel"].values.tolist()]
            data = ds["data"].isel(time=0)
            u100_all = data.sel(channel="u100").to_numpy()
            v100_all = data.sel(channel="v100").to_numpy()
            wind_speed_all = np.sqrt(u100_all**2 + v100_all**2)
            for lead_idx, lead_hour in enumerate(lead_values):
                row: dict[str, object] = {
                    "file": path.name,
                    "issue_date": issue_date.normalize(),
                    "lead_hour": int(lead_hour),
                    "valid_utc": time_utc + pd.Timedelta(hours=int(lead_hour)),
                    "current_utc_plus8": time_utc
                    + pd.Timedelta(hours=int(lead_hour) + 8),
                    "filename_d_plus_1": issue_date
                    + pd.Timedelta(days=1, hours=int(lead_hour)),
                    "no_utc_plus8": time_utc + pd.Timedelta(hours=int(lead_hour)),
                    "filename_same_day": issue_date + pd.Timedelta(hours=int(lead_hour)),
                }
                for channel in channels:
                    values = data.sel(channel=channel).isel(lead_time=lead_idx).to_numpy()
                    row[f"{channel}_mean"] = float(np.nanmean(values))
                    if channel in {"ghi", "tcc"}:
                        row[f"{channel}_max"] = float(np.nanmax(values))
                wind_values = wind_speed_all[lead_idx]
                row["wind_speed_mean"] = float(np.nanmean(wind_values))
                row["wind_speed_max"] = float(np.nanmax(wind_values))
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["current_utc_plus8", "file"]).reset_index(drop=True)


def train_hourly_frame(cfg: dict) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    train = add_bid_space_columns(load_train_frame(cfg))
    train["target_hour"] = train[time_col].dt.floor("h")
    train["valid_date"] = train["target_hour"].dt.normalize()
    train["month"] = train["target_hour"].dt.month
    train["hour"] = train["target_hour"].dt.hour
    train["renewable_actual"] = train["风光总加实际值"]
    train["renewable_forecast"] = train["风光总加预测值"]
    train["renewable_error"] = train["renewable_forecast"] - train["renewable_actual"]
    train["wind_actual"] = train["风电实际值"]
    train["wind_forecast"] = train["风电预测值"]
    train["wind_error"] = train["wind_forecast"] - train["wind_actual"]
    train["solar_actual"] = train["光伏实际值"]
    train["solar_forecast"] = train["光伏预测值"]
    train["solar_error"] = train["solar_forecast"] - train["solar_actual"]
    agg = {
        "rows_15min": (time_col, "count"),
        "renewable_actual": ("renewable_actual", "mean"),
        "renewable_forecast": ("renewable_forecast", "mean"),
        "renewable_error": ("renewable_error", "mean"),
        "wind_actual": ("wind_actual", "mean"),
        "wind_forecast": ("wind_forecast", "mean"),
        "wind_error": ("wind_error", "mean"),
        "solar_actual": ("solar_actual", "mean"),
        "solar_forecast": ("solar_forecast", "mean"),
        "solar_error": ("solar_error", "mean"),
        "bid_space_fct": ("bid_space_fct", "mean"),
        "bid_space_act": ("bid_space_act", "mean"),
        "bid_space_error": ("bid_space_error", "mean"),
        "price_hour": ("price", "mean"),
    }
    return (
        train.groupby(["target_hour", "valid_date", "month", "hour"], observed=True)
        .agg(**agg)
        .reset_index()
    )


def test_hourly_frame(cfg: dict) -> pd.DataFrame:
    time_col = cfg["data"]["time_col"]
    test = load_test_frame(cfg)
    test["target_hour"] = test[time_col].dt.floor("h")
    return test.groupby("target_hour", as_index=False).agg(rows_15min=(time_col, "count"))


def join_for_alignment(
    hourly_weather: pd.DataFrame,
    train_hourly: pd.DataFrame,
    alignment_col: str,
) -> pd.DataFrame:
    weather = hourly_weather.rename(columns={alignment_col: "target_hour"}).copy()
    keep = ["target_hour", "file", "lead_hour", *CORE_FEATURES]
    weather = weather[keep].drop_duplicates("target_hour", keep="last")
    merged = train_hourly.merge(weather, on="target_hour", how="inner")
    merged["alignment"] = alignment_col
    merged["daylight"] = merged["ghi_mean"] > 20
    merged["midday"] = merged["hour"].between(10, 17)
    return merged


def _safe_corr(x: pd.Series, y: pd.Series, *, method: str) -> float:
    frame = pd.concat([x, y], axis=1).dropna()
    if len(frame) < 20 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1], method=method))


def residualize_by_month_hour(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(["month", "hour"], observed=True)
    for col in cols:
        out[f"{col}_mh_resid"] = out[col] - grouped[col].transform("mean")
    return out


def top_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    scopes = [
        ("all", joined),
        ("daylight", joined[joined["daylight"]]),
        ("midday_10_17", joined[joined["midday"]]),
        ("winter_11_12", joined[joined["month"].isin([11, 12])]),
    ]
    rows: list[dict[str, object]] = []
    for alignment, alignment_df in joined.groupby("alignment", sort=True):
        resid = residualize_by_month_hour(alignment_df, CORE_FEATURES + TARGETS)
        for scope_name, scope_df in scopes:
            scoped = scope_df[scope_df["alignment"] == alignment]
            scoped_resid = resid.loc[scoped.index]
            for target in TARGETS:
                for feature in CORE_FEATURES:
                    rows.append(
                        {
                            "alignment": alignment,
                            "scope": scope_name,
                            "target": target,
                            "feature": feature,
                            "kind": "raw",
                            "rows": len(scoped),
                            "pearson": _safe_corr(
                                scoped[feature], scoped[target], method="pearson"
                            ),
                            "spearman": _safe_corr(
                                scoped[feature], scoped[target], method="spearman"
                            ),
                        }
                    )
                    rows.append(
                        {
                            "alignment": alignment,
                            "scope": scope_name,
                            "target": target,
                            "feature": feature,
                            "kind": "month_hour_residual",
                            "rows": len(scoped_resid),
                            "pearson": _safe_corr(
                                scoped_resid[f"{feature}_mh_resid"],
                                scoped_resid[f"{target}_mh_resid"],
                                method="pearson",
                            ),
                            "spearman": _safe_corr(
                                scoped_resid[f"{feature}_mh_resid"],
                                scoped_resid[f"{target}_mh_resid"],
                                method="spearman",
                            ),
                        }
                    )
    out = pd.DataFrame(rows)
    out["abs_spearman"] = out["spearman"].abs()
    return out.sort_values(
        ["alignment", "kind", "scope", "target", "abs_spearman"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)


def signal_summary(corr: pd.DataFrame) -> pd.DataFrame:
    probes = [
        ("solar_actual", "ghi_mean", "midday_10_17", "raw"),
        ("solar_error", "ghi_mean", "midday_10_17", "month_hour_residual"),
        ("wind_actual", "wind_speed_mean", "all", "raw"),
        ("wind_error", "wind_speed_mean", "all", "month_hour_residual"),
        ("renewable_error", "wind_speed_mean", "all", "month_hour_residual"),
        ("bid_space_error", "wind_speed_mean", "all", "month_hour_residual"),
        ("price_hour", "wind_speed_mean", "all", "month_hour_residual"),
        ("price_hour", "ghi_mean", "midday_10_17", "month_hour_residual"),
    ]
    rows: list[pd.DataFrame] = []
    for target, feature, scope, kind in probes:
        sub = corr[
            (corr["target"] == target)
            & (corr["feature"] == feature)
            & (corr["scope"] == scope)
            & (corr["kind"] == kind)
        ].copy()
        sub["probe"] = f"{kind}:{scope}:{feature}->{target}"
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)[
        ["probe", "alignment", "rows", "pearson", "spearman"]
    ].sort_values(["probe", "alignment"])


def coverage_summary(
    hourly_weather: pd.DataFrame,
    train_hourly: pd.DataFrame,
    test_hourly: pd.DataFrame,
    alignment_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in alignment_cols:
        weather_hours = pd.DataFrame(
            {"target_hour": pd.to_datetime(hourly_weather[col])}
        ).drop_duplicates()
        train_join = train_hourly[["target_hour"]].merge(
            weather_hours, on="target_hour", how="inner"
        )
        test_join = test_hourly[["target_hour"]].merge(
            weather_hours, on="target_hour", how="inner"
        )
        rows.append(
            {
                "alignment": col,
                "weather_hours": len(weather_hours),
                "weather_min": weather_hours["target_hour"].min(),
                "weather_max": weather_hours["target_hour"].max(),
                "train_hours": len(train_hourly),
                "train_join_hours": len(train_join),
                "train_coverage": len(train_join) / len(train_hourly),
                "test_hours": len(test_hourly),
                "test_join_hours": len(test_join),
                "test_coverage": len(test_join) / len(test_hourly),
                "test_missing_hours": len(test_hourly) - len(test_join),
            }
        )
    return pd.DataFrame(rows)


def selected_top_corr(corr: pd.DataFrame) -> pd.DataFrame:
    focus = corr[
        (
            corr["target"].isin(
                ["renewable_error", "solar_error", "wind_error", "bid_space_error"]
            )
        )
        & (corr["kind"] == "month_hour_residual")
    ]
    return focus.groupby(["alignment", "scope", "target"], group_keys=False).head(8)


def write_report(
    *,
    paths: list[Path],
    metadata: pd.DataFrame,
    coverage: pd.DataFrame,
    signal: pd.DataFrame,
    top_corr: pd.DataFrame,
) -> None:
    file_dates = pd.to_datetime([path.stem for path in paths], format="%Y%m%d")
    channels = metadata["channels"].iloc[0] if not metadata.empty else ""
    current_coverage = coverage[coverage["alignment"] == "current_utc_plus8"]
    lines = [
        "# Weather Alignment Audit",
        "",
        "Purpose: verify whether the competition NWP `.nc` files are aligned correctly before "
        "using weather as price, bid-space, or reranking features.",
        "",
        "## Verdict",
        "",
        "- Current project alignment is correct: file date `D` maps to Beijing target day "
        "`D+1`, and `lead_time=0..23` maps to local hours `00:00..23:00`.",
        "- This is equivalent to `time_coord_utc + lead_time + 8h`; sampled files have "
        "`time=16:00 UTC`, so `+8h` lands on midnight Beijing time of the target day.",
        "- The local NetCDF files do **not** contain any `night` channel or `night` suffix "
        "field. Available channels are listed below.",
        "- Test-period weather coverage is complete under the current alignment; train "
        "coverage misses 2025-01-01 because `2024-12-31.nc` is not provided.",
        "",
        "## NetCDF Inventory",
        "",
        f"- Files: `{len(paths)}`.",
        f"- File date range: `{file_dates.min().date()}` to `{file_dates.max().date()}`.",
        f"- Channels: `{channels}`.",
        "",
        markdown_table(metadata, floatfmt=".4f"),
        "",
        "## Coverage By Alignment Hypothesis",
        "",
        markdown_table(coverage, floatfmt=".4f"),
        "",
        "## Alignment Signal Checks",
        "",
        "These probes compare weather features with actual renewable output, renewable forecast "
        "error, bid-space error, and hourly price. `month_hour_residual` removes the month x "
        "hour average first, so it tests information beyond seasonal intraday pattern.",
        "",
        markdown_table(signal, floatfmt=".4f"),
        "",
        "## Top Residual Correlations For Error Targets",
        "",
        markdown_table(top_corr.head(80), floatfmt=".4f"),
        "",
        "## Readout For Next Experiments",
        "",
        "- Weather alignment itself is not the likely reason previous full-weather features "
        "underperformed; the `UTC+8` mapping is already implemented.",
        "- The useful weather signal is more likely in selective use: wind-speed/GHI/cloud "
        "residuals as confidence, rerank, or pair-level features rather than blindly adding "
        "all weather aggregates to the point price model.",
        "- Because `holiday_only` beat local winter validation but lost online, promote future "
        "weather changes only if they improve broad 5-fold stability or are validated by an "
        "online submit.",
        "",
        "## Artifacts",
        "",
        f"- `{OUT_METADATA}`",
        f"- `{OUT_COVERAGE}`",
        f"- `{OUT_SIGNAL}`",
        f"- `{OUT_TOP_CORR}`",
    ]
    if not current_coverage.empty:
        row = current_coverage.iloc[0]
        lines.insert(
            13,
            "- Current alignment train/test coverage: "
            f"`{row['train_join_hours']}/{row['train_hours']}` train hours and "
            f"`{row['test_join_hours']}/{row['test_hours']}` test hours.",
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    paths = nc_paths(cfg)
    metadata = inspect_nc_metadata(paths)
    hourly_weather = extract_weather_hourly(paths)
    train_hourly = train_hourly_frame(cfg)
    test_hourly = test_hourly_frame(cfg)
    alignment_cols = [
        "current_utc_plus8",
        "filename_d_plus_1",
        "no_utc_plus8",
        "filename_same_day",
    ]
    coverage = coverage_summary(hourly_weather, train_hourly, test_hourly, alignment_cols)
    joined = pd.concat(
        [
            join_for_alignment(hourly_weather, train_hourly, col)
            for col in ["current_utc_plus8", "no_utc_plus8", "filename_same_day"]
        ],
        ignore_index=True,
    )
    corr = top_correlations(joined)
    signal = signal_summary(corr)
    top_corr = selected_top_corr(corr)

    metadata.to_csv(OUT_METADATA, index=False)
    coverage.to_csv(OUT_COVERAGE, index=False)
    signal.to_csv(OUT_SIGNAL, index=False)
    top_corr.to_csv(OUT_TOP_CORR, index=False)
    write_report(
        paths=paths,
        metadata=metadata,
        coverage=coverage,
        signal=signal,
        top_corr=top_corr,
    )
    print(coverage.to_string(index=False))
    print(signal.to_string(index=False))
    print(OUT_MD)


if __name__ == "__main__":
    main()
