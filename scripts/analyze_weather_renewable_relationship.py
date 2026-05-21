from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from electricity.features.bid_space import markdown_table
from electricity.features.weather import (
    CHANNEL_UNITS,
    CORE_WEATHER_FEATURES,
    merge_weather_renewable,
)

CONFIG_PATH = Path("configs/base.yaml")
REPORTS_DIR = Path("reports")
OUT_MD = REPORTS_DIR / "weather_renewable_relationship.md"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_corr(x: pd.Series, y: pd.Series, *, method: str) -> float:
    frame = pd.concat([x, y], axis=1).dropna()
    if len(frame) < 10 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1], method=method))


def residualize_by_month_hour(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(["month", "hour"], observed=True)
    for col in cols:
        out[f"{col}_mh_resid"] = out[col] - grouped[col].transform("mean")
    return out


def corr_table(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]) -> pd.DataFrame:
    scopes = [
        ("all", df),
        ("daylight", df[df["daylight"]]),
        ("midday_10_17", df[df["midday"]]),
        ("winter_11_12", df[df["month"].isin([11, 12])]),
        ("spring_02_05", df[df["month"].isin([2, 3, 4, 5])]),
    ]
    rows: list[dict[str, object]] = []
    for scope, group in scopes:
        for target in target_cols:
            for feature in feature_cols:
                rows.append(
                    {
                        "scope": scope,
                        "target": target,
                        "feature": feature,
                        "rows": len(group),
                        "pearson": _safe_corr(group[feature], group[target], method="pearson"),
                        "spearman": _safe_corr(
                            group[feature], group[target], method="spearman"
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    out["abs_spearman"] = out["spearman"].abs()
    return out.sort_values(["scope", "target", "abs_spearman"], ascending=[True, True, False])


def month_segment_error_weather(df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = {
        "hours": ("target_hour", "count"),
        "renewable_error_mean": ("renewable_error", "mean"),
        "renewable_abs_error_mean": ("renewable_abs_error", "mean"),
        "solar_error_mean": ("solar_error", "mean"),
        "wind_error_mean": ("wind_error", "mean"),
    }
    for col in CORE_WEATHER_FEATURES:
        agg_cols[col] = (col, "mean")
    return df.groupby(["month", "segment"], observed=True).agg(**agg_cols).reset_index()


def write_report(
    *,
    merged: pd.DataFrame,
    raw_corr: pd.DataFrame,
    resid_corr: pd.DataFrame,
    month_segment: pd.DataFrame,
) -> None:
    top_error = raw_corr[raw_corr["target"] == "renewable_error"].groupby(
        "scope", group_keys=False
    ).head(12)
    top_error_resid = resid_corr[resid_corr["target"] == "renewable_error_mh_resid"].groupby(
        "scope", group_keys=False
    ).head(12)
    top_actual = raw_corr[raw_corr["target"] == "renewable_actual"].groupby(
        "scope", group_keys=False
    ).head(10)
    midday_bias = month_segment[month_segment["segment"].isin(["10_14", "14_18"])]
    midday_bias = midday_bias.sort_values("renewable_error_mean", ascending=False).head(18)

    valid_dates = merged["valid_date"].dt.date
    lines = [
        "# Weather vs Renewable Diagnostics",
        "",
        "This report checks whether official NWP forecasts can explain renewable forecast "
        "value and forecast error.",
        "",
        "## Time Alignment",
        "",
        "- Competition document says each `.nc` file is published on date `D` and forecasts "
        "date `D+1`.",
        "- The `.nc` coordinate `time` is UTC. Example: `20250101.nc` has "
        "`time=2025-01-01 16:00 UTC`, which equals `2025-01-02 00:00` Beijing time.",
        "- `lead_time=0..23` maps to Beijing hours `00:00..23:00` of the target day.",
        "- Boundary data is 15-minute Beijing time. For this diagnostic, 15-minute renewable "
        "values are averaged to hourly values before joining NWP.",
        "",
        "## Units",
        "",
        markdown_table(
            pd.DataFrame(
                [{"variable": k, "unit": v} for k, v in CHANNEL_UNITS.items()]
            ),
            floatfmt=".4f",
        ),
        "",
        "## Join Coverage",
        "",
        f"- Joined hourly rows: `{len(merged)}`.",
        f"- Date range: `{valid_dates.min()}` to `{valid_dates.max()}`.",
        f"- Unique days: `{merged['valid_date'].nunique()}`.",
        "- 2025-01-01 is expected to be absent because the matching NWP file would be "
        "2024-12-31, which is not provided locally.",
        "",
        "## Top Weather Features For Renewable Forecast Error",
        "",
        "Target is `renewable_error = renewable_forecast - renewable_actual`.",
        "Positive correlation means larger weather feature values tend to come with more "
        "over-forecasting of wind+solar output.",
        "",
        markdown_table(top_error, floatfmt=".4f"),
        "",
        "## Top Weather Features After Removing Month-Hour Mean",
        "",
        "This is stricter: both weather features and renewable error are residualized by "
        "month x hour. It tests whether weather adds signal beyond seasonal intraday pattern.",
        "",
        markdown_table(top_error_resid, floatfmt=".4f"),
        "",
        "## Top Weather Features For Renewable Actual Output",
        "",
        markdown_table(top_actual, floatfmt=".4f"),
        "",
        "## Midday Month-Segment Bias With Weather Means",
        "",
        markdown_table(midday_bias, floatfmt=".4f"),
        "",
        "## Readout",
        "",
        "- Strong raw correlations can be mostly time-of-day seasonality; residual correlations "
        "are more useful for correction.",
        "- If residual weather-error correlation is weak, NWP probably does not add much beyond "
        "the provided renewable forecast.",
        "- If GHI/TCC/wind residual correlations are stable in specific months/segments, use them "
        "as candidates for renewable-error correction or bid-space reliability features.",
        "",
        "## Artifacts",
        "",
        "- `reports/weather_hourly_spatial_features.csv`",
        "- `reports/weather_renewable_joined_hourly.csv`",
        "- `reports/weather_renewable_corr.csv`",
        "- `reports/weather_renewable_residual_corr.csv`",
        "- `reports/weather_renewable_month_segment.csv`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    merged = merge_weather_renewable(cfg)
    feature_cols = [col for col in CORE_WEATHER_FEATURES if col in merged.columns]
    target_cols = [
        "renewable_actual",
        "renewable_forecast",
        "renewable_error",
        "solar_actual",
        "solar_error",
        "wind_actual",
        "wind_error",
    ]
    raw_corr = corr_table(merged, feature_cols, target_cols)

    resid_cols = feature_cols + target_cols
    resid = residualize_by_month_hour(merged, resid_cols)
    resid_feature_cols = [f"{col}_mh_resid" for col in feature_cols]
    resid_target_cols = [f"{col}_mh_resid" for col in target_cols]
    resid_corr = corr_table(resid, resid_feature_cols, resid_target_cols)

    month_segment = month_segment_error_weather(merged)
    merged.to_csv(REPORTS_DIR / "weather_renewable_joined_hourly.csv", index=False)
    raw_corr.to_csv(REPORTS_DIR / "weather_renewable_corr.csv", index=False)
    resid_corr.to_csv(REPORTS_DIR / "weather_renewable_residual_corr.csv", index=False)
    month_segment.to_csv(REPORTS_DIR / "weather_renewable_month_segment.csv", index=False)
    write_report(
        merged=merged,
        raw_corr=raw_corr,
        resid_corr=resid_corr,
        month_segment=month_segment,
    )
    print(raw_corr[raw_corr["target"] == "renewable_error"].head(30).to_string(index=False))
    print(OUT_MD)


if __name__ == "__main__":
    main()
