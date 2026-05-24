from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_sequence_residual import evaluate_predictions, load_config
from electricity.data import load_train_frame


REPORTS_DIR = Path("reports")
DEFAULT_CONFIG = Path("configs/sequence_mlp_residual_5fold.yaml")
DEFAULT_PREDICTIONS = REPORTS_DIR / "predictions_sequence_mlp_residual_5fold.csv"
DEFAULT_TEST_LIKE = REPORTS_DIR / "online_test_like_day_weights.csv"


def markdown_table(df: pd.DataFrame, *, limit: int = 50, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_无数据_"
    block = df.head(limit).copy()
    cols = list(block.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in block.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float | np.floating):
                cells.append("" if pd.isna(value) else format(float(value), floatfmt))
            else:
                cells.append("" if pd.isna(value) else str(value).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_gate_frame(predictions: pd.DataFrame, *, test_like_path: Path) -> pd.DataFrame:
    out = predictions.copy()
    out["times"] = pd.to_datetime(out["times"])
    out["date"] = out["times"].dt.normalize()
    out["month"] = out["times"].dt.month
    out["hour"] = out["times"].dt.hour
    out["slot"] = out["times"].dt.hour * 4 + out["times"].dt.minute // 15

    if test_like_path.exists():
        weights = pd.read_csv(test_like_path, parse_dates=["date"])
        weights["date"] = weights["date"].dt.normalize()
        out = out.merge(
            weights[["date", "test_like_distance", "test_like_weight_raw"]],
            on="date",
            how="left",
        )
    else:
        out["test_like_distance"] = np.nan
        out["test_like_weight_raw"] = np.nan
    return out


def complete_day_filter(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["fold", "date"])["slot"].agg(["count", "min", "max"]).reset_index()
    complete = counts[(counts["count"] == 96) & (counts["min"] == 0) & (counts["max"] == 95)]
    return df.merge(complete[["fold", "date"]], on=["fold", "date"], how="inner")


def gate_variants(df: pd.DataFrame, weights: list[float]) -> list[dict[str, object]]:
    day_info = (
        df.groupby(["fold", "date"], as_index=False)
        .agg(
            month=("month", "first"),
            test_like_distance=("test_like_distance", "first"),
            test_like_weight_raw=("test_like_weight_raw", "first"),
        )
    )
    q25 = float(day_info["test_like_distance"].quantile(0.25))
    q50 = float(day_info["test_like_distance"].quantile(0.50))
    q75 = float(day_info["test_like_distance"].quantile(0.75))

    variants: list[dict[str, object]] = []
    for weight in weights:
        variants += [
            {
                "variant": f"global_w{weight:g}",
                "weight": weight,
                "gate_type": "global",
                "description": "所有日期都启用残差",
                "gate": lambda d, q=q50: np.ones(len(d), dtype=bool),
            },
            {
                "variant": f"jan_feb_only_w{weight:g}",
                "weight": weight,
                "gate_type": "month",
                "description": "只在 Jan-Feb-like 验证月启用",
                "gate": lambda d, q=q50: d["month"].isin([1, 2]).to_numpy(),
            },
            {
                "variant": f"late_winter_w{weight:g}",
                "weight": weight,
                "gate_type": "month",
                "description": "只在 12/1/2 月启用",
                "gate": lambda d, q=q50: d["month"].isin([12, 1, 2]).to_numpy(),
            },
            {
                "variant": f"winter_w{weight:g}",
                "weight": weight,
                "gate_type": "month",
                "description": "只在 11/12/1/2 月启用",
                "gate": lambda d, q=q50: d["month"].isin([11, 12, 1, 2]).to_numpy(),
            },
            {
                "variant": f"testlike_q25_w{weight:g}",
                "weight": weight,
                "gate_type": "test_like_distance",
                "description": f"只在最像测试集的 25% 历史日启用，阈值={q25:.4f}",
                "gate": lambda d, q=q25: d["test_like_distance"].to_numpy(dtype=float) <= q,
            },
            {
                "variant": f"testlike_q50_w{weight:g}",
                "weight": weight,
                "gate_type": "test_like_distance",
                "description": f"只在最像测试集的 50% 历史日启用，阈值={q50:.4f}",
                "gate": lambda d, q=q50: d["test_like_distance"].to_numpy(dtype=float) <= q,
            },
            {
                "variant": f"testlike_q75_w{weight:g}",
                "weight": weight,
                "gate_type": "test_like_distance",
                "description": f"只在最像测试集的 75% 历史日启用，阈值={q75:.4f}",
                "gate": lambda d, q=q75: d["test_like_distance"].to_numpy(dtype=float) <= q,
            },
            {
                "variant": f"late_winter_testlike_q50_w{weight:g}",
                "weight": weight,
                "gate_type": "month+test_like_distance",
                "description": f"只在 12/1/2 月且测试相似度前 50% 启用，阈值={q50:.4f}",
                "gate": lambda d, q=q50: (
                    d["month"].isin([12, 1, 2]).to_numpy()
                    & (d["test_like_distance"].to_numpy(dtype=float) <= q)
                ),
            },
        ]
    return variants


def summarize_overall(
    by_fold: pd.DataFrame,
    daily: pd.DataFrame,
    anchor_by_fold: pd.DataFrame,
    anchor_daily: pd.DataFrame,
) -> pd.DataFrame:
    anchor = anchor_by_fold.set_index("fold")
    anchor_p90_regret = float(anchor_daily["regret"].quantile(0.9))
    anchor_worst_day_profit = float(anchor_daily["profit"].min())
    rows = []
    for variant, group in by_fold.groupby("variant", sort=False):
        weighted = daily[daily["variant"] == variant].copy()
        p90_regret = float(weighted["regret"].quantile(0.9))
        worst_day_profit = float(weighted["profit"].min())
        rows.append(
            {
                "variant": variant,
                "gate_type": group["gate_type"].iloc[0],
                "blend_weight": group["blend_weight"].iloc[0],
                "enabled_days": int(group["enabled_days"].sum()),
                "mean_profit": float(group["avg_profit"].mean()),
                "min_fold_profit": float(group["avg_profit"].min()),
                "loss_days": int(group["loss_days"].sum()),
                "mean_regret": float(group["avg_regret"].mean()),
                "mean_charge_gap": float(group["mean_abs_charge_gap"].mean()),
                "mean_discharge_gap": float(group["mean_abs_discharge_gap"].mean()),
                "profit_delta_vs_anchor": float(
                    (group.set_index("fold")["avg_profit"] - anchor["avg_profit"]).mean()
                ),
                "loss_day_delta_vs_anchor": int(
                    (group.set_index("fold")["loss_days"] - anchor["loss_days"]).sum()
                ),
                "regret_delta_vs_anchor": float(
                    (group.set_index("fold")["avg_regret"] - anchor["avg_regret"]).mean()
                ),
                "p90_regret": p90_regret,
                "p90_regret_delta_vs_anchor": p90_regret - anchor_p90_regret,
                "worst_day_profit": worst_day_profit,
                "worst_day_delta_vs_anchor": worst_day_profit - anchor_worst_day_profit,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["profit_delta_vs_anchor", "loss_day_delta_vs_anchor", "p90_regret"],
        ascending=[False, True, True],
    )


def summarize_regimes(daily: pd.DataFrame, anchor_daily: pd.DataFrame) -> pd.DataFrame:
    regimes = {
        "standard_09_12": {"valid_2025_09", "valid_2025_10", "valid_2025_11", "valid_2025_12"},
        "jan_feb_like": {"valid_2025_jan_feb"},
        "winter_11_12_jan_feb": {"valid_2025_11", "valid_2025_12", "valid_2025_jan_feb"},
        "late_winter_12_jan_feb": {"valid_2025_12", "valid_2025_jan_feb"},
        "all_5fold": None,
    }
    rows = []
    for variant, group in daily.groupby("variant", sort=False):
        anchor_group = anchor_daily.copy()
        for regime, folds in regimes.items():
            sub = group if folds is None else group[group["fold"].isin(folds)]
            base = anchor_group if folds is None else anchor_group[anchor_group["fold"].isin(folds)]
            if sub.empty:
                continue
            rows.append(
                {
                    "variant": variant,
                    "regime": regime,
                    "days": int(len(sub)),
                    "enabled_days": int(sub["residual_enabled"].sum()),
                    "mean_profit": float(sub["profit"].mean()),
                    "profit_delta_vs_anchor": float(sub["profit"].mean() - base["profit"].mean()),
                    "loss_days": int((sub["profit"] < -1e-9).sum()),
                    "loss_day_delta_vs_anchor": int(
                        (sub["profit"] < -1e-9).sum() - (base["profit"] < -1e-9).sum()
                    ),
                    "p90_regret": float(sub["regret"].quantile(0.9)),
                    "worst_day_profit": float(sub["profit"].min()),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    *,
    overall: pd.DataFrame,
    by_regime: pd.DataFrame,
    descriptions: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    safe = overall[
        (overall["profit_delta_vs_anchor"] > 0)
        & (overall["loss_day_delta_vs_anchor"] <= 0)
        & (overall["p90_regret_delta_vs_anchor"] <= 0)
        & (overall["worst_day_delta_vs_anchor"] >= 0)
    ].copy()
    lines = [
        "# Gated Sequence Residual Experiment",
        "",
        "## Scope",
        "",
        "- 复用 `sequence_mlp_residual_5fold` 已保存的 OOF 残差预测，不重新训练。",
        "- 目标不是替代 champion，而是验证残差信号能否只在低风险日期局部启用。",
        "- 所有候选仍使用 champion dispatch prior，与当前线上冠军保持一致。",
        "- 这是诊断实验；month/test-like gate 是规则假设，不应只凭 Jan-Feb-like 单点结果提交。",
        "",
        "## Overall Ranking",
        "",
        markdown_table(overall, limit=40),
        "",
        "## Regime Breakdown",
        "",
        markdown_table(
            by_regime.sort_values(["variant", "regime"]).head(120),
            limit=120,
        ),
        "",
        "## Gate Definitions",
        "",
        markdown_table(descriptions),
        "",
        "## Decision",
        "",
    ]
    if safe.empty:
        lines += [
            "- 没有候选同时满足：全 5 fold 正收益、不增加亏损日、p90 regret 不恶化、最差日不恶化。",
            "- 结论：MLP 残差目前仍只能作为诊断信号，不进入 submit 主线。",
        ]
    else:
        lines += [
            "- 存在局部正信号，但仍需用 online-risk dashboard 再做测试相似度/已知线上差异校验。",
            "- 若提升不足 +100/day，不生成提交文件。",
        ]
        lines += ["", markdown_table(safe.head(10))]
    lines += [
        "",
        "## Artifacts",
        "",
        *[f"- `{path}`" for path in paths.values()],
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--test-like", default=str(DEFAULT_TEST_LIKE))
    parser.add_argument("--weights", default="0.05,0.10,0.15,0.20")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    time_col = cfg["data"]["time_col"]
    target_col = cfg["data"]["target_col"]
    train_labels = load_train_frame(cfg)[[time_col, target_col]].copy()

    predictions = pd.read_csv(
        args.predictions,
        parse_dates=[time_col, "train_start", "train_end", "valid_start", "valid_end"],
    )
    predictions = build_gate_frame(predictions, test_like_path=Path(args.test_like))
    predictions = complete_day_filter(predictions)
    weights = [float(x) for x in args.weights.split(",") if x.strip()]

    variants = gate_variants(predictions, weights)
    descriptions = pd.DataFrame(
        [
            {
                "variant": v["variant"],
                "gate_type": v["gate_type"],
                "blend_weight": v["weight"],
                "description": v["description"],
            }
            for v in variants
        ]
    )

    rows = []
    daily_rows = []
    for variant in variants:
        name = str(variant["variant"])
        weight = float(variant["weight"])
        work = predictions.copy()
        gate = variant["gate"](work)
        work["residual_enabled"] = gate.astype(int)
        work["pred_gated"] = work["anchor_pred"]
        work.loc[gate, "pred_gated"] = (
            work.loc[gate, "anchor_pred"] + weight * work.loc[gate, "residual_pred"]
        )
        for fold, subset in work.groupby("fold", sort=False):
            first = subset.iloc[0]
            summary, day_rows = evaluate_predictions(
                subset,
                time_col=time_col,
                target_col=target_col,
                pred_col="pred_gated",
                cfg=cfg,
                train_labels=train_labels,
                fold=str(fold),
                train_start=first["train_start"],
                train_end=first["train_end"],
                valid_start=first["valid_start"],
                valid_end=first["valid_end"],
            )
            summary["variant"] = name
            summary["gate_type"] = variant["gate_type"]
            summary["blend_weight"] = weight
            summary["enabled_days"] = int(
                subset.groupby("date")["residual_enabled"].max().sum()
            )
            rows.append(summary)
            enabled_by_day = subset.groupby("date")["residual_enabled"].max().to_dict()
            for row in day_rows:
                row["variant"] = name
                row["gate_type"] = variant["gate_type"]
                row["blend_weight"] = weight
                row["residual_enabled"] = int(enabled_by_day[pd.Timestamp(row["date"])])
                daily_rows.append(row)

    by_fold = pd.DataFrame(rows)
    daily = pd.DataFrame(daily_rows)

    # Recompute the exact anchor from the same prediction file for stable deltas.
    anchor_rows = []
    anchor_daily_rows = []
    anchor = predictions.copy()
    anchor["pred_anchor"] = anchor["anchor_pred"]
    for fold, subset in anchor.groupby("fold", sort=False):
        first = subset.iloc[0]
        summary, day_rows = evaluate_predictions(
            subset,
            time_col=time_col,
            target_col=target_col,
            pred_col="pred_anchor",
            cfg=cfg,
            train_labels=train_labels,
            fold=str(fold),
            train_start=first["train_start"],
            train_end=first["train_end"],
            valid_start=first["valid_start"],
            valid_end=first["valid_end"],
        )
        anchor_rows.append(summary)
        anchor_daily_rows.extend(day_rows)
    anchor_by_fold = pd.DataFrame(anchor_rows)
    anchor_daily = pd.DataFrame(anchor_daily_rows)

    overall = summarize_overall(by_fold, daily, anchor_by_fold, anchor_daily)
    by_regime = summarize_regimes(daily, anchor_daily)

    REPORTS_DIR.mkdir(exist_ok=True)
    paths = {
        "overall": REPORTS_DIR / "sequence_mlp_residual_gated_overall.csv",
        "by_fold": REPORTS_DIR / "sequence_mlp_residual_gated_by_fold.csv",
        "by_regime": REPORTS_DIR / "sequence_mlp_residual_gated_by_regime.csv",
        "daily": REPORTS_DIR / "sequence_mlp_residual_gated_daily.csv",
        "gate_definitions": REPORTS_DIR / "sequence_mlp_residual_gated_definitions.csv",
        "report": REPORTS_DIR / "sequence_mlp_residual_gated_experiment.md",
    }
    overall.to_csv(paths["overall"], index=False)
    by_fold.to_csv(paths["by_fold"], index=False)
    by_regime.to_csv(paths["by_regime"], index=False)
    daily.to_csv(paths["daily"], index=False)
    descriptions.to_csv(paths["gate_definitions"], index=False)
    write_report(overall=overall, by_regime=by_regime, descriptions=descriptions, paths=paths)
    print(overall.head(20).to_string(index=False))
    print(f"report_path={paths['report']}")


if __name__ == "__main__":
    main()
