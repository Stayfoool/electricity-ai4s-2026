from __future__ import annotations

from dataclasses import dataclass

from analyze_strategy_gate import (
    BASE_GATE_FEATURES,
    CHAMPION,
    REPORTS_DIR,
    build_gate_dataset,
    load_config,
    profit_col,
)
import numpy as np
import pandas as pd

CHAMPION_PROFIT = profit_col(CHAMPION)

NO_TRADE_FEATURES = [
    *BASE_GATE_FEATURES,
    f"{CHAMPION}__predicted_spread",
    f"{CHAMPION}__top2_spread",
    f"{CHAMPION}__top5_spread_mean",
    f"{CHAMPION}__top5_spread_std",
    f"{CHAMPION}__top1_top2_gap",
    f"{CHAMPION}__top1_top5_mean_gap",
]


@dataclass(frozen=True)
class NoTradeRule:
    kind: str
    feature_a: str
    op_a: str
    threshold_a: float
    feature_b: str | None
    op_b: str | None
    threshold_b: float | None
    train_mean_profit: float
    train_lift_vs_champion: float
    train_no_trade_days: int


def _mask(df: pd.DataFrame, feature: str, op: str, threshold: float) -> pd.Series:
    if op == "<=":
        return df[feature] <= threshold
    if op == ">":
        return df[feature] > threshold
    raise ValueError(f"unsupported op={op}")


def _apply_rule(df: pd.DataFrame, rule: NoTradeRule) -> pd.Series:
    mask = _mask(df, rule.feature_a, rule.op_a, rule.threshold_a)
    if rule.feature_b is not None and rule.op_b is not None and rule.threshold_b is not None:
        mask = mask & _mask(df, rule.feature_b, rule.op_b, rule.threshold_b)
    return mask


def _profit_with_no_trade(df: pd.DataFrame, no_trade_mask: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(no_trade_mask, 0.0, df[CHAMPION_PROFIT].to_numpy(dtype=float)),
        index=df.index,
        dtype=float,
    )


def _candidate_thresholds(values: pd.Series) -> list[float]:
    clean = values.dropna()
    if clean.nunique() < 4:
        return []
    return sorted(set(float(x) for x in clean.quantile([0.1, 0.2, 0.33, 0.5, 0.67, 0.8]).to_list()))


def _is_usable_mask(mask: pd.Series, train_days: int) -> bool:
    no_trade_days = int(mask.sum())
    return 1 <= no_trade_days <= max(1, int(train_days * 0.2))


def _build_rule(
    train_df: pd.DataFrame,
    *,
    kind: str,
    feature_a: str,
    op_a: str,
    threshold_a: float,
    feature_b: str | None = None,
    op_b: str | None = None,
    threshold_b: float | None = None,
) -> NoTradeRule | None:
    rule = NoTradeRule(
        kind=kind,
        feature_a=feature_a,
        op_a=op_a,
        threshold_a=threshold_a,
        feature_b=feature_b,
        op_b=op_b,
        threshold_b=threshold_b,
        train_mean_profit=0.0,
        train_lift_vs_champion=0.0,
        train_no_trade_days=0,
    )
    mask = _apply_rule(train_df, rule)
    if not _is_usable_mask(mask, len(train_df)):
        return None

    profit = _profit_with_no_trade(train_df, mask)
    champion_mean = float(train_df[CHAMPION_PROFIT].mean())
    return NoTradeRule(
        kind=kind,
        feature_a=feature_a,
        op_a=op_a,
        threshold_a=threshold_a,
        feature_b=feature_b,
        op_b=op_b,
        threshold_b=threshold_b,
        train_mean_profit=float(profit.mean()),
        train_lift_vs_champion=float(profit.mean() - champion_mean),
        train_no_trade_days=int(mask.sum()),
    )


def learn_best_no_trade_rule(train_df: pd.DataFrame) -> NoTradeRule:
    best: NoTradeRule | None = None
    feature_thresholds = {
        feature: _candidate_thresholds(train_df[feature])
        for feature in NO_TRADE_FEATURES
        if feature in train_df.columns
    }

    for feature, thresholds in feature_thresholds.items():
        for threshold in thresholds:
            for op in ["<=", ">"]:
                candidate = _build_rule(
                    train_df,
                    kind="single",
                    feature_a=feature,
                    op_a=op,
                    threshold_a=threshold,
                )
                if candidate and (
                    best is None or candidate.train_mean_profit > best.train_mean_profit
                ):
                    best = candidate

    features = list(feature_thresholds)
    for left_idx, feature_a in enumerate(features):
        for feature_b in features[left_idx + 1 :]:
            for threshold_a in feature_thresholds[feature_a]:
                for threshold_b in feature_thresholds[feature_b]:
                    for op_a in ["<=", ">"]:
                        for op_b in ["<=", ">"]:
                            candidate = _build_rule(
                                train_df,
                                kind="pair",
                                feature_a=feature_a,
                                op_a=op_a,
                                threshold_a=threshold_a,
                                feature_b=feature_b,
                                op_b=op_b,
                                threshold_b=threshold_b,
                            )
                            if candidate and (
                                best is None
                                or candidate.train_mean_profit > best.train_mean_profit
                            ):
                                best = candidate

    if best is not None:
        return best

    champion_mean = float(train_df[CHAMPION_PROFIT].mean())
    return NoTradeRule(
        kind="none",
        feature_a=f"{CHAMPION}__predicted_spread",
        op_a="<=",
        threshold_a=float("-inf"),
        feature_b=None,
        op_b=None,
        threshold_b=None,
        train_mean_profit=champion_mean,
        train_lift_vs_champion=0.0,
        train_no_trade_days=0,
    )


def _summarize_profit(df: pd.DataFrame, profit: pd.Series, prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_mean_profit": float(profit.mean()),
        f"{prefix}_worst_profit": float(profit.min()),
        f"{prefix}_loss_days": int((profit < 0).sum()),
        f"{prefix}_no_trade_days": int((profit == 0).sum()),
    }


def cross_fold_evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []

    for fold in sorted(df["fold"].unique()):
        train_df = df[df["fold"] != fold].copy()
        valid_df = df[df["fold"] == fold].copy()
        rule = learn_best_no_trade_rule(train_df)
        no_trade_mask = _apply_rule(valid_df, rule)
        gated_profit = _profit_with_no_trade(valid_df, no_trade_mask)
        champion_profit = valid_df[CHAMPION_PROFIT].astype(float)

        row: dict[str, object] = {
            "heldout_fold": fold,
            "valid_days": len(valid_df),
            "rule_kind": rule.kind,
            "feature_a": rule.feature_a,
            "op_a": rule.op_a,
            "threshold_a": rule.threshold_a,
            "feature_b": rule.feature_b,
            "op_b": rule.op_b,
            "threshold_b": rule.threshold_b,
            "train_lift_vs_champion": rule.train_lift_vs_champion,
            "train_no_trade_days": rule.train_no_trade_days,
            "valid_no_trade_days": int(no_trade_mask.sum()),
            "valid_filtered_loss_days": int(((champion_profit < 0) & no_trade_mask).sum()),
            "valid_filtered_profit_days": int(((champion_profit > 0) & no_trade_mask).sum()),
        }
        row.update(_summarize_profit(valid_df, champion_profit, "champion"))
        row.update(_summarize_profit(valid_df, gated_profit, "no_trade_gate"))
        row["valid_lift_vs_champion"] = float(gated_profit.mean() - champion_profit.mean())
        fold_rows.append(row)

        for idx, day in valid_df.iterrows():
            daily_rows.append(
                {
                    "date": day["date"].date().isoformat(),
                    "fold": fold,
                    "champion_profit": float(champion_profit.loc[idx]),
                    "no_trade": bool(no_trade_mask.loc[idx]),
                    "no_trade_gate_profit": float(gated_profit.loc[idx]),
                    "lift_vs_champion": float(gated_profit.loc[idx] - champion_profit.loc[idx]),
                    "rule_kind": rule.kind,
                    "feature_a": rule.feature_a,
                    "feature_a_value": float(day[rule.feature_a]),
                    "threshold_a": rule.threshold_a,
                    "feature_b": rule.feature_b,
                    "feature_b_value": (
                        None if rule.feature_b is None else float(day[rule.feature_b])
                    ),
                    "threshold_b": rule.threshold_b,
                }
            )

    return pd.DataFrame(fold_rows), pd.DataFrame(daily_rows)


def build_overall(daily: pd.DataFrame) -> pd.DataFrame:
    champion = daily["champion_profit"]
    gated = daily["no_trade_gate_profit"]
    return pd.DataFrame(
        [
            {
                "method": "champion",
                "mean_profit": float(champion.mean()),
                "lift_vs_champion": 0.0,
                "worst_day_profit": float(champion.min()),
                "loss_days": int((champion < 0).sum()),
                "no_trade_days": 0,
            },
            {
                "method": "no_trade_gate",
                "mean_profit": float(gated.mean()),
                "lift_vs_champion": float(gated.mean() - champion.mean()),
                "worst_day_profit": float(gated.min()),
                "loss_days": int((gated < 0).sum()),
                "no_trade_days": int(daily["no_trade"].sum()),
            },
        ]
    )


def write_markdown(*, folds: pd.DataFrame, daily: pd.DataFrame, overall: pd.DataFrame) -> None:
    filtered = daily[daily["no_trade"]].copy()
    lines = [
        "# No-Trade Gate Diagnostics",
        "",
        "## Scope",
        "",
        "Default action is the current champion trade. A learned rule may set a whole day "
        "to no-trade, giving that day profit `0`.",
        "",
        "## Overall Held-Out Result",
        "",
        overall.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Fold Detail",
        "",
        folds.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## No-Trade Days",
        "",
    ]
    if filtered.empty:
        lines.append("No held-out no-trade days were selected.")
    else:
        keep_cols = [
            "date",
            "fold",
            "champion_profit",
            "lift_vs_champion",
            "rule_kind",
            "feature_a",
            "feature_a_value",
            "threshold_a",
            "feature_b",
            "feature_b_value",
            "threshold_b",
        ]
        lines.append(filtered[keep_cols].to_markdown(index=False, floatfmt=".4f"))

    lines += [
        "",
        "## Decision",
        "",
    ]
    lift = float(overall.loc[overall["method"] == "no_trade_gate", "lift_vs_champion"].iloc[0])
    if lift > 0:
        lines += [
            "- The held-out no-trade gate improves mean profit in this diagnostic.",
            "- Promotion still requires checking that gains are not from one fragile fold.",
        ]
    else:
        lines += [
            "- The held-out no-trade gate does not improve mean profit.",
            "- Do not promote no-trade filtering yet.",
        ]

    REPORTS_DIR.joinpath("no_trade_gate_diagnostics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    df = build_gate_dataset(load_config())
    folds, daily = cross_fold_evaluate(df)
    overall = build_overall(daily)

    df.to_csv(REPORTS_DIR / "no_trade_gate_dataset.csv", index=False)
    folds.to_csv(REPORTS_DIR / "no_trade_gate_crossfold_summary.csv", index=False)
    daily.to_csv(REPORTS_DIR / "no_trade_gate_crossfold_daily.csv", index=False)
    overall.to_csv(REPORTS_DIR / "no_trade_gate_overall.csv", index=False)
    write_markdown(folds=folds, daily=daily, overall=overall)

    print(overall.to_string(index=False))
    print(f"summary_path={REPORTS_DIR / 'no_trade_gate_crossfold_summary.csv'}")
    print(f"daily_path={REPORTS_DIR / 'no_trade_gate_crossfold_daily.csv'}")
    print(f"markdown_path={REPORTS_DIR / 'no_trade_gate_diagnostics.md'}")


if __name__ == "__main__":
    main()
