from __future__ import annotations

from pathlib import Path

import pandas as pd

REPORTS_DIR = Path("reports")
EXPERIMENT_MAP = REPORTS_DIR / "experiment_map.csv"
ONLINE_RISK = REPORTS_DIR / "online_risk_dashboard.csv"
TOPK_OVERALL = REPORTS_DIR / "current_prior_topk_pair_reranker_foldsafe_overall.csv"
CONSERVATIVE_SWITCH_OVERALL = (
    REPORTS_DIR / "current_prior_topk_conservative_switch_overall.csv"
)
OUT_MD = REPORTS_DIR / "matrix_experiment_plan.md"
OUT_QUEUE = REPORTS_DIR / "next_experiment_queue.csv"


def fmt(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(df: pd.DataFrame, cols: list[str], limit: int = 20) -> str:
    if df.empty:
        return "_无数据_"
    block = df[cols].head(limit).copy()
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in block.iterrows():
        cells = [fmt(row.get(col)).replace("|", "/") for col in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def make_queue() -> pd.DataFrame:
    rows = [
        {
            "priority": 0,
            "experiment_id": "shape_centered_champion_5fold",
            "axis": "target",
            "base": "ens_champion_segmented6_prior_5fold",
            "reuse_component": "centered target",
            "hypothesis": "日内形状目标可能改善窗口排序，而不依赖价格绝对平移。",
            "action": (
                "已运行 configs/ensemble_champion_segmented6_prior_centered_5fold.yaml；"
                "结果为全 fold 弱于 champion。"
            ),
            "promotion_gate": (
                "未通过：all_5fold -1196.6/日，Jan-Feb-like -1057.7/日，"
                "loss days 5->11。"
            ),
            "risk": "如果收益下降，说明形状目标没有转化为合法窗口选择优势。",
            "status": "rejected_done",
        },
        {
            "priority": 1,
            "experiment_id": "topk_conservative_bad_day_switch",
            "axis": "dispatch_selector",
            "base": "current champion top1",
            "reuse_component": "top10 candidate + bid_space/weather risk",
            "hypothesis": (
                "top10 oracle 有 +737.6/日上限，但学习式 reranker 过拟合；"
                "保守规则只改少数高风险日可能更稳。"
            ),
            "action": (
                "默认 top1；只在 champion 低信心、bid_space 排序冲突、"
                "天气残差信号一致时换 top2/top3。"
            ),
            "promotion_gate": (
                "已通过方向性约束但未达提交门槛：all_5fold +25.7/日，"
                "loss days 不增加，低于 +100/日阈值。"
            ),
            "risk": "局部规则太激进会复制已失败 reranker 的问题。",
            "status": "diagnostic_done",
        },
        {
            "priority": 2,
            "experiment_id": "bad_day_action_not_no_trade",
            "axis": "risk_action",
            "base": "current champion top1",
            "reuse_component": "no-trade gate diagnostics",
            "hypothesis": (
                "坏日识别仍有价值，但动作不应直接不交易，而应换保守 pair "
                "或 consensus pair。"
            ),
            "action": (
                "已运行 fold-safe 单规则动作选择；动作候选为 prior-disabled pair、"
                "consensus pair、rank2/3 pair、aux bid_space/support pair。"
            ),
            "promotion_gate": (
                "未通过：all_5fold 仅 +0.86/日，loss days 不变，"
                "远低于 +100/日提交门槛。"
            ),
            "risk": "坏日样本少，容易把赚钱日误判为坏日。",
            "status": "diagnostic_done",
        },
        {
            "priority": 3,
            "experiment_id": "winter_weekly_holiday_local_blend",
            "axis": "regime_specialist",
            "base": "current champion",
            "reuse_component": "weekly_delta_percentile + holiday_only",
            "hypothesis": "冬季/节假日信号在部分验证制度下有正信号，但不能全局替换 champion。",
            "action": (
                "已运行低权重全局 blend：holiday05/10、weekly05/10、"
                "holiday05_weekly05。"
            ),
            "promotion_gate": (
                "未通过：最接近的 holiday05 仍为 all_5fold -38.5/日、"
                "test-like -29.9/日、Jan-Feb-like -3.1/日，且 p90 regret 上升。"
            ),
            "risk": "holiday/weekly 不能作为全局价格预测混合项；后续只能作为 top-k/risk 辅助信号。",
            "status": "rejected_done",
        },
        {
            "priority": 4,
            "experiment_id": "weather_bidspace_topk_aux",
            "axis": "feature_as_auxiliary",
            "base": "current champion top10",
            "reuse_component": "bid_space + NWP residual + renewable forecast error",
            "hypothesis": "天气/竞价空间全局入模失败，但可能解释 top-k pair 中哪些候选更危险。",
            "action": (
                "已运行 fold-safe top10 辅助切换；候选分数包括 bid_space、net_load、"
                "cloud/GHI、wind residual 和专家 support。"
            ),
            "promotion_gate": (
                "未通过：all_5fold -10.2/日，Jan-Feb-like +0.6/日，"
                "loss days 不变但均值未超过 anchor。"
            ),
            "risk": "天气/竞价空间信号可解释部分候选，但不足以稳定决定切换。",
            "status": "rejected_done",
        },
        {
            "priority": 5,
            "experiment_id": "model_family_consensus_fallback",
            "axis": "model_family",
            "base": "current champion",
            "reuse_component": "CatBoost/XGBoost/segmented_margin candidates",
            "hypothesis": "非 LGBM 模型单独不强，但分歧可作为风险信号或 fallback 信号。",
            "action": (
                "已运行 09-12 月共同覆盖 120 天 fold-safe fallback：默认保留 champion，"
                "只在低置信和模型近似一致时切到 Cat/XGB/seg6/margin6 top1。"
            ),
            "promotion_gate": (
                "未通过：family top1 oracle 有 +982.4/日上限，但 learned fallback "
                "为 -56.8/日，改 8 天中 3 天正向、5 天负向。"
            ),
            "risk": "简单平均已不够强，不能再做无约束 ensemble。",
            "status": "rejected_done",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    exp = read_csv(EXPERIMENT_MAP)
    risk = read_csv(ONLINE_RISK)
    topk = read_csv(TOPK_OVERALL)
    conservative = read_csv(CONSERVATIVE_SWITCH_OVERALL)
    queue = make_queue()
    queue.to_csv(OUT_QUEUE, index=False)

    bucket = (
        exp["decision_bucket"].value_counts().rename_axis("decision_bucket").reset_index(name="count")
        if not exp.empty
        else pd.DataFrame()
    )
    unmeasured = exp[exp["decision_bucket"].eq("unmeasured")] if not exp.empty else pd.DataFrame()
    reusable = (
        exp[exp["opportunity_tags"].fillna("").ne("")]
        .sort_values(
            ["avg_profit_mean", "jan_feb_like_avg_profit"],
            ascending=False,
            na_position="last",
        )
        if not exp.empty
        else pd.DataFrame()
    )
    online_cols = [
        "model",
        "risk_class",
        "online_score",
        "all_5fold_mean_profit",
        "jan_feb_like_mean_profit",
        "test_like_weighted_profit",
        "all_5fold_loss_days",
        "all_5fold_p90_regret",
    ]
    online_view = (
        risk[[c for c in online_cols if c in risk.columns]].head(8) if not risk.empty else risk
    )
    topk_view = topk.sort_values("mean_profit", ascending=False) if not topk.empty else topk
    queue_todo = queue[queue["status"] != "rejected_done"].copy()
    centered_view = exp[
        exp["model"].eq("ens_champion_segmented6_prior_centered_5fold")
    ].copy() if not exp.empty else pd.DataFrame()
    conservative_view = pd.DataFrame()
    if not conservative.empty:
        row = conservative[conservative["method"] == "conservative_switch"].copy()
        if not row.empty:
            row = row.assign(
                experiment_id="topk_conservative_bad_day_switch",
                decision="diagnostic_positive_below_submit_threshold",
            )
            conservative_view = row[
                [
                    "experiment_id",
                    "decision",
                    "mean_profit",
                    "profit_delta_vs_anchor",
                    "loss_days",
                    "loss_day_delta_vs_anchor",
                    "changed_days",
                ]
            ]

    model_family_view = pd.DataFrame()
    model_family = read_csv(REPORTS_DIR / "model_family_consensus_fallback_overall.csv")
    if not model_family.empty:
        row = model_family[model_family["method"] == "model_family_consensus_fallback"].copy()
        if not row.empty:
            row = row.assign(
                experiment_id="model_family_consensus_fallback",
                decision="rejected_negative_foldsafe_lift",
            )
            model_family_view = row[
                [
                    "experiment_id",
                    "decision",
                    "mean_profit",
                    "profit_delta_vs_anchor",
                    "loss_days",
                    "loss_day_delta_vs_anchor",
                    "changed_days",
                    "positive_lift_days",
                    "negative_lift_days",
                ]
            ]

    lines = [
        "# Matrix Experiment Plan",
        "",
        "## 结论",
        "",
        (
            "- 当前确实有局部最优风险：我们沿着 `LightGBM 点预测 + 枚举调度` "
            "主线做了大量局部改动，但很多改动是在同一条路径上微调。"
        ),
        (
            "- 但这不等于要推倒重来。当前线上 champion 仍是最强锚点，"
            "下一步应做“矩阵式局部组合”，而不是盲目换大模型或全局堆特征。"
        ),
        (
            "- 已失败组件不能简单丢弃：天气、竞价空间、窗口/pair 模型、"
            "周相对特征、CatBoost 等，全局替代失败，但仍可作为 top-k "
            "风险信号、冬季局部专家或 fallback 信号。"
        ),
        "- 简单不交易门槛已经失败；坏日处理的动作应优先是“换更稳的 pair”，不是“当天不交易”。",
        "",
        "## 已覆盖实验空间",
        "",
        md_table(bucket, ["decision_bucket", "count"]),
        "",
        "## 当前关键证据",
        "",
        (
            "- 线上记录：`champion=5482`，`holiday_only=5370`。holiday 在 "
            "Jan-Feb-like 本地更好，但线上更差，说明不能只相信单一冬季验证。"
        ),
        (
            "- 当前 champion 的主要损失不是价差太小，而是高 regret 日选错窗口；"
            "已有 no-trade gate 过滤了赚钱日，没有过滤亏损日。"
        ),
        (
            "- champion top10 候选仍有空间：top10 oracle 在 all_5fold 上有 "
            "`+737.6/日` 上限；但 fold-safe learned reranker 已经失败，"
            "说明要做保守局部修正。"
        ),
        (
            "- 特征删减实验显示 7 个边界预测值和时间特征整体都有用；"
            "当前没有明显可以直接删掉的强噪声特征。"
        ),
        (
            "- forecast-error augmentation 能减少部分亏损日，但平均收益和尾部 "
            "regret 变差，适合作为稳定性/坏日信号，不适合全局替代。"
        ),
        (
            "- `centered_5fold` 已完成并拒绝：全 fold 均弱于 champion，"
            "说明仅优化日内偏离形状没有改善最终充放电窗口选择。"
        ),
        (
            "- `topk_conservative_bad_day_switch` 已完成：方向为正但幅度太小，"
            "保留为诊断信号，不生成提交。"
        ),
        (
            "- `bad_day_action_not_no_trade` 已完成：换 pair 替代不交易的方向也只有 "
            "`+0.86/日`，说明单规则坏日处理不能解释 top10 oracle 上限。"
        ),
        (
            "- `model_family_consensus_fallback` 已完成：family top1 oracle 有 "
            "`+982.4/日`，但 fold-safe 简单 fallback 为 `-56.8/日`，说明"
            "跨模型分歧有上限但不能靠简单规则直接利用。"
        ),
        "",
        "## Online 风险视图",
        "",
        md_table(online_view, list(online_view.columns), limit=8),
        "",
        "## Top-K 证据",
        "",
        md_table(
            topk_view,
            [
                "regime",
                "method",
                "days",
                "mean_profit",
                "loss_days",
                "mean_lift_vs_anchor",
                "changed_days",
            ],
            limit=8,
        ),
        "",
        "## 可复用但不应全局替代的组件",
        "",
        md_table(
            reusable,
            [
                "model",
                "decision_bucket",
                "model_family",
                "feature_flags",
                "avg_profit_mean",
                "jan_feb_like_avg_profit",
                "opportunity_tags",
            ],
            limit=20,
        ),
        "",
        "## 未测配置",
        "",
        md_table(
            unmeasured,
            ["model", "model_family", "target_family", "feature_flags", "config_path"],
            limit=20,
        ),
        "",
        "## 已完成的矩阵实验",
        "",
        md_table(
            centered_view,
            [
                "model",
                "decision_bucket",
                "model_family",
                "target_family",
                "avg_profit_mean",
                "jan_feb_like_avg_profit",
                "daily_loss_days",
            ],
            limit=20,
        ),
        "",
        md_table(
            conservative_view,
            [
                "experiment_id",
                "decision",
                "mean_profit",
                "profit_delta_vs_anchor",
                "loss_days",
                "loss_day_delta_vs_anchor",
                "changed_days",
            ],
            limit=20,
        ),
        "",
        "## 已完成的坏日动作实验",
        "",
        md_table(
            read_csv(REPORTS_DIR / "bad_day_action_not_no_trade_overall.csv"),
            [
                "method",
                "mean_profit",
                "profit_delta_vs_anchor",
                "loss_days",
                "loss_day_delta_vs_anchor",
                "changed_days",
            ],
            limit=10,
        ),
        "",
        "## 已完成的模型家族 fallback 实验",
        "",
        md_table(
            model_family_view,
            [
                "experiment_id",
                "decision",
                "mean_profit",
                "profit_delta_vs_anchor",
                "loss_days",
                "loss_day_delta_vs_anchor",
                "changed_days",
                "positive_lift_days",
                "negative_lift_days",
            ],
            limit=10,
        ),
        "",
        "## 下一批实验队列",
        "",
        md_table(
            queue_todo,
            [
                "priority",
                "experiment_id",
                "status",
                "axis",
                "base",
                "reuse_component",
                "hypothesis",
                "promotion_gate",
            ],
            limit=20,
        ),
        "",
        "## 执行原则",
        "",
        (
            "- 每次只新增一个矩阵维度：例如先固定 champion 和特征，只换 target；"
            "或固定 champion top10，只换选择规则。"
        ),
        (
            "- 每个实验必须同时看 `all_5fold`、`standard_09_12`、"
            "`Jan-Feb-like`、`test-like weighted`、loss days、p90 regret。"
        ),
        "- 本地提升小于 `100/日` 且风险指标不改善，不进入线上提交候选。",
        "- 对 top-k/risk/gate 类实验，优先要求“少改天数、少伤普通日”，不要追求每天都重新选择。",
        "- 任何使用实际值、真实价格、验证集统计归一化的路径都要做 fold-safe 检查，防止本地虚高。",
        "",
        "## 建议立即执行",
        "",
        "1. 停止加复杂度到单规则坏日切换和简单 model-family fallback；当前信号不足以线上提交。",
        "2. 下一步优先做冬季/线上分布导向的局部专家或 test-like 风险分析。",
        (
            "3. 同步维护 online risk dashboard，把任何准备提交的候选都放进去与 "
            "champion、holiday_only 对比。"
        ),
        "",
        "## Artifacts",
        "",
        f"- `{OUT_MD}`",
        f"- `{OUT_QUEUE}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_path={OUT_MD}")
    print(f"queue_path={OUT_QUEUE}")
    print(f"queue_rows={len(queue)}")


if __name__ == "__main__":
    main()
