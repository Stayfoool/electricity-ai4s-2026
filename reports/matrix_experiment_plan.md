# Matrix Experiment Plan

## 结论

- 当前确实有局部最优风险：我们沿着 `LightGBM 点预测 + 枚举调度` 主线做了大量局部改动，但很多改动是在同一条路径上微调。
- 但这不等于要推倒重来。当前线上 champion 仍是最强锚点，下一步应做“矩阵式局部组合”，而不是盲目换大模型或全局堆特征。
- 已失败组件不能简单丢弃：天气、竞价空间、窗口/pair 模型、周相对特征、CatBoost 等，全局替代失败，但仍可作为 top-k 风险信号、冬季局部专家或 fallback 信号。
- 简单不交易门槛已经失败；坏日处理的动作应优先是“换更稳的 pair”，不是“当天不交易”。

## 已覆盖实验空间

| decision_bucket | count |
| --- | --- |
| global_failed | 41 |
| rejected | 21 |
| candidate | 8 |
| promoted | 5 |
| online_tested | 2 |
| near_champion | 1 |

## 当前关键证据

- 线上记录：`champion=5482`，`holiday_only=5370`。holiday 在 Jan-Feb-like 本地更好，但线上更差，说明不能只相信单一冬季验证。
- 当前 champion 的主要损失不是价差太小，而是高 regret 日选错窗口；已有 no-trade gate 过滤了赚钱日，没有过滤亏损日。
- champion top10 候选仍有空间：top10 oracle 在 all_5fold 上有 `+737.6/日` 上限；但 fold-safe learned reranker 已经失败，说明要做保守局部修正。
- 特征删减实验显示 7 个边界预测值和时间特征整体都有用；当前没有明显可以直接删掉的强噪声特征。
- forecast-error augmentation 能减少部分亏损日，但平均收益和尾部 regret 变差，适合作为稳定性/坏日信号，不适合全局替代。
- `centered_5fold` 已完成并拒绝：全 fold 均弱于 champion，说明仅优化日内偏离形状没有改善最终充放电窗口选择。
- `topk_conservative_bad_day_switch` 已完成：方向为正但幅度太小，保留为诊断信号，不生成提交。
- `bad_day_action_not_no_trade` 已完成：换 pair 替代不交易的方向也只有 `+0.86/日`，说明单规则坏日处理不能解释 top10 oracle 上限。
- `model_family_consensus_fallback` 已完成：family top1 oracle 有 `+982.4/日`，但 fold-safe 简单 fallback 为 `-56.8/日`，说明跨模型分歧有上限但不能靠简单规则直接利用。

## Online 风险视图

| model | risk_class | online_score | all_5fold_mean_profit | jan_feb_like_mean_profit | test_like_weighted_profit | all_5fold_loss_days | all_5fold_p90_regret |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold | baseline_champion | 5482.0 | 9952.0 | 14153.5 | 10152.6 | 4 | 9990.9 |
| ens_champion_segmented6_prior_5fold_holiday_only | winter_overfit_risk | 5370.0 | 9735.4 | 14370.7 | 9938.4 | 5 | 11377.1 |
| ensemble_champion_segmented6_prior_blend_holiday05_5fold | watch |  | 9913.5 | 14150.4 | 10122.7 | 4 | 10338.6 |
| ensemble_champion_segmented6_prior_blend_weekly05_5fold | watch |  | 9874.2 | 14161.7 | 10080.9 | 4 | 10338.6 |
| ensemble_champion_segmented6_prior_blend_holiday10_5fold | watch |  | 9864.4 | 14149.2 | 10074.9 | 4 | 10359.3 |
| ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold | watch |  | 9834.3 | 14213.8 | 10042.1 | 3 | 10080.7 |
| ensemble_champion_segmented6_prior_blend_holiday05_weekly05_5fold | watch |  | 9825.5 | 14161.7 | 10051.4 | 4 | 10359.3 |
| ensemble_champion_segmented6_prior_blend_holiday20_5fold | watch |  | 9823.8 | 14210.0 | 10051.7 | 4 | 10359.3 |

## Top-K 证据

| regime | method | days | mean_profit | loss_days | mean_lift_vs_anchor | changed_days |
| --- | --- | --- | --- | --- | --- | --- |
| all_5fold | oracle_top10 | 176 | 10689.6 | 2 | 737.6 | 155 |
| all_5fold | anchor_top1 | 176 | 9952.0 | 4 | 0.0 | 0 |
| all_5fold | gbr_lift | 176 | 9947.4 | 4 | -4.7 | 12 |
| all_5fold | gbr_true_profit | 176 | 9840.3 | 4 | -111.7 | 41 |

## 可复用但不应全局替代的组件

| model | decision_bucket | model_family | feature_flags | avg_profit_mean | jan_feb_like_avg_profit | opportunity_tags |
| --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_5fold | near_champion | ensemble | base | 9282.4 | 14153.5 | has_jan_feb_evidence |
| ensemble_champion_segmented6_prior_blend_holiday05_5fold | rejected | ensemble | holiday | 9233.8 | 14150.4 | has_jan_feb_evidence |
| ensemble_champion_segmented6_prior_blend_weekly05_5fold | rejected | ensemble | weekly_relative | 9186.3 | 14161.7 | has_jan_feb_evidence/temporal_relative_candidate |
| ensemble_champion_segmented6_prior_blend_holiday10_5fold | rejected | ensemble | holiday | 9177.4 | 14149.2 | has_jan_feb_evidence |
| ens_champion_segmented6_prior_clean_features_5fold | global_failed | ensemble | bias_correction | 9132.5 | 14438.7 | has_jan_feb_evidence |
| ensemble_champion_segmented6_prior_blend_holiday05_weekly05_5fold | rejected | ensemble | holiday/weekly_relative | 9131.6 | 14161.7 | has_jan_feb_evidence/temporal_relative_candidate |
| ensemble_champion_segmented6_prior_blend_holiday15_weekly15_5fold | global_failed | ensemble | holiday/weekly_relative | 9130.9 | 14213.8 | has_jan_feb_evidence/temporal_relative_candidate |
| ensemble_champion_segmented6_prior_blend_holiday20_5fold | global_failed | ensemble | holiday | 9117.5 | 14210.0 | has_jan_feb_evidence |
| ensemble_champion_segmented6_prior_blend_weekly10_5fold | rejected | ensemble | weekly_relative | 9109.9 | 14162.3 | has_jan_feb_evidence/temporal_relative_candidate |
| ensemble_champion_segmented6_prior_blend_weekly20_5fold | global_failed | ensemble | weekly_relative | 9103.0 | 14149.7 | has_jan_feb_evidence/temporal_relative_candidate |
| ens_champion_segmented6_prior_shallow_d4_l15_leaf200_5fold | global_failed | ensemble | base | 9039.6 | 13916.7 | has_jan_feb_evidence |
| ens_champion_segmented6_prior_shallow_d5_l31_leaf150_5fold | rejected | ensemble | base | 9028.0 | 13939.3 | has_jan_feb_evidence |
| ens_champion_segmented6_prior_replace_features_5fold | global_failed | ensemble | bias_correction | 8991.2 | 14347.5 | has_jan_feb_evidence |
| ens_champion_segmented6_prior_clean_features_shrink05_5fold | global_failed | ensemble | bias_correction | 8988.1 | 14314.4 | has_jan_feb_evidence |
| ens_champion_segmented6_prior_feaug_ws025_w05_5fold | global_failed | ensemble | base | 8976.9 | 13812.1 | has_jan_feb_evidence |
| ens_champion_segmented6_prior_5fold_holiday_only | global_failed | ensemble | holiday | 8976.3 | 14370.7 | has_jan_feb_evidence |
| ens_champion_segmented6_prior_5fold_cap_only | global_failed | ensemble | capacity | 8958.2 | 13644.8 | has_jan_feb_evidence |
| ens_champion_segmented6_prior_feaug_ws_late050_w05_5fold | global_failed | ensemble | base | 8931.0 | 13750.7 | has_jan_feb_evidence |
| ens_champion_segmented6_prior_shallow_d3_l7_leaf200_5fold | global_failed | ensemble | base | 8913.9 | 13966.3 | has_jan_feb_evidence |
| ens_champion_segmented6_prior_weekly_delta_percentile_5fold | global_failed | ensemble | weekly_relative | 8877.4 | 14164.9 | has_jan_feb_evidence/temporal_relative_candidate |

## 未测配置

_无数据_

## 已完成的矩阵实验

| model | decision_bucket | model_family | target_family | avg_profit_mean | jan_feb_like_avg_profit | daily_loss_days |
| --- | --- | --- | --- | --- | --- | --- |
| ens_champion_segmented6_prior_centered_5fold | rejected | ensemble | shape_centered | 8058.9 | 13095.8 | 11.0 |

| experiment_id | decision | mean_profit | profit_delta_vs_anchor | loss_days | loss_day_delta_vs_anchor | changed_days |
| --- | --- | --- | --- | --- | --- | --- |
| topk_conservative_bad_day_switch | diagnostic_positive_below_submit_threshold | 9977.7 | 25.7 | 4 | 0 | 16 |

## 已完成的坏日动作实验

| method | mean_profit | profit_delta_vs_anchor | loss_days | loss_day_delta_vs_anchor | changed_days |
| --- | --- | --- | --- | --- | --- |
| oracle_top10 | 10689.6 | 737.6 | 2 | -2 | 131 |
| bad_day_action_not_no_trade | 9952.9 | 0.9 | 4 | 0 | 16 |
| anchor_top1 | 9952.0 | 0.0 | 4 | 0 | 0 |

## 已完成的模型家族 fallback 实验

| experiment_id | decision | mean_profit | profit_delta_vs_anchor | loss_days | loss_day_delta_vs_anchor | changed_days | positive_lift_days | negative_lift_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_family_consensus_fallback | rejected_negative_foldsafe_lift | 7934.6 | -56.8 | 4.0 | 0.0 | 8.0 | 3.0 | 5.0 |

## 下一批实验队列

| priority | experiment_id | status | axis | base | reuse_component | hypothesis | promotion_gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | topk_conservative_bad_day_switch | diagnostic_done | dispatch_selector | current champion top1 | top10 candidate + bid_space/weather risk | top10 oracle 有 +737.6/日上限，但学习式 reranker 过拟合；保守规则只改少数高风险日可能更稳。 | 已通过方向性约束但未达提交门槛：all_5fold +25.7/日，loss days 不增加，低于 +100/日阈值。 |
| 2 | bad_day_action_not_no_trade | diagnostic_done | risk_action | current champion top1 | no-trade gate diagnostics | 坏日识别仍有价值，但动作不应直接不交易，而应换保守 pair 或 consensus pair。 | 未通过：all_5fold 仅 +0.86/日，loss days 不变，远低于 +100/日提交门槛。 |

## 执行原则

- 每次只新增一个矩阵维度：例如先固定 champion 和特征，只换 target；或固定 champion top10，只换选择规则。
- 每个实验必须同时看 `all_5fold`、`standard_09_12`、`Jan-Feb-like`、`test-like weighted`、loss days、p90 regret。
- 本地提升小于 `100/日` 且风险指标不改善，不进入线上提交候选。
- 对 top-k/risk/gate 类实验，优先要求“少改天数、少伤普通日”，不要追求每天都重新选择。
- 任何使用实际值、真实价格、验证集统计归一化的路径都要做 fold-safe 检查，防止本地虚高。

## 建议立即执行

1. 停止加复杂度到单规则坏日切换和简单 model-family fallback；当前信号不足以线上提交。
2. 下一步优先做冬季/线上分布导向的局部专家或 test-like 风险分析。
3. 同步维护 online risk dashboard，把任何准备提交的候选都放进去与 champion、holiday_only 对比。

## Artifacts

- `reports/matrix_experiment_plan.md`
- `reports/next_experiment_queue.csv`
