# Constrained Winter / Weekly Blend Experiment

## 目的

在 `holiday_only` 线上低于 champion 之后，本实验不再做全局替换，而是测试低权重混合：把 champion 与 holiday / weekly relative 专家按 5% 或 10% 小权重融合，检查冬季信号能否在不明显伤害 broad/test-like 验证的前提下带来收益。

## 实验配置

- 锚点：`ens_champion_segmented6_prior_5fold`。
- 调度：沿用 current champion 的 dispatch prior，不改变充放电枚举逻辑。
- 变体：`holiday05`、`holiday10`、`weekly05`、`weekly10`、`holiday05_weekly05`。
- 评估：5-fold，包括 `valid_2025_09`、`valid_2025_10`、`valid_2025_11`、`valid_2025_12`、`valid_2025_jan_feb`。
- 提交门槛：`all_5fold_delta >= -50`、`test_like_delta >= -50`、`loss_day_delta <= 0`，同时 Jan-Feb-like 或 late-winter 有正向理由。

## 总体结果

| model | std_09_12 | std_delta | jan_feb | jan_delta | all_5fold | all_delta | test_like | test_like_delta | loss_days | loss_delta | p90_regret | p90_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| champion | 7991.4 | 0.0 | 14153.5 | 0.0 | 9952.0 | 0.0 | 10152.6 | 0.0 | 4 | 0 | 9990.9 | 0.0 |
| holiday05 | 7936.4 | -55.0 | 14150.4 | -3.1 | 9913.5 | -38.5 | 10122.7 | -29.9 | 4 | 0 | 10338.6 | 347.7 |
| holiday10 | 7864.9 | -126.5 | 14149.2 | -4.3 | 9864.4 | -87.6 | 10074.9 | -77.7 | 4 | 0 | 10359.3 | 368.4 |
| weekly05 | 7873.4 | -118.0 | 14161.7 | 8.2 | 9874.2 | -77.8 | 10080.9 | -71.7 | 4 | 0 | 10338.6 | 347.7 |
| weekly10 | 7779.8 | -211.5 | 14162.3 | 8.8 | 9810.6 | -141.4 | 10027.2 | -125.4 | 4 | 0 | 10359.3 | 368.4 |
| holiday05_weekly05 | 7801.9 | -189.5 | 14161.7 | 8.2 | 9825.5 | -126.6 | 10051.4 | -101.3 | 4 | 0 | 10359.3 | 368.4 |

## 分 Fold 结果

| model | fold | avg_profit | loss_days | avg_regret |
| --- | --- | --- | --- | --- |
| holiday05 | valid_2025_09 | 11033.4 | 1 | 3327.6 |
| holiday05 | valid_2025_10 | 7889.1 | 3 | 6032.8 |
| holiday05 | valid_2025_11 | 7112.5 | 0 | 895.4 |
| holiday05 | valid_2025_12 | 5983.6 | 1 | 4152.0 |
| holiday05 | valid_2025_jan_feb | 14150.4 | 0 | 2504.7 |
| holiday10 | valid_2025_09 | 10989.4 | 1 | 3371.6 |
| holiday10 | valid_2025_10 | 7676.8 | 3 | 6245.2 |
| holiday10 | valid_2025_11 | 7112.5 | 0 | 895.4 |
| holiday10 | valid_2025_12 | 5958.9 | 1 | 4176.6 |
| holiday10 | valid_2025_jan_feb | 14149.2 | 0 | 2505.8 |
| weekly05 | valid_2025_09 | 10994.8 | 1 | 3366.3 |
| weekly05 | valid_2025_10 | 7742.4 | 3 | 6179.6 |
| weekly05 | valid_2025_11 | 7067.6 | 0 | 940.3 |
| weekly05 | valid_2025_12 | 5964.9 | 1 | 4170.6 |
| weekly05 | valid_2025_jan_feb | 14161.7 | 0 | 2493.3 |
| weekly10 | valid_2025_09 | 10787.9 | 1 | 3573.1 |
| weekly10 | valid_2025_10 | 7554.2 | 3 | 6367.8 |
| weekly10 | valid_2025_11 | 7059.3 | 0 | 948.6 |
| weekly10 | valid_2025_12 | 5985.7 | 1 | 4149.8 |
| weekly10 | valid_2025_jan_feb | 14162.3 | 0 | 2492.7 |
| holiday05_weekly05 | valid_2025_09 | 11031.9 | 1 | 3329.1 |
| holiday05_weekly05 | valid_2025_10 | 7432.6 | 3 | 6489.4 |
| holiday05_weekly05 | valid_2025_11 | 7059.3 | 0 | 948.6 |
| holiday05_weekly05 | valid_2025_12 | 5972.2 | 1 | 4163.3 |
| holiday05_weekly05 | valid_2025_jan_feb | 14161.7 | 0 | 2493.3 |

## 结论

- 没有任何低权重 blend 达到提交门槛。
- `holiday05` 是最接近安全线的版本，但 `Jan-Feb-like` 也没有提升，`all_5fold=-38.5/日`，`test_like=-29.9/日`，且 p90 regret 上升。
- `weekly05` 和 `weekly10` 在 `Jan-Feb-like` 有很小提升，但 broad/test-like 下降更明显，不适合作为提交候选。
- `holiday05_weekly05` 同时混合两个专家后没有互补，反而扩大 standard/test-like 损失。
- 这条路线的结论是：holiday/weekly 信号不能作为价格模型的全局低权重融合项；如果继续用，只能转为 top-k/risk 的辅助信号，而不是直接混入价格预测。

## 决策

- 不生成提交文件。
- 不继续扩大 holiday/weekly 全局 blend 网格。
- 后续转向 `weather_bidspace_topk_aux` 或 `model_family_consensus_fallback`，即只在窗口候选选择或风险识别阶段使用这些弱信号。

## Artifacts

- `reports/constrained_winter_blend_experiment.md`
- `reports/constrained_winter_blend_experiment_summary.csv`
- `reports/constrained_winter_blend_experiment_by_fold.csv`
- `reports/online_risk_dashboard.csv`
- `reports/online_validation_gap_candidate_filter.csv`
