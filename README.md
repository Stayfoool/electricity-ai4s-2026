# Electricity AI4S 2026

本仓库是第四届世界科学智能大赛“电力市场交易赛道：储能电站收益优化”的实验工程。项目目标不是只做电价点预测，而是在给定储能约束下，基于未来 24 小时、96 个 15 分钟电价预测结果，生成实际收益更高的充放电策略。

仓库已经公开，欢迎对代码、实验设计、特征工程、回测口径、调度策略和文档进行审阅与改进。

## 任务概述

比赛要求参赛队伍基于历史实时电价、边界条件预测值和气象预报信息，预测蒙西地区某节点次日 0 点起未来 24 小时实时市场电价序列。

储能调度约束：

- 每天 `96` 个 15 分钟点。
- 充电必须连续 `8` 个点，即 2 小时。
- 放电必须连续 `8` 个点，即 2 小时。
- 放电窗口必须在充电窗口之后，且两者不能重叠。
- 当前调度器枚举每日所有合法充放电窗口对，共 `3321` 个 pair，并选择预测收益最高的 pair。

提交文件中的 `power` 会被官方用隐藏真实电价计算收益，因此最终目标是收益最大化，而不是单纯降低 RMSE。

## 当前方案

当前线上最好方案是：

```text
ens_champion_segmented6_prior
```

结构：

- `0.25 * lgb_baseline`
- `0.25 * lgb_baseline_last_180d`
- `0.50 * lgb_segmented_6_last_180d`
- 调度阶段加入历史 oracle slot prior：
  - `lambda_charge = 0.18`
  - `lambda_discharge = 0.40`
  - `alpha = 0.5`

线上记录：

| model | online score | note |
|---|---:|---|
| `ens_champion_segmented6_prior` | `5482` | current best |
| `ens_lgb180_seg90_w65_35` | `5454` | close but lower |
| `ens_champion_segmented6_prior_clean_features_5fold` | `5026` | Jan-Feb-like local positive, online rejected |
| `lgb_segmented_6_last_180d_prior` | `4946` | standalone segmented with prior failed |
| `lgb_segmented_6_last_90d` | `4875` | pure winter/90d hypothesis failed online |
| `lgb_segmented_6_spatial_last_180d` | `4599` | always-on spatial weather features failed online |

详细记录见：

- `reports/online_submission_scores.md`
- `reports/experiment_lineage_matrix.md`
- `reports/jan_feb_decision_matrix.md`

## 已有实验结论

比较稳定的结论：

- LightGBM 点预测 + 分时段建模 + ensemble + dispatch prior 是目前最稳主线。
- `lgb_segmented_6_last_180d` 证明不同时段电价机制不同，分时段模型有效。
- 简单加天气、直接修正 bid space、直接窗口均价预测、直接 pair spread 预测，目前都没有超过 champion。
- 2025 年 1-2 月的 Jan-Feb-like 验证集已经连续误导线上结果，不能单独作为提交依据。
- 纯冬季专家、纯 90 天 segmented、always-on 空间天气特征都已经被线上结果否定。

仍值得继续审阅和尝试的方向：

- 更稳健的 top-k pair rerank，而不是只选择预测 top1 pair。
- 将天气、bid space、边界条件误差修正作为风险或 rerank 特征，而不是直接全量喂入主价格模型。
- 更可靠的线上代理验证设计，尤其是解释为什么 2025-01/02 与 2026-01/02 分布不一致。
- 小幅、可解释、低风险地改进 champion，而不是大规模堆特征。

## 仓库结构

```text
.
├── configs/                 # Hydra/OmegaConf 风格的实验配置
├── src/electricity/         # 核心代码
│   ├── data/                # 数据读取
│   ├── dispatch/            # 充放电调度与 prior
│   ├── eval/                # backtest / segmented / window / VMD 等评估
│   ├── features/            # 时间、天气、bid space、bias correction 等特征
│   ├── models/              # LightGBM / linear / tabular 模型封装
│   └── submit.py            # 提交生成与校验
├── scripts/                 # 诊断、实验矩阵、报告生成脚本
├── tests/                   # 单元测试
├── reports/                 # 实验报告与结果摘要
├── outputs/                 # 选定提交文件
├── Makefile                 # 统一命令入口
└── requirements.txt
```

## 数据与公开范围

比赛原始数据不随仓库发布。

原因：

- 比赛数据来源和再分发权限需遵守官方规则。
- 本仓库只公开代码、配置、报告和可复现实验流程。
- 如果你参加同一比赛，请从官方平台下载数据，并按相同目录结构放置。
- 不要把官方比赛原始数据、官方未公开测试标签、私有账号信息或平台 cookie 提交到本仓库。

默认数据路径：

```text
eletricmaterial/to_sais_new/train/mengxi_boundary_anon_filtered.csv
eletricmaterial/to_sais_new/train/mengxi_node_price_selected.csv
eletricmaterial/to_sais_new/test/test_in_feature_ori.csv
```

`eletricmaterial/` 已在 `.gitignore` 中排除。

说明：

- 代码和文档按 MIT License 开源。
- 比赛原始数据不包含在本许可证授权范围内，仍受比赛主办方规则约束。
- `outputs/` 中的提交文件和 `reports/` 中的实验结果可能暴露策略细节；公开仓库中保留它们是为了接受审阅。如果你 fork 后不想公开自己的提交策略，请把私有提交文件保留在本地。

## 环境安装

推荐 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果使用 AutoDL/Conda，请确保 `Makefile` 中的 `PY` 指向正确 Python：

```bash
make test PY=.venv/bin/python
```

## 常用命令

运行测试：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

运行普通 backtest：

```bash
PYTHONPATH=src .venv/bin/python -m electricity.cli backtest \
  --config configs/lgb_baseline_last_180d.yaml
```

运行 ensemble backtest：

```bash
PYTHONPATH=src .venv/bin/python -m electricity.cli ensemble \
  --config configs/ensemble_champion_segmented6_prior.yaml
```

生成提交文件：

```bash
PYTHONPATH=src .venv/bin/python -m electricity.cli submit \
  --config configs/ensemble_champion_segmented6_prior.yaml
```

使用 Makefile：

```bash
make test PY=.venv/bin/python
make ensemble PY=.venv/bin/python CONFIG=configs/ensemble_champion_segmented6_prior.yaml
make submit PY=.venv/bin/python CONFIG=configs/ensemble_champion_segmented6_prior.yaml
```

## 如何审阅和贡献

欢迎通过 issue 或 pull request 帮忙审阅以下问题：

- 回测是否存在时间泄漏。
- 测试集推理是否只使用比赛允许的特征。
- dispatch prior 是否过拟合历史月份。
- Jan-Feb-like 验证为什么和线上分数不一致。
- 特征工程是否有噪声特征、重复特征或不稳定特征。
- 天气数据的 UTC/北京时间对齐是否严格正确。
- 3321 个合法 pair 的枚举和收益计算是否与赛题一致。
- 是否有更稳健的 top-k rerank / risk-aware dispatch 方法。
- 是否有更合理的 winter-like / 2026 Jan-Feb proxy validation。

贡献建议：

- 新增实验前，请先说明它改变的是“数据、特征、模型、训练窗口、调度策略、验证口径”中的哪一项。
- 新增结果请登记到 `reports/experiment_lineage_matrix.md` 或对应报告。
- PR 前请至少运行：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

如果修改 Python 代码，也建议运行：

```bash
.venv/bin/python -m ruff check src scripts tests
```

## 当前优先问题

我们目前最需要外部审阅的不是“再堆一个模型”，而是以下几个判断：

1. 为什么 `2025-01/02` 的 Jan-Feb-like fold 对 `2026-01/02` 线上测试有明显误导？
2. 当前 champion 的收益来源主要来自价格曲线预测、分时段模型，还是 dispatch prior？
3. 有没有办法在不牺牲线上稳定性的前提下，从 top-k 合法 pair 中选出比 top1 更好的 pair？
4. 天气和 bid space 已经显示出物理意义，但直接加入价格模型失败，是否更适合作为 gating/rerank/risk 特征？
5. 现有模型是否存在时间特征、边界条件预测值、天气预测值的对齐问题？

## 合规说明

- 请勿向本仓库提交官方未授权再分发的比赛原始数据。
- 请勿提交私有账号、token、SSH key、AutoDL 密码或平台 cookie。
- 如引用论文、外部数据或预训练模型，请记录来源和许可证。
- 任何第三方数据源都应优先使用官方来源。

## License

This project is released under the MIT License. See `LICENSE`.

Competition data is not included and remains subject to the competition organizer's terms.
