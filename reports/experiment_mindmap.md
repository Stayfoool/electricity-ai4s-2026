# AI4S Electricity Experiments Mindmap

> 状态标记：`[线上]` 已线上验证；`[冠军]` 当前主锚点；`[候选]` 有局部价值但未提交；`[拒绝]` 已明确不作为主线；`[诊断]` 只用于理解问题。

```mermaid
mindmap
  root((储能收益优化实验全景))
    当前主线
      点电价预测
        LightGBM ensemble
        预测 96 点电价曲线
        枚举 3321 个合法充放电窗口对
        选择预测价差最大 pair
      调度约束
        充电连续 8 点
        放电连续 8 点
        放电必须在充电之后
        不重叠
      当前冠军
        ens_champion_segmented6_prior_5fold [冠军][线上 5482]
        champion prior dispatch
        5 fold 验证
        线上强于 holiday_only

    基础模型路线
      LightGBM baseline
        lgb_baseline
        lgb_baseline_last_180d
        lgb_baseline_last_90d [拒绝]
        business_last_180d [早期提升]
      分时段 LightGBM
        lgb_segmented_6_last_180d [候选]
        lgb_segmented_6_margin_core_last_180d [稳定性候选]
        lgb_segmented_6_business_last_180d [拒绝]
        selective_business_segmented [拒绝]
      模型家族对比
        CatBoost
          cat_baseline
          cat_baseline_last_180d [不如冠军]
        XGBoost
          xgb_baseline [拒绝]
          xgb_baseline_last_180d [拒绝]
        Equal ensemble
          lgb180_cat180 [不如冠军]
          base_base180_cat180 [不如冠军]
      正则与浅树
        shallow_d3_d4_d5 [拒绝]
        max_depth / num_leaves 控制过拟合
        结论：浅树没有改善冠军

    特征工程路线
      原始边界预测值
        系统负荷预测
        风光总加预测
        联络线预测
        风电预测
        光伏预测
        水电预测
        非市场化机组预测
        结论：删特征没有找到强噪声特征
      竞价空间与业务特征
        bid_space
          负荷 - 风光 - 联络线 - 水电 - 非市场化机组
          全局入模未显著提升
          作为辅助诊断仍有价值
        net_load
        renewable_ratio
        non_market_ratio
        tie_line_ratio
        margin_core [单独不够强]
      天气 NWP
        全网格聚合 NWP [拒绝]
        core weather [拒绝]
        NWP interactions [拒绝]
        天气与风光误差诊断
          GHI 主要反映日内/季节规律
          风速残差更像真实修正信号
        weather correction [不提升收益]
        结论：物理预测更准不等于窗口选择更准
      时间与日历
        month / hour / slot
        holiday_only [线上 5370][本地冬季强但线上弱]
        low weight holiday blend [拒绝]
        weekly_relative / weekly_delta [拒绝]
        capacity / cap_only [拒绝]
        lag / weekly lag [已试，整体不稳]
      预测误差增强
        forecast_error_augmentation [拒绝]
        feaug_ws025 / late050 [拒绝]
        结论：减少部分亏损日但均值和尾部 regret 变差

    目标与训练策略
      绝对价格目标
        当前主线
        适合直接预测价差
      形状目标
        zscore / rank [早期尝试]
        centered target [拒绝]
        结论：形状更像不等于窗口 pair 更准
      鲁棒损失
        L1
        L2
        Huber alpha 0.5 / 0.7 / 1.2
        结论：没有超过 champion
      训练窗口
        all past
        last 180d
        last 90d [拒绝]
        winter weight [局部有信号但不稳]
      冬季与线上分布
        Jan-Feb-like 验证
        test-like weighted
        online risk dashboard
        结论：Jan-Feb-like 不能单独作为提交标准

    窗口与 E2E 路线
      窗口均价模型
        89 个 2 小时滑动窗口
        window_mean_last_180d [拒绝]
        问题：窗口均价还可以，但极值窗口排序不够准
      pair spread 模型
        合法充放电 pair 价差预测
        pair_spread_last_180d [拒绝]
        pair_spread_bidspace_weather_resid_5fold [拒绝]
        问题：全局替代点预测不如当前冠军
      E2E / 直接策略
        方向有讨论
        当前没有超过点预测 + 枚举调度
        后续只适合作为 reranker 辅助

    调度与策略路线
      基础调度
        每天都交易
        选择预测价差最大合法 pair
      prior dispatch
        current champion prior
        明显增强当前冠军
      no trade / tau
        简单不交易阈值 [拒绝]
        原因：过滤掉赚钱日，没有精准过滤亏损日
      topK 诊断
        champion top10 oracle +737.6/日 [诊断]
        exact oracle pair 很少直接在 top10
        说明问题是 rerank / selector
      topK reranker
        learned reranker [拒绝]
        fold-safe NWP reranker [拒绝]
        weather_bidspace_topk_aux [拒绝]
        conservative_switch [候选但低于提交阈值 +25.7/日]
      坏日处理
        bad_day_action_not_no_trade [拒绝]
        动作包括 rank2 / rank3 / consensus / prior-disabled / aux
        结果 +0.86/日，不够
      模型家族 fallback
        family top1 oracle +982.4/日 [诊断]
        learned fallback -56.8/日 [拒绝]
        结论：不同模型有上限，但简单规则学不会哪天该信谁

    诊断与风控工具
      daily error 复盘
        高 regret 日
        亏损日
        charge gap / discharge gap
      特征重要性
        gain / split importance
        permutation importance
        删特征回测
      bid_space 诊断
        预测 bid_space vs 实际 bid_space
        实际 bid_space vs 实际电价
        月份 / 时段热力图
        bid_space 误差 vs 坏日
      天气对齐审计
        UTC + 8 小时
        lead_time 对齐边界条件
        防止错天匹配
      在线风险看板
        champion 5482
        holiday_only 5370
        all_5fold / Jan-Feb-like / test-like weighted 对比

    协作与工程 harness
      GitHub 协作
        本地开发
        GitHub 同步
        AutoDL 拉取运行
      固定命令入口
        make backtest
        make submit
        make tune
      实验台账
        experiment_index.csv
        experiment_map.csv
        matrix_experiment_plan.md
        online_risk_dashboard.md
      质量门禁
        fold-safe 检查
        数据泄露检查
        submit 格式校验
        lint / test / backtest

    当前结论
      保留冠军为锚点
        不轻易全局替换 champion
        新方案必须证明线上风险更低
      已经不建议继续加复杂度
        简单 bad-day gate
        简单 model-family fallback
        全局天气入模
        全局 holiday / weekly blend
      仍有潜在空间
        top10 oracle 空间大
        family oracle 空间大
        关键是学会何时换 pair
      下一步更合理方向
        冬季 / test-like 分布导向局部专家
        更强的可见风险信号
        只做少量高置信局部修正
```

## How To Read

- `当前主线` 是目前最重要的生产路径：点电价预测 + 合法 pair 枚举 + prior dispatch。
- `基础模型路线` 记录换模型家族和集成的尝试。
- `特征工程路线` 记录业务变量、天气、日历、滞后、预测误差增强。
- `目标与训练策略` 记录换 target、loss、训练窗口等实验。
- `窗口与 E2E 路线` 记录从“预测点电价”切到“预测窗口/价差/策略”的尝试。
- `调度与策略路线` 记录不改价格模型，只改选 pair 或是否交易的尝试。
- `诊断与风控工具` 是我们用来解释为什么本地和线上不一致、为什么某些日子亏损的工具。
- `当前结论` 是行动建议：不要继续盲目扩模型，重点找能解释高 regret 日的可见信号。
