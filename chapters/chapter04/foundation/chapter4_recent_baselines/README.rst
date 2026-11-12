第四章：近期样本效率基线与统一实验入口
====================================

当前目标是保持第三章的资源配置任务、公开观测和终点评价，通过更有效利用
真实交互数据，用更少的新环境交互达到预先规定的配置质量。本目录准备
PROPS、FoG、ADDQ、TD-MPC2、PGR 五条比较路线，不预设其中哪条更好。

当前只进行了来源审查、接口检查与有限 smoke 更新，没有启动本批方法的
正式多种子长训练，也没有获得五方法与 A/F1 的正式性能排名。
最终五方法的双环境原生 GPU smoke 与统一入口预检全部通过，已达到可开始
训练状态；实际验收以逐方法 receipt 和结果根 readiness_summary.json 为准。
就绪验收也不等于方法已经证明优于 F1。

最终五方法及固定官方来源
------------------------

.. list-table:: 固定仓库、版本、许可与使用的核心
   :header-rows: 1
   :widths: 12 17 18 12 41

   * - 方法 / 正式发表
     - 官方仓库
     - 固定提交
     - 许可事实
     - 本地官方核心（相对本目录）
   * - PROPS / TMLR 2026
     - https://github.com/NicholasCorrado/PPOROS
     - a0c8708529047f85dcf98f90d5ea264dd43aeba2
     - 仓库未声明 LICENSE
     - vendor/props/PROPS/ppo_props_continuous.py；直接使用 update_ppo 与 update_props
   * - FoG / ICML 2025
     - https://github.com/nothingbutbut/FoG
     - de74db3a295971c9edbdda31efae3b35b8111c4f
     - 仓库未声明 LICENSE
     - vendor/fog/jaxrl/agents/obac/obac_learner.py；weighted replay 的 ER decay、OBAC 更新与 network growth
   * - ADDQ / ICML 2025
     - https://github.com/BommeHD/ADDQ
     - 60f91bcab1fc53d602d8631b98bc639556393166
     - MIT
     - vendor/addq/Deep/rl_zoo3/custom_algos/adqrsac/adqrsac.py 和 policies.py；连续动作 ADQRSAC
   * - TD-MPC2 / ICLR 2024
     - https://github.com/nicklashansen/tdmpc2
     - 332f19e4aba4bffe2dc736f45ea7f7bdc585903a
     - MIT
     - vendor/tdmpc2/tdmpc2/tdmpc2.py、common/world_model.py、common/buffer.py
   * - PGR / ICLR 2025
     - https://github.com/renwang435/pgr
     - b52b41b681f67608f8a5678211ce6cd5d933f9cc
     - MIT
     - vendor/pgr/synther/online/online_cond.py、redq_rlpd_agent.py、diffusion/elucidated_diffusion.py

PROPS 使用作者明确提供行为策略更新的 PPOROS 实现，而不是只有部分参数、
缺少完整行为更新的另一个 PROPS 仓库。其“作者公开”与“明确开源许可”是
两件事；PROPS 和 FoG 均记录为许可未声明，不将它们写成 MIT。其他仓库保留原 LICENSE
与归属，来源和适配记录随各方法 receipt 保存。

TD-MPC2 固定的是 2026-10-20 官方维护版本，含 episodic/termination 支持，
不是声称逐字复现 2024 首发提交。本任务启用 episodic 和 gamma=1，保留
世界模型、序列回放、MPPI 及优化更新。PGR 的 REDQ gitlink 也固定为
``7b5d1bff39291a57325a2836bd397a55728960bb``，位于 vendor/pgr/synther/REDQ，
来源校验同时检查主仓库和该子模块。ADDQ 使用作者的 ADQRSAC 主版本，
不是普通 SAC 或其他 qrsac 消融换名。

FoG 的类名仍是 OBACLearner，但其完整方法必须同时包含经验回放权重衰减、
OBAC 更新和 critic 网络增长调度；不能只运行早期普通更新便记为 FoG。
有限机制检查须分别记录 decay、growth、BC 开启和 reset 的实际触发次数。

BRO 和 CrossQ 均保留为已检查的备用实现，不计入最终五方法。BRO 官方仓库为
https://github.com/naumix/BiggerRegularizedOptimistic，提交
``5870a13d4bf4f66eeddfc291a59df6d2009de600``，MIT，位于
``vendor/bro/jaxrl/bro/bro_learner.py``。旧 93 区灌溉第三章曾将 bro_full
登记为比较方法，且有 162 步训练、81 步评价与 checkpoint 恢复的实际工程
smoke 记录；虽不等于正式长训练，仍按本次严格历史不重复口径排除。

CrossQ 的官方仓库为
https://github.com/adityab/CrossQ，固定提交
``a318a266679bec3f9d93c44e029860a6da077c89``，MIT，位于
``vendor/crossq/sbx/sac/sac.py``。历史旧洪涝第三章规范提及旧 CrossQ 模型、
诊断及验收；为避免用户要求的不重复存在歧义，本轮最终名单以 PGR 替代它。
不能将 CrossQ 描述成“历史从未使用”或“只有占位”。

第二三章排除口径
----------------

当前 radio/v5 正式第二三章已使用 PASPO、RPO-SAC、TD7、SimBaV2、DIME，
这五项不再作为本批新增方法。普通 PPO、SAC、TD3、SimBa-v1 也不通过改名
加入。MR.Q 另有旧课题第三章实现及旧水资源第四章真实短测，本轮回避；
BRO 和 CrossQ 也按上述历史第三章实际记录排除。FoG 与旧 OBAC 论文身份不同，
且两者的旧第二三章基线记录均未发现；旧 OBAC 记录在第四、第五章。
这些排除基于实际论文/算法身份，不意味着所有 SAC 或 PPO 派生方法均重复。

历史来源复杂，不应将“本轮未在现行第二三章注册”扩写成“从未在任何研究中
实现或运行”。精确范围见结果根的 baseline_exclusions.json 与
inventory_review.rst；研究报告已按后续发现同步修正。

同一环境、观测、动作与奖励
--------------------------

``env_adapter.py`` 从 ``/root/gamewar/chapter23_final/allocation23`` 装载原生
Task，并使用独立 namespace，避免与其他冻结 allocation23 包发生导入冲突。
本目录不修改冻结环境、既有 A/F1 源码、原始图片或灌溉资料。

* radio：100 终端、32 轮、305 维 flat observation、19 维联合潜动作。
* v5：120 区、24 轮、244 维 flat observation、13 维联合潜动作。

神经观测是原 local 展平、global，再追加 remaining/initial_budget 和
budget/initial_budget。追加两项来自 A 自身 quota_head 的公开输入，证据为
``chapter23_final/allocation23/methods/policy.py:100--113``。剩余预算虽在
global 已出现一次，仍保留这两个明确输入，以覆盖同等公开信息。

“精简状态”指去掉动作历史特征，并不表示删除了所有进度：radio 保留公开的
confirmed_packets，v5 保留 remaining_demand 和 total_demand_so_far。它们与
当前 A/F1 一致；不读取未来信道、原图像素、模拟器私有状态或真值公式。

PROPS 使用 raw 的完整联合潜动作。其他连续控制算法使用 [-1,1] 的
``tanh_latent`` 坐标，由 ``atanh(clip(a, ±(1−1e−6)))`` 还原 raw；没有额外
缩放。raw 经同一固定 A_FULL/adaptive 解码器变成物理配置。训练概率或
critic 动作输入对应算法实际记录的潜动作坐标，不能将多对一投影后的
高维物理配置冒充 Gaussian 样本。

物理预算、动作和解码保持 float64；神经输入默认 float32。环境只在真实
回合终点返回原生奖励，gamma=1；非终点奖励为零。所有 info 均剥离，最后
一轮为 true terminated，truncated=False，终点不 bootstrap。无效动作在
执行前拒绝，不消耗真实交互步；审计计数仅随真实 env.step 增加。

评价协议与有效性指标
--------------------

正式开发评价的统一默认值是：step 0 起始评价，以后每 48 个完整训练回合
在固定 32 条 dev 路径上评价随机策略。smoke 可以减小路径数和检查间隔，
必须保留 smoke 标记，不能当成正式成绩。开发评价恢复训练随机流；涉及
JAX、PROPS 行为网络或 MPPI 内部状态时，runner 同时保存并恢复相应状态。

比较横坐标是新增真实训练环境步，包含 warmup，不包含回放抽样和学习模型
内部模拟。开发评价步、梯度步、模型转移计算和墙钟另列，不能把高 UTD 或
规划计算视为免费。

``metrics.py`` 定义以下指标：

* N_q：预先固定阈值 q，连续两次开发评价达标时，第二次评价对应的真实
  训练交互数。未达标种子保持 N_q=None、right_censored=True。
* 达标种子比例随交互预算的曲线，同时保留提前停止造成的不确定范围。
* 共同观测区间上的学习曲线 AUC/区间长度，不虚构 step 0 成绩、不使用
  best-so-far 包络替代原始开发成绩。
* 固定预算末质量、墙钟、计算成本作为必要的补充报告。

主要证据是相同质量少用多少真实交互，以及相同交互预算下的学习曲线。
复用次数、权重、ESS、critic loss 等只能解释机制，单独变好不能证明方法
有效。与 F1 的完整算法比较也不能替代其各组件的匹配消融。

启动方式
--------

从 ``/root/gamewar`` 执行。统一入口会按方法选择隔离的 JAX/Torch 运行时，
避免升级或覆盖当前正式实验的公共依赖。默认 GPU；TD-MPC2 原生实现要求
CUDA。运行时在结果根 runtime/，缺少时先检查 bootstrap.sh 与该目录日志。
bootstrap 只用于重建依赖，不是每次运行前都需执行。

零环境交互的配置/接口检查：

::

    bash chapter4_recent_baselines/run.sh --method addq --domain radio --dry-run
    bash chapter4_recent_baselines/run.sh --method props --domain v5 --dry-run

有限 smoke（真实采集少量完整回合、执行真实更新并验证保存/读取）：

::

    bash chapter4_recent_baselines/run.sh --method addq --domain v5 --smoke
    bash chapter4_recent_baselines/run.sh --method props --domain radio --smoke

正式训练命令示例，本次准备工作没有执行这些命令：

::

    bash chapter4_recent_baselines/run.sh --method fog --domain radio --seed 0 --episodes 960
    bash chapter4_recent_baselines/run.sh --method tdmpc2 --domain v5 --seed 0 --episodes 960
    bash chapter4_recent_baselines/run.sh --method pgr --domain radio --seed 0 --episodes 960

正式方法键为 props、fog、addq、tdmpc2、pgr；bro、crossq 只作为备用入口。可用
``--config /hy-tmp/配置.json`` 覆盖记录在配置中的方法参数，``--output-dir``
只能指向 /hy-tmp。显式 ``--total-steps`` 必须遵守完整回合/方法批次约束，
PROPS 要求四回合目标更新边界；不能通过静默取整隐藏交互成本。

正式比较之前须统一预算、种子、评价路径和参数选择规则；本 README 的
960 回合示例不是性能结论，也不替代最终实验方案登记。

代码核查：

::

    python -m pytest -q chapter4_recent_baselines/tests

保存与已知兼容处理
------------------

官方算法核心保留原版本，接口和计数由 runners/ 负责。ADDQ 官方
ADQRSAC.load 在 deferred-init 路径过早读取 self.critic；本工程提供
``runners.addq_runner.load_checkpoint``：先完整初始化同一官方类，再恢复
SB3 zip 的参数、optimizer 和温度变量。没有改动 ADDQ 更新公式。

各方法必须明确 checkpoint 范围；如标准模型 zip 没保存 replay 和全局 RNG，
不宣称支持任意中断后的精确续训。动作一致性、参数有限和 optimizer/核心
组件变化见逐方法 readiness receipt，不能只凭成功 import 判定实现正确。

结果位置与状态边界
------------------

统一结果根：

``/hy-tmp/resource_allocation_results/chapter4/recent_baselines_ready_20261102``

其中研究、排除名单、环境与指标测试、dryrun、有限 smoke、来源 receipt 和
运行时依赖分开保存。关键文件包括 inventory_review.rst、
baseline_exclusions.json、environment_metrics_verification.json、
addq_readiness.json、pgr_readiness_summary.json、逐方法 receipt、
dryrun/summary.json 与 official_source_verification.json。BRO/CrossQ 的旧 smoke
保留原身份，不参与最终五方法就绪计数。

本次完成的是“可核查来源、与当前环境匹配、通过有限机制与保存/读取检查”的
训练准备。正式比较应以之后独立启动的多种子实验及其有效性指标为准。

最终验收（2026-11-02）：PROPS/FoG/ADDQ/TD-MPC2/PGR 在radio/v5的10组原生GPU smoke全部通过，官方固定来源与10组CLI预检通过，环境/指标测试9项通过。最终状态以结果根readiness_summary.json为准。没有正式性能训练或新五方法排名。

运行速度核查与等价实现优化（2026-11-02）
--------------------------------------

最新报告与代码/证据哈希位于：
``/hy-tmp/resource_allocation_results/chapter4/runtime_audit_20261102/运行速度核查与提速结论.rst``
与 ``optimization_manifest.json``。正式网络/批量的真实更新计时和9项回归已完成。

默认已启用 TD-MPC2 回合边界缓存（可 ``cache_replay_boundaries=false`` 还原）、
FoG 私有 JAX 编译缓存、PROPS 完整JSONL实时日志/最终CSV与集中有限检查、
ADDQ 末尾评估去重和普通Adam计数器恢复、PGR延迟诊断读回/EMA设备遍历一次/采样schedule读回一次
（可 ``safe_runtime_optimizations=false`` 还原）。算法参数、采样量、奖励和保留评估标准未改。

PROPS实时日志现为 ``training.jsonl``，保留每条原始记录；任务结束写同内容的
``training.csv``。具体短对照模型/优化器/物理轨迹和最终CSV均逐位一致。

计时处于共享GPU持续繁忙的负载，不是独占硬件速度。PGR原始长预算成本很高，
正式启动前CLI会打印并保存 ``operation_budget.json``；这只显示计算量，
不自动调整参数、不启动额外任务，也不承诺工期。生产默认设备仍为GPU；
PROPS的CPU/GPU硬件候选实测另存报告，设备浮点路径差异独立披露。
