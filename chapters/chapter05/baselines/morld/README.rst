MORL/D：第五章唯一当前实现
==========================

用户已选定橙色曲线对应的critic_lr1e4。后续训练、评价和对比只使用这一配置，
不再提供其他当前候选版本。显示名称统一为MORL/D。

固定设置
--------

* Critic学习率0.0001，Actor学习率0.0003。
* 三个独立MOSAC成员、共享RAM回放、PSA，每个训练回合后执行其他成员10遍更新。
* 固定部署初始偏好[0.8,0.2]的成员，评价不挑选历史峰值或其他成员。
* 原生MOSAC、轮转和PSA内核及MIT许可证保持；完整内部评价的交互继续计入预算。
* 无线n_step32、应急n_step1的已有任务适配保持，物理环境、原奖励和全额分配不改。

唯一配置在configs.json，各域仅有critic_lr1e4；Agent默认也采用这一设置。
冲突的学习率、旧profile、零额外更新或cadence16会明确报错，不能静默退回旧版。
新模型记录选定配置身份并使用独立格式，避免把旧模型误标为当前版。

使用与验证
----------

Agent(obs_dim, action_dim, seed, device, config)。configs.json给出当前配方，
seed、设备、输出、来源冻结和训练/开发/测试主种子仍须在新运行协议中显式登记。
不恢复旧队列，不在本文件中隐式启动训练。每512步评价，保留完整原始数据。

CPU工程检查不创建物理环境、不保存模型::

    python validate_cpu.py --output /hy-tmp/<new-path>/selected_cpu_gate.json

真实已测橙色证据是应急v5的两个种子9840/9841、98304收费步、193个DEV点。
无线采用同一当前实现和Critic学习率，但尚未新增该学习率的无线训练证据；
工程API检查不能当成无线学习效果或独立终测通过。选定记录及原模型身份见
selected_implementation.json。既有橙色模型按各自原冻结源码和config/SHA读取，
不改写模型格式或把其他模型重标为橙色版本。

来源
----

Multi-Objective Reinforcement Learning Based on Decomposition: A Taxonomy and Framework，
Felten/Talbi/Danoy，JAIR2024，DOI 10.1613/jair.1.15702。
作者固定提交、原论文参数与许可见provenance.json及vendor/。原论文Critic学习率
0.001的来源记录保留；本任务唯一选定值为0.0001，明确属于任务配置。
