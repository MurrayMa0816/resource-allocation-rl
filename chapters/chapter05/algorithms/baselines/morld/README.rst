MORL/D
======

本目录提供 MORL/D 的资源配置任务适配。第五章采用 ``critic_lr1e4`` 配置；
对应模型见章级 ``models/index.json``，实际运行参数见 ``configs/recorded/configs/``。

配置与接口
----------

* Actor 学习率为 0.0003，Critic 学习率为 0.0001。
* 三个独立 MOSAC 成员共享 RAM 回放，使用 PSA；每个训练回合后对其他成员执行 10 遍更新。
* 部署使用初始偏好为 [0.8, 0.2] 的成员，不按开发集峰值选择检查点或成员。
* 无线任务使用 n_step32，应急任务使用 n_step1；内部策略评价的环境交互计入学习预算。

参数定义在 ``configs.json``。接口为
``Agent(obs_dim, action_dim, seed=0, device='cpu', config=None)``，
通过 ``act`` 和 ``update`` 完成交互与更新。
配置检查会拒绝与 ``critic_lr1e4`` 不一致的学习率和更新节奏。

验证范围与来源
--------------

``validate_cpu.py`` 检查 CPU 接口，不创建物理环境或保存模型。
``selected_implementation.json`` 中的参考验证覆盖 v5 种子 9840/9841、
98304 个计费交互步和 193 个 DEV 点，未包含该记录下的无线学习验证或独立 TEST。
论文采用的检查点与结果以章级模型和结果索引为准。

算法来源为 Felten、Talbi、Danoy 的 JAIR 2024 论文
``Multi-Objective Reinforcement Learning Based on Decomposition: A Taxonomy and Framework``，
DOI ``10.1613/jair.1.15702``。固定源码提交与 MIT 许可见 ``provenance.json`` 和 ``vendor/``。
原论文 Critic 学习率为 0.001；本任务配置采用 0.0001。
