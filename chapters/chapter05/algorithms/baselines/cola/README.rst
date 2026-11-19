COLA
====

本目录提供 COLA 的资源配置任务适配。第五章采用 ``fixed_oadm`` 配置，
模型和实际运行参数分别见章级 ``models/index.json`` 与 ``configs/recorded/configs/``。
第六章的 COLA 任务适配也引用本目录实现，其评价范围以第六章索引为准。

配置与接口
----------

接口为 ``Agent(obs_dim, action_dim, seed=0, device='cpu', config=None)``。
``configs.json`` 按 ``configurations[domain][profile]`` 组织参数，
``domain`` 为 ``radio`` 或 ``v5``；``legacy`` 和 ``fixed_oadm`` 仅在
``oadm_one_step_target`` 开关上不同，两个任务均使用 n_step8。

Agent 使用 ``action_mode=tanh_latent``，通过 ``act`` 和 ``update`` 交互与更新。
每个完整并行训练回合调用 ``train_preferences(step, total_steps, rng)``，
生成连续内点偏好；部署评价使用固定偏好 [0.8, 0.2]。

``fixed_oadm`` 的 batch 需提供 ``oadm_next_obs``，即同一转移的真实一步后继观察。
Bellman ``next_obs`` 仍是 n 步后继，累计原奖励和 done 不变。
开启观察归一化时，训练观察拟合后冻结统计，并一致变换 ``obs``、``next_obs``
和 ``oadm_next_obs``；开发集评价不更新统计。无线任务使用前 2048 个随机训练观察拟合映射。

状态与验证范围
--------------

``state_dict`` 不含 Adam 和 replay，不能用作完整续训状态。
开发集评价需保存和恢复 Agent 与公共采集器的随机状态，使用固定训练终点检查点。
n_step1 的更新等价检查只覆盖对应工程接口，不能据此断言 OADM 修正必然提高学习分数。

来源
----

COLA 来源为 ``https://github.com/yeshenpy/COLA``，固定提交
``c437df0ef28169e9356ea54505350ffde13d925a``，MIT 许可见 ``LICENSE.COLA``。
``sac_base.py`` 使用 Q-Pensieve 来源的 SAC 原语，相应 MIT 许可见
``LICENSE.QPensieve_primitives``。
