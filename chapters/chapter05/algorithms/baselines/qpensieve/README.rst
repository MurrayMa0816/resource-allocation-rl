Q-Pensieve
==========

本目录提供 Q-Pensieve 的资源配置任务适配。第五章采用 ``legacy_recipe`` 配置；
对应模型见章级 ``models/index.json``，实际运行参数见 ``configs/recorded/configs/``。

配置与接口
----------

入口为 ``from baselines.qpensieve import Agent, load_config``。
论文对应配置通过 ``load_config(domain, profile='legacy_recipe')`` 读取，
``domain`` 为 ``radio`` 或 ``v5``。

``Agent(obs_dim, action_dim, seed=0, device='cpu', config=None)`` 提供 ``act``、``update``
和 ``observe_training``。观察归一化由 Agent 管理，``observe_training`` 每个真实训练步
调用一次，不叠加第二层归一化；开发集评价不更新训练统计。
自有随机状态通过 ``rng_state_dict`` 保存，公共 Python、NumPy、Torch、CUDA 随机状态
由训练程序保存和恢复。紧凑检查点不包含优化器和回放，不能用于精确续训。

``configs.json`` 另列出 ``fixed`` 配置，其区别为 Xavier-uniform 权重和零偏置初始化。
两种配置的初始权重不同，不能视为相同初始策略下的数值配对；初始化修正本身也不构成
学习效果改善或与上游源码逐位等价的证据。

来源
----

来源为 ``https://github.com/rl-bandits-lab/Q-Pensieve``，固定提交
``07385086539126db74284402d798dd89ce04ff0b``。MIT 许可证保存在 ``vendor/LICENSE``，
实现来源与任务适配范围见 ``provenance.json``。
