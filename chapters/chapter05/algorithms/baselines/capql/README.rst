CAPQL
=====

本目录提供 CAPQL 的资源配置任务适配。第五章采用 ``fixed`` 配置；
对应模型见章级 ``models/index.json``，实际运行参数见 ``configs/recorded/configs/``。

配置与接口
----------

入口为 ``from baselines.capql import Agent, load_config``。
``load_config(domain, profile='fixed')`` 从 ``configs.json`` 读取 ``radio`` 或 ``v5`` 参数。
``fixed`` 与 ``legacy_recipe`` 使用相同的数值配方。

``Agent(obs_dim, action_dim, seed=0, device='cpu', config=None)`` 提供 ``act``、``update``
和 ``observe_training``。观察归一化由 Agent 管理，``observe_training`` 每个真实训练步
调用一次，不叠加第二层归一化；开发集评价不更新训练统计。

自有随机状态通过 ``rng_state_dict`` 保存。训练程序还需保存和恢复公共
Python、NumPy、Torch、CUDA 随机状态。紧凑检查点不包含优化器和回放，不能用于精确续训。

来源与使用范围
--------------

来源为 CAPQL 作者仓库 ``https://github.com/haoyelu/CAPQL``，固定提交
``0ef02f3d3b3d0c512d282cd5ec0f00475349beee``；文件来源记录见 ``provenance.json``
和 ``SOURCE_RELEASE.json``。该上游快照未声明根代码许可证，本目录不另行指定许可证。

工程接口检查不替代物理环境学习实验；学习表现以章级结果索引所列记录为准。
