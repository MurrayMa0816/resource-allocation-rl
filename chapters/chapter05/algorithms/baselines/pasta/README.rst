PASTA
=====

本目录提供 PASTA 的资源配置任务适配。第五章采用 ``current_recipe`` 配置，
模型和实际运行参数分别见章级 ``models/index.json`` 与 ``configs/recorded/configs/``。
第六章的 PASTA 任务适配也引用本目录实现；第六章对比记录包含训练与共同 DEV，
没有与 SeDVal 合做的新增独立 TEST。

配置与接口
----------

接口为 ``Agent(obs_dim, action_dim, seed=0, device='cpu', config=None)``。
``configs.json`` 按 ``configurations[domain]['current_recipe']`` 组织参数。
无线任务学习率为 0.0001，应急任务为 0.0003；适配保留 GAE 的 done[t] 索引和
PCGrad 的 objective-ID 对应关系修正。

Agent 使用 ``on_policy=True`` 和 ``action_mode=raw``。
每个完整回合 panel 调用 ``train_preferences``，每个 tick 通过 ``sample_info``
获取动作、logp 和 values，完整 panel 收集后调用 ``update_rollout``。
轨迹形状为 obs[T,B,O]、actions[T,B,A]、logp[T,B,1]、values/rewards[T,B,2]、
dones[T,B,1]；仅末 tick 的 done 为 1，中间奖励为 0，终奖在末 tick 给出。

B=8，每 8 个 panel 优化一次；无线 T=32，应急 T=24。
首次优化分别发生于 2048 和 1536 个训练步，优化前的平线按真实步数保留。
具体交互预算以对应实验配置为准，不将不同预算的结果混作同一比较。

状态与验证范围
--------------

DEV 可在精确 512 步处执行，评价前后需保持未完成的 panel、训练状态和随机状态一致。
``state_dict`` 用于 RAM 中的完整状态恢复（含 Adam）；固定终点使用
``checkpoint_state_dict``，只保存 actor 和必要的公共观察统计，不含 critic、Adam 或回放。
``manages_observation_normalization=True``，所列配方关闭内部归一化，不叠加外部映射。
CPU 接口检查不能代替实际学习与独立测试结果，具体评价范围以章级结果索引为准。

来源
----

来源为 RSS 2026 PASTA 作者仓库 ``https://github.com/AlejandroMllo/PASTA``，固定提交
``745d769740b6ce648c7ea8209933733dea8a1eab``。``_native.py`` 保留作者源码。
该固定快照未声明根代码许可证，本目录不将论文许可视为代码许可，也不另行指定 MIT 许可。
