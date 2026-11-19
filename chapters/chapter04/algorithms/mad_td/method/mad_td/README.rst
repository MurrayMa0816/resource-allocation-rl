MAD-TD：第四章有限终奖任务适配
================================

方法与来源
----------

MAD-TD: Model-Augmented Data stabilizes High Update Ratio RL，ICLR 2025。
作者仓库为 adaptive-agents-lab/MAD-TD，固定提交为
685293e222accf1a0f36afcab933ce1902d4452a。发表年按正式 proceedings 和作者
README 的 2025 记载。vendor_manifest.json 登记保留的 59 个原文件及其 SHA；
该提交未提供 LICENSE 文件，不能据此声明代码许可。

实现保留 512 维 SimNorm 编码、512×512 Actor、双 151-bin HL-Gauss critic、
512×512 latent/reward model、DAML/BYOL 辅助目标、Optax Adam、梯度裁剪、
目标更新和 vmaped_utd_update。critic 的每批输入由 486 条真实回放与 26 条
使用当前策略动作生成的模型转移组成。

固定配置与任务适配
------------------

随机预热为 5000 个真实环境步，从第 5001 步起每次交互执行 UTD8。
slow_model=True，model 和 encoder 每轮只在第一次更新，critic 和 actor 各更新
8 次。batch=512，RL/model 学习率为 3e-4，encoder 学习率为 1e-4，clip=10，
不重置网络。tau=.99 是目标保留系数；与论文表中 .995 的差异见
scientific_profile.json。MPC 关闭。探索使用作者的均匀预热和均值动作加高斯噪声，
sigma=.1、噪声截断为 ±.3。

任务固定时域 H，只在真实终点返回奖励。适配后的价值目标为
r + mask * gamma * nextQ，终点仍监督 critic、encoder 的 critic 分支和 reward
model；BYOL/DAML 的终点下一 latent 损失继续由 mask 屏蔽。生成动作使用源转移
对应的 terminal mask，因为终止由公开固定时钟决定。

gamma=1，真实奖励不缩放。两域统一使用 HL-Gauss 支持区间 [-1,1]、151 bins，
保留编码和 zero prior。区间由任务终奖的公开边界确定，来源与固定 H 的源码 SHA
见 scientific_profile.json。这是有限终奖任务适配，不是原 DMC 数值配置的逐位复现。

作者驱动中 5% 的在线交互随机进入内部诊断 replay，此部分不用于网络更新，也不
作为公共开发或独立测试面板。随机分流、回放抽样和 JAX key 可保存、恢复，以隔离
评价。每个决策对应一次真实环境交互，无 DMC 动作重复；公开 vector 状态对应
19/13 维连续动作。实现不使用外部离线或专家数据。

接口与依赖
----------

Agent(cfg, observation_dim, action_dim, device) 接受 seed、total_steps、domain
及固定的 gamma=1、reward_scale=1、value_support=[-1,1]。提供 act、
policy_action、observe、diagnostics、get_rng_state、set_rng_state、
native_state_fingerprint、save 和 load。确定性策略直接执行 encoder→actor。

推理终点使用 msgpack，仅保存 encoder/actor 参数、构造参数和计数，不包含
critic、model、Adam 或 replay，不能用于精确续训。

运行依赖为 JAX、Flax、Optax、Hydra、NumPy 和 SciPy。native_loader 只移除
ReplayBuffer 中未执行的 Env 类型注解导入，不加载 DMC、Brax、Gymnax 或 WandB；
get_policy_action 和学习函数来自保留的作者源码。旧 JAX API 别名单独声明。

检查范围
--------

zero_env_gate.py 使用合成数据执行完整尺寸 JAX 更新，检查终奖对 critic/reward
model 的影响、终点目标与 nextQ 的隔离、非终点 bootstrap、原生 UTD8 计数和
推理参数序列化。该检查不创建真实环境，不能替代双任务中的动作、全额分配、终奖、
保存重载及评价 RNG 隔离检查，也不能单独证明学习效果。
