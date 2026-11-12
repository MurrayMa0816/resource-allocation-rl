MAD-TD：第四章有限终奖任务适配
================================

方法与来源
----------
MAD-TD: Model-Augmented Data stabilizes High Update Ratio RL，ICLR2025。
作者仓库adaptive-agents-lab/MAD-TD，固定685293e222accf1a0f36afcab933ce1902d4452a。
repository About的2024与正式proceedings及README引用2025不符，本文按正式发表年2025。
vendor59文件原字节保留，vendor_manifest逐SHA校验；该pin没有LICENSE文件，不虚构许可证。

本实现直接执行作者的512维SimNorm编码、512×512 Actor、双151-bin HL-Gauss
critic、512×512 latent/reward model、DAML/BYOL辅助目标、Optax Adam与梯度裁剪、
目标更新，以及vmaped_utd_update。486条真实回放与26条当前策略动作的模型生成
转移共同训练critic；不是给TD3换一个名字，也不执行百万扩散样本或128步去噪。

固定配置
--------
真实随机预热5000步，5001步起每次新交互执行UTD8；slow_model=True，故model及
encoder仅在8次更新的第一次更新，critic及actor各8次。batch512、RL/model学习率
3e-4、encoder1e-4、clip10；不重置。tau=.99按官方YAML，表示目标保留系数；
不能误当源参数插值权重，论文表中.995差异已在scientific_profile.json声明。
MPC默认关闭，这是作者提供的完整MAD-TD配置；不把未执行MPC说成执行了规划。
训练策略保留作者均匀预热、均值动作加sigma.1且clip±.3的高斯探索。

有限终奖的必要适配
------------------
当前物理任务固定H、只在真正终点返回奖励。作者DMC持续任务路径将mask乘整个
critic损失及reward模型损失，且target没有done。如果直接照搬，就会丢掉唯一终奖。
内存AST适配明确改为r+mask*gamma*nextQ，终点critic/encoder critic和reward模型
保留监督；BYOL/DAML的终点下一latent损失仍masked。同一公开固定时钟决定终止，
所以生成的当前策略动作使用源row的terminal mask合法；没有查询未来或额外奖励。
gamma固定1，真实奖励不缩放。真实任务的Q范围由单一终奖公开边界确定，故在任何
MAD-TD开发结果产生前两域统一登记HL-Gauss[-1,1]、151bins，编码与zero prior保留。
范围来源及固定H源码SHA见scientific_profile.json。该配置不是原DMC数值配置逐位复现。

官方训练驱动将5%已付费在线交互随机送入内部诊断replay，本实现保留。它不参与
网络更新，也不是公共开发或独立终测面板，不使用外部离线/专家数据。随机分流与
原生回放抽样、JAX显式key均可保存/恢复以隔离评价。每真实决策对应一轮物理执行，
没有DMC动作重复。策略接收同一公开精简vector状态，输出19/13维公共连续动作。

接口与工件
----------
Agent(cfg,observation_dim,action_dim,device)，cfg仅接受seed/total_steps/domain及固定
gamma1/reward_scale1/value_support[-1,1]；提供act、policy_action、observe、diagnostics、
get_rng_state、set_rng_state、native_state_fingerprint、save与load。
确定性部署直接encoder→actor，不带采样噪声。终点msgpack只含encoder/actor参数、
构造参数和计数，无critic/model/Adam/磁盘replay/周期模型，不能精确续训。

最小运行依赖为JAX/Flax/Optax/Hydra/NumPy/SciPy。native_loader只移除ReplayBuffer
未执行的Env类型注解import，不加载DMC/Brax/Gymnax及WandB；直接提取作者原样
get_policy_action函数，学习函数仍来自保留的原件。旧JAX API同义aliases单独声明。

zero_env_gate.py执行真实完整尺寸JAX更新，但没有创建真实环境。它要求终奖对
critic和reward模型产生非零参数变化、终点更新不依赖nextQ，原作者mask路径的
负对照确实丢失终奖，同时非终点仍bootstrap；原生UTD8计数和推理参数序列化核验。
GPU门禁须由父任务显式安排远端物理GPU0，不能独自占用设备。正式训练前还需
真实双环境动作/全额资源/终奖/保存重载/评价RNG隔离门禁；代码可编译不等于通过。
所有模型、曲线、日志、门禁和缓存实体放新HY根mad_td_pilot_20261107；维护代码
经现有ROOT comparison/replacement_candidates路径指向canonical HY代码目录。
