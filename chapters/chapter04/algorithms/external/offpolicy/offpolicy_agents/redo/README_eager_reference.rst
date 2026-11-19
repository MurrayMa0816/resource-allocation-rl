SAC+ReDo：Torch eager 参考实现
================================

来源与配置
----------

来源为 ICML 2023 论文 The Dormant Neuron Phenomenon in Deep Reinforcement
Learning，作者代码 google/dopamine，固定提交
5873f5494ee0c2d7c016d0ab2ad530354fec59d0，Apache 2.0。来源、许可和移植差异见
PROVENANCE.json。本说明对应 eager 参考路径。

scientific_profile.json 固定 warmup=10000、RR=8、总训练步数 49152，预期执行
313216 次完整 SAC 迭代。作者采用 pre-increment 时钟，首次回收在
update=200001、env=35001；周期为 200000 次梯度迭代，该预算预期回收一次。
网络为两层 256×256 ReLU，gamma=.99、alpha=1、target_entropy=-A/2、tau=.005、
batch=256、lr=.0003，采用 TF Adam epsilon-hat=1e-7。原终奖不变，任务适配使用
learner reward_scale=1。

回收与更新
----------

回收覆盖 actor 和两个在线 critic 的全部 hidden 层。score 是同一训练 batch
上 mean(abs activation) 除以同层平均值加 1e-9，阈值为 <=0；critic 和 actor
分别使用各自更新前的激活。先固定全部层的 mask，再重采全部入边 Glorot 权重并
清零 bias，最后清零全部出边。kernel 的 Adam m/v 按 mask 清零，bias slots 和
优化器全局时钟保留。actor 的 mean/logstd 合并输出头均被最后 hidden 层出边覆盖。

target、alpha 和 replay 不直接重置。每次更新顺序为 critic→actor→ReDo→
用更新后 actor 重新采样计算 alpha→counter+1→target Polyak。保留 TF-Agents
的熵目标、logstd 和 Adam eps 位置；Torch 与 TF 随机流及浮点轨迹不保证逐位相同。

接口与保存
----------

Agent(cfg, observation_dim, action_dim, device) 的 act 用于采集；
policy_action(obs, deterministic=False) 绕过均匀预热，评价实际策略且不改变
steps、updates 或 replay。observe(obs, action, reward, next_obs, done) 接收
一条真实转移并返回 steps、updates 和本步 recycle_events。n_updates 统计完整
SAC 迭代，每次含 critic、actor、alpha 各一步 Adam。

diagnostics 包含计数、alpha、loss 和逐层回收信息，包括 dead_indices、mask
条目、清零的 Adam 槽及回收前后 target/replay SHA。评价需隔离 Torch 全局 RNG
和 agent.rng；native_state_fingerprint 可用于检查评价副作用。

save 只保存 actor 推理终点、配置和诊断；load 后不支持 observe 续训。critic、
Adam 和 replay 不落盘，replay 在训练期间保存在 RAM。环境和预算由公共 worker
管理。

检查范围
--------

test_zero_env.py 的 CPU 检查覆盖固定 batch、TF 公式 Adam/SAC、mask 轴、相邻层
回收顺序、bias moment、target/replay、预热时钟及推理重载。强制 update=200001
为合成时钟 fixture，不代表执行了 200001 次更新或 35001 个环境步。

gpu_gate_command.json 给出 CUDA 检查和吞吐测试的参数与环境，覆盖 305/19 和
244/13 两种输入尺寸、CPU/CUDA 数值误差与回收 hook，每域计时 32 次；不创建
环境或保存模型。CPU 合成结果不能替代 CUDA 等价检查或真实任务结果。

已记录的 CPU 固定 batch 耗时约为 32 次 0.70 秒、每次 21.95 ms，按 313216 次
估算约 6874 秒，不包括采样、评价、传输或设备争用，不能用作 CUDA 或完整训练耗时。
预热 10000、总步数 10272 时，完整 SAC 更新数为 2176。实际回收事件、非零 mask
及学习表现应由对应实验记录判断。
