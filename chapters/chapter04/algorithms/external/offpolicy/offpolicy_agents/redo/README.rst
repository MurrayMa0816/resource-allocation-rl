SAC+ReDo：Torch 核心与 steady_graph 执行
========================================

来源与配置
----------

来源为 ICML 2023 论文 The Dormant Neuron Phenomenon in Deep Reinforcement
Learning，作者代码 google/dopamine，固定提交
5873f5494ee0c2d7c016d0ab2ad530354fec59d0，Apache 2.0。来源、许可和移植差异见
PROVENANCE.json。eager 核心由 agent.Agent 提供；当前配置使用 steady_graph 执行封装。

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

gpu_gate_command.json 给出 eager CUDA 检查的参数与环境；对应实际结果已登记于
DELIVERY.json 的 root_eager_GPU_gate。检查覆盖 305/19 和 244/13 两种输入尺寸、
CPU/CUDA 数值误差及回收 hook，不创建环境或保存模型。

已记录的 CPU 固定 batch 耗时约为 32 次 0.70 秒、每次 21.95 ms，按 313216 次
估算约 6874 秒，不包括采样、评价、传输或设备争用，不能用作 CUDA 或完整训练耗时。
预热 10000、总步数 10272 时，完整 SAC 更新数为 2176。实际回收事件、非零 mask
及学习表现应由对应实验记录判断。

steady_graph 执行
----------------

cfg.execution_variant='steady_graph'，worker 使用
replacement_candidates.redo.graph_agent.Agent。封装仍调用原 _update_batch，
只改变一组 8 次既有更新的执行调度，不改 Tensor 公式、beta、epsilon、学习率、
时程或 RNG 算法；agent.py 的原 eager 来源指纹为 e3f4c13f。

前 40000 次 optimizer 迭代使用 eager；随后 Python double 偏差修正已 round 为 1，
仅不跨回收边界的 8 次更新块使用 capture/replay。跨 update=200001 的整块仍执行
eager，env=35001 回收后恢复使用图，首次 graph 约在 env=15001。313216 次有效
迭代分为 eager 40008 次与 graph 273208 次，后者为 34151 块。

图准备另含 8 次可逆 warmup，恢复参数、slots、梯度、时钟和 RNG；另有 8 次 Python
capture trace，均不计为有效学习或环境步。diagnostics.execution 单独记录
warmup、capture、graph/eager 块数及执行开销。

steady_graph_gate.py 的既有结果记录两域参数、Adam m/v、梯度、loss、全局
CPU/CUDA RNG、回收 mask、target 和 replay 与 eager 路径逐位一致。局部固定
batch 核心加速为 8.7888/8.5381 倍，不是完整训练倍率。对应记录为
redo/steady_graph_gate.json，SHA256 为
002e989cb2c43b4d10acff2858d8989024bfa79894f2794ea9a550451e2c69c7。

DELIVERY.json 与 source_manifest.json 绑定当前核心、封装和检查记录。
README_eager_reference.rst 说明 eager 路径，source_manifest_eager_reference.json
保存相应来源身份，不替代当前执行配置。
