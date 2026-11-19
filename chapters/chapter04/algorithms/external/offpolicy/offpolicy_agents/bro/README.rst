BRO (Fast)：原生 JAX 核心的资源配置接口
========================================

来源与实现
----------

来源为 NeurIPS 2024 论文 Bigger, Regularized, Optimistic: scaling for compute
and sample-efficient continuous control。论文：
https://papers.neurips.cc/paper_files/paper/2024/file/cd3b5d2ed967e906af24b33d6a356cac-Paper-Conference.pdf。
作者仓库为 https://github.com/naumix/BiggerRegularizedOptimistic，固定提交
5870a13d4bf4f66eeddfc291a59df6d2009de600，MIT 许可。vendor_manifest.json
登记保留的 17 个原文件及 SHA。Fast 对应作者的 RR2 配置。

实现使用完整 jaxrl/bro 核心，包括 BroNet 残差和 LayerNorm、AdamW、两个
100 分位 critic、部署 actor、乐观采集 actor、KL 目标、自适应乐观/正则系数、
温度学习和完整状态重置。保持作者网络尺寸；通信和应急的在线双 critic 参数量
分别为 2546888 和 2478280，target 同量，另含独立双 actor。

固定配置
--------

每环境步 RR2，batch=128，replay 容量 1000000。actor/critic/temp 学习率为
3e-4，optimism/KL regularizer 学习率为 3e-5；gamma=.99、tau=.005、初温=1、
目标熵=-A/2、KL 目标=.05、乐观 actor 标准差乘数=.75、pessimism=0。
任务奖励缩放为 1，原终奖保持。

作者循环从 i=1 开始，i<2500 时均匀采集；i=2500 起使用乐观策略并执行每步两次
更新。每步先插入真实转移、抽取两批，再执行原生 update。Fast 在 env=15001
完成一次 reset，位于本步转移插入后、两次更新前。重置对象包括双 actor、
critic/target、温度、optimism、KL regularizer、全部 Adam/AdamW 状态及原生
PRNG；replay 保留。适配器仅记录 reset 指纹，不重复重置。

49152 个真实训练步对应预期 93306 次完整 BRO 迭代：reset 前累计 25002 次，
reset 后 68304 次，最终 native step=68305。每次迭代更新部署/乐观 actor、
critic、target、温度、乐观和 KL 参数；该计数不能直接等同其他算法的一步 SGD。

接口与依赖
----------

Agent(cfg, observation_dim, action_dim, device) 提供 act、policy_action、
observe、n_updates、diagnostics、native_state_fingerprint、get_rng_state、
set_rng_state、save 和 load。输入为公开 obs、原生动作、reward、next_obs 和
done，不创建物理环境；305/19 与 244/13 的公开向量在原生 SinglePrecision
边界转为 float32。

act 使用均匀预热和乐观采集策略；policy_action 使用部署 actor。
deterministic=True 对应作者 temperature=0，评价协议采用此设置；False 保留
随机部署策略诊断。两种评价结果应分别标记。评价需恢复 JAX key、Box RNG 以及
公共 worker 的 Python/NumPy/Torch RNG；回放抽样沿用作者的 NumPy 全局 RNG。

save 保存部署 actor 的 Flax msgpack、PRNG 和必要元数据，不含 critic、Adam、
乐观 actor 或 replay；扩展名 .pt 也不表示 Torch pickle。load 只建立推理 actor，
不分配学习回放，observe 会拒绝续训。

作者软件依赖为 JAX 0.4.19、Flax 0.7.5 和 TFP-nightly；本适配使用 JAX 0.4.35、
Flax 0.8.5，不保证与原软件栈逐位一致。load_native 跳过未使用的模拟器导入，
使用 flax_return_frozendict=False 保持 Model.create 的 dict.pop 语义。
独立进程使用私有 jaxrl 包，拒绝与其他已加载的 jaxrl 实现混用。

检查范围
--------

zero_env_gate.py 使用合成转移比较原生实现和适配器的参数、优化器、PRNG、
量化 Huber loss、终点处理、首更新索引、reset、replay 保留、评价 RNG 恢复和
推理重载。reset 的 15000/25002 计数使用 fixture；每域记录为 2501 条合成转移、
8 次完整 BRO 迭代及 2 次独立 critic 更新，两域合计 5002 条、16 次及 4 次，
真实环境交互为零。这些计数不代表对应规模的真实训练。

CUDA 检查命令见 gpu_gate_command.json。CPU 合成检查不能替代 CUDA 检查或
真实环境中的学习效果验证。
