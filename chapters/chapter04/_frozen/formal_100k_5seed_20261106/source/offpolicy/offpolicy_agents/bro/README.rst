BRO (Fast)：作者原生JAX核心的资源配置接口
======================================

来源与身份
----------
对应NeurIPS2024论文Bigger, Regularized, Optimistic: scaling for compute and
sample-efficient continuous control。论文官方PDF为
https://papers.neurips.cc/paper_files/paper/2024/file/cd3b5d2ed967e906af24b33d6a356cac-Paper-Conference.pdf。
作者仓库https://github.com/naumix/BiggerRegularizedOptimistic，固定commit
5870a13d4bf4f66eeddfc291a59df6d2009de600，MIT许可原文保留。vendor_manifest.json
逐SHA绑定17份原文件，所有vendor字节不改。BRO(Fast)是论文和作者README明确给出的
RR2正式配置；本实现没有把自创缩小网络版本重新命名成BRO。

采用完整jaxrl/bro核心，不采用删去乐观策略和重置的bro_minimal。保留BroNet残差与
LayerNorm、AdamW、两个100分位critic、部署actor与乐观采集actor、KL目标/自适应
乐观系数/正则系数、温度学习以及完整状态重置。作者原网络没有缩小；本任务在线双critic
参数量为通信2546888、应急2478280，target分别同量，另有独立双actor。

固定原生配置
------------
每环境步RR2、batch128、replay容量1000000，actor/critic/temp学习率3e-4，
optimism/KL regularizer学习率3e-5，gamma=.99、tau=.005、初温1、目标熵-A/2、
KL目标.05、乐观actor标准差乘数.75、pessimism=0。任务奖励缩放1，保持原终奖。

作者循环从i=1开始，i<2500时uniform采集；i=2500以乐观策略采集并首次执行两次更新。
每步先插入真实转移、抽取两批，再调用原生update。Fast只有env15001一个完整reset，
发生在本步转移已经收集和插入之后、两次更新之前。reset归初态的对象包括部署actor、
乐观actor、critic、target、温度、optimism、KL regularizer、所有Adam/AdamW和原生PRNG；
历史replay保留。adapter只在原生reset外包一层哈希审计，不自行重复reset。

49152真实训练步的预期为93306次完整BRO迭代；reset前累计25002，reset后周期68304，
最终native step68305。各完整迭代包含部署actor/乐观actor/critic/target/温度/乐观/KL
参数各自原生更新，不能把这个计数与其他算法的一个SGD步直接等同。

评估与公共接口
--------------
Agent(cfg, observation_dim, action_dim, device)提供act、policy_action、observe、
n_updates、diagnostics、native_state_fingerprint、get_rng_state/set_rng_state和save/load。
接口只接公开obs、执行的原生低维action、原reward、next_obs和真实done，不建立物理Env。
输入305/19与244/13的原始公开向量，在原生SinglePrecision边界转float32。

act训练中先uniform、后使用乐观采集策略。policy_action始终消费部署用pessimistic actor，
不会返回uniform warmup或采集策略。deterministic=True对应作者temperature=0评价；
False显式保留随机部署策略的诊断能力。本轮root协议采用True，真DEV0与后续保持一致，
不混入此前stochastic评价批次作同配方排名。DEV后须恢复原生JAX key与Box RNG；公共
worker另行保存恢复全局Python/NumPy/Torch RNG，replay抽样继续用作者全局NumPy RNG。

save只写一个部署actor的Flax msgpack推理终点、PRNG及必要元数据，不写critic、Adam、
乐观actor或replay。文件名即使为.pt也不是Torch pickle。load仅建立已有部署actor，
不会重新初始化大网络或分配学习回放；observe会拒绝，不能声称支持精确续训。

依赖与兼容边界
--------------
原作者依赖JAX0.4.19/Flax0.7.5/TFP-nightly，当前准备复用本机JAX0.4.35、Flax0.8.5
环境。不是2024原软件逐位复现。load_native仅跳过未使用的模拟器导入，并使用Flax
支持的flax_return_frozendict=False维持BRO原Model.create的dict.pop语义；不能照搬
旧SR-SAC的True开关。原生数学与模型定义不改。独立进程使用私有jaxrl包，若已有别的
作者jaxrl模块在进程中加载会拒绝，防止混用SR/BRO源码。

门禁与计算成本
--------------
zero_env_gate.py为纯合成CPU/CUDA门禁，不调用环境：比较原生克隆与adapter完整7组
参数/优化器/PRNG；独立NumPy量化Huber loss；终末下一状态扰动不变；真实首更新索引；
完整reset/历史回放保持；DEV RNG恢复；推理保存重载与禁止训练。

reset采用明确的15000/25002计数fixture，不能冒称执行过15001个环境步或25002次更新。
每域实际2501条合成转移、8次完整BRO迭代（含独立native参考）及2次单独critic更新；
双域合计5002合成转移、16完整迭代、4critic更新、0真实环境步。临时推理fixture经过
重载/哈希后由临时目录清理；真实工程与科学任务仍各保一个固定终点。

CPU门禁由本实现代理执行；GPU门禁命令在gpu_gate_command.json，必须由root按当前
资源统一准入后执行，不能把CPU检查说成真实CUDA验收。没有自行GPU或科学训练。
