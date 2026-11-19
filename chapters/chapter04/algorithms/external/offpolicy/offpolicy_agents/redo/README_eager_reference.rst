SAC+ReDo：作者状态连续控制核心的显式Torch移植
==========================================

论文为ICML2023 The Dormant Neuron Phenomenon in Deep Reinforcement Learning。
作者代码google/dopamine，固定5873f5494ee0c2d7c016d0ab2ad530354fec59d0；Apache2.0。
来源/许可/代码差异见PROVENANCE.json。只维护本目录，不改旧封印、F1或其他方法。

主配方固定scientific_profile.json：warmup10000、RR8、49152总真实训练步，
共313216次完整SAC迭代。按作者pre-increment时钟，首次回收发生在
update200001/env35001；回收周期200000梯度迭代而非环境步，预计一次回收。
两层256×256 ReLU，gamma.99，alpha1，target_entropy=-A/2，tau.005，batch256，
lr.0003，TF Adam epsilon-hat1e-7。原终奖不变；learner reward_scale=1为任务尺度适配。

回收同时覆盖actor和两个在线critic的全部hidden层；score为mean(abs activation)
除以同层平均+1e-9，阈值<=0。critic分数取critic更新前、actor分数取actor更新前
的同一训练batch激活。先冻结全层mask，再全部入边Glorot重采/bias清零，最后全部
出边清零。kernel的Adam m/v按mask清零；bias slots及优化器全局时钟保留。
actor输出是合并的mean/logstd头，确保最后hidden出边同时覆盖两个分布参数。
不直接reset target、alpha或replay；target仍在hook后按原SAC执行Polyak。

更新顺序忠于TF-Agents：critic→actor→ReDo→用更新后actor重新采样算alpha→
counter+1→target。没有静默沿用SB3不同的alpha-first顺序、-A熵目标、logstd
截断或Adam eps位置。Torch与TF的随机流和浮点轨迹不承诺逐位相同。

接口
----

Agent(cfg, observation_dim, action_dim, device)，act用于训练采集；
policy_action(obs, deterministic=False)始终评价真实策略，包括DEV0，绕过uniform
warmup，不改steps/updates/replay。observe(obs, action, reward, next_obs, done)
接一条真实transition并返回steps/updates/本步recycle_events。
n_updates计完整SAC迭代；每次含critic、actor、alpha各一步Adam。
diagnostics()含计数、alpha、loss和完整last_recycle_event；event含逐层dead_indices、
mask条目数、已清零Adam槽数，以及target/replay回收前后SHA。
native_state_fingerprint()可用于DEV无副作用检查。随机数只有Torch global和
agent.rng这个公开np.Generator，公共worker应隔离两者。

save(path)只写一个actor推理终点及配置/诊断，load后禁止observe继续训练。
replay全部保存在RAM并贯穿回收；不落盘Adam、critic、replay或周期模型。
算法本身不建立Env、不写日志文件、不接入GPU；由公共worker处理环境和注册预算。

工程门禁与速度
--------------

test_zero_env.py已在CPU1、CUDA隐藏执行固定batch、独立TF公式Adam/SAC、mask轴、
相邻层覆盖顺序、bias moment、target/replay、预热与强制时钟、推理重载门禁。
强制200001时钟是明示合成fixture，不虚报20万实际更新或35001环境步。
CPU32次固定batch约0.70秒，约21.95ms/次；313216次核心粗估约6874秒，
不包含物理采样、DEV、传输或争用，不能当CUDA耗时。

gpu_gate_command.json提供同源纯CUDA门禁及吞吐argv/env，由root在实时准入后
显式执行；本代理未运行GPU。它同时测305/19与244/13、CPU/CUDA数值误差和实际
强制时钟hook，每域32次计时，无Env调用、不存模型。先保eager数学，再据真实CUDA
门禁判断是否需要另行证明CUDA Graph等价；没有静默部署加速候选。

原生wrapper门禁若10000预热、10272总步，则更新数是2176，不是2184。
科学是否实际触发回收、每层是否有非零mask及末段是否退化，必须以真实结果判断。
