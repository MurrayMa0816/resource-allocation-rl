MR.Q：连续资源配置任务接口
============================

来源与实现
----------

来源为 ICLR 2025 论文 Towards General-Purpose Model-Free Reinforcement Learning，
作者为 Scott Fujimoto、Pierluca D'Oro、Amy Zhang、Yuandong Tian、Michael Rabbat。
作者仓库 facebookresearch/MRQ，固定提交
073adcd6b30f84694c241e00bd684db50242a215；原件采用 CC-BY-NC 4.0。
保留的 8 个源码/说明文件按 Git blob 校验，包括原 CRLF。

实现使用原生多步 LAP、5 步表示学习、reward TwoHot、真实终止辅助头、3 步 Q
目标、reward scale、AdamW、LayerNorm 和固定 target 同步周期。表示学习使用
真实观测到的 next-state、reward 和 done，不使用外部离线/专家数据、虚构物理
交互或额外过程奖励。

native_loader 在隔离模块名下加载 MRQ.py、models.py、buffer.py 和 utils.py，
随后恢复临时 bare-import 别名，不导入 main.py 或 Gym/Atari 环境注册。
Observer 记录调用与返回值，网络、损失、优化器和采样函数沿用保留原件。

ObservedReplay 显式将 priority 未写槽置零，以避免 torch.empty 中的 NaN 经
priority*mask 影响全容量 cumsum。初始化不消耗 RNG，有效转移仍由原生 add
写入 max_priority，保持 LAP 分布及采样/更新公式。对照检查对直接作者参考采用
同样的无效槽初值；其结论针对这一已定义初始化，不能等同未经修改的软件栈。

接口与终止处理
--------------

Agent(cfg, obsdim, actiondim, device) 使用连续动作、vector 观测、history=1、
max_action=1。policy_action 为确定性部署策略，act 保持作者预热与探索。
observe 接收真实 (s,a,r,next_s,terminated,truncated=False)，计数按真实交互
累计；固定 H 的末步是真实终止。

原生 buffer 以 not_done=0 和 env_terminates=True 记录终止。5 步表示展开按
此前 not_done 逐步屏蔽，3 步 Q 回报和 discount 在首终点后停止；终点不进行
bootstrap。公开奖励不缩放，内部 reward_scale/target_reward_scale 保留并
单独记录，不是新增即时奖励。

推理终点仅保存 zs_mlp 状态编码、policy 权重、配置与计数。load 不分配回放，
前向对应作者 encoder.zs→policy.act；不保存价值网络、辅助模型或 Adam，不能
精确续训。

配置与设备
----------

batch=256，hidden=512，原生回放容量 1000000，encoder horizon=5，Q horizon=3。
显式 cuda:0 与作者 buffer 中 torch.device('cuda') 的设备分支判断不同，本接口
沿原分支将观测存于 CPU。通信/应急观测分别占 1220000000/976000000 字节，
两组 history 索引共 8000000 字节；GPU 上 action/reward/not_done 分别合计
84000000/60000000 字节，priority/mask 共 8000000 字节。网络、Adam 和 batch
工作区另计，不能据 CPU 检查推断 CUDA 峰值显存。

预热阈值按 replay.size 而非物理步数；每次终止额外占一个观测槽。
select_action 条件为 size<10000，train 条件为 size>10000。通信 H=32 的首学习
在 9698 个真实步，应急 H=24 在 9601 步，首学习时 size 均为 10001。
预热 Box 分布使用私有 seeded RandomState，不保证与旧 Gym 随机流逐位一致；
学习后的 Torch 探索噪声为 .2。

每次 train 包含价值和策略各一步 AdamW；第 1、251、501…次 RL 更新前同步
targets/reward scale，并执行 250 次 encoder AdamW。encoder 更新不增加环境
交互计数。按 10368 个真实步推算，通信为 671/750/3 次 RL/encoder/target 同步，
应急为 768/1000/4；按 49152 步推算，分别为 39455/39500/158 和
39552/39750/159。这些是调度计算值，不代表对应训练已执行。

验证边界
--------

默认 CUDA 检查发现原生重复执行的 replay.priority 部分位置不一致，记录为
FAIL_NATIVE_NONDETERMINISM；网络、targets、Adam、梯度和 RNG 的比较一致。
mrq-priority-index-put-strict-v1 仅在 update_priority 内临时启用严格
Torch deterministic algorithms，并恢复 enabled/warn_only；不改变赋值、
采样、priority 公式、cumsum 或 CUBLAS 配置，不提供全栈确定性保证。
所保留的早期准备记录未提供该 profile 的完整 CUDA PASS 结论，检查范围见
GPU_GATE.rst；论文采用的训练与评价记录以章级模型、结果索引为准。

旧 zero_env_gate.py 的完整默认网络、batch=256 合成结果绑定 priority 修复前
源码，覆盖参数、targets、AdamW、梯度、RNG、3 步 done mask、TwoHot、终点遮罩
和推理重载；实际合成更新为 encoder/value/policy 各 4 次，真实环境交互为零。
另一个容量 11000、batch=8 的时钟 fixture 检查 251 次 RL 调度对应 500 次 encoder
调度，不执行优化器。新 priority 初始化的 CPU 检查仅覆盖无效槽初值、RNG 和
采样一致性；上述结果不代替新版完整 CUDA 或真实环境验证，也不是学习效果证据。
