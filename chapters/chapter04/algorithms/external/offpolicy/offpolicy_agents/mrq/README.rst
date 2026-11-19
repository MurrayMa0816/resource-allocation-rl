MR.Q：ICLR2025作者核心预备适配
=============================

范围与身份
----------
Towards General-Purpose Model-Free Reinforcement Learning，ICLR2025。
Scott Fujimoto、Pierluca D'Oro、Amy Zhang、Yuandong Tian、Michael Rabbat。
官方facebookresearch/MRQ，固定commit073adcd6b30f84694c241e00bd684db50242a215。
原件CC-BY-NC4.0许可证及8个源码/说明文件保留，逐Git blob核验，CRLF亦保持。
本目录是备用代码；没有加入当前已封闭的四候选科学计划，未运行GPU/真实环境。

MR.Q与TD7有作者和表示学习思想沿承，但为不同论文/完整方法。这里不能只在TD7上
贴MR.Q标签：保留原生多步LAP、5步模型表示学习、奖励TwoHot、真实终止辅助头、
3步Q目标、reward scale、AdamW、LayerNorm与固定target同步周期，未换成TD7核心。
它使用观测到的next-state/reward/done来学习表示，不进行真环境可微求导、生成
虚构物理交互或引入额外过程奖励。无需离线专家数据。

原件直接复用与接口
------------------
native_loader仅在隔离模块名下加载MRQ.py、models.py、buffer.py和utils.py；临时
bare-import别名随后恢复，不导入main.py、Gym/Atari注册或其他任务环境。Observer
子类记录调用与已有返回值；源网络、损失、优化器和采样函数原字节不改。
另有一处显式工程修复：ObservedReplay在super后将priority初值置零。作者
torch.empty未写槽会参与全容量priority*mask，即使mask=0也不能消除NaN。
初始化不消耗RNG；任何有效transition仍由原生add写入max_priority，LAP分布、
采样公式和更新规则保持。门禁对直接原生参考定义同样的无效槽初值，不冒称
未经修改的全软件栈；vendor仍逐字节保持。修复前原件在独立preflight/before保留。

默认CUDA实际门禁发现：网络、targets、Adam、梯度和RNG精确一致，但原生重复
执行的priority少数位置不一致，与重复sampled_ind在GPU原地赋值竞态一致。
现显式采用mrq-priority-index-put-strict-v1软件执行profile，仅包住原生
update_priority临时严格deterministic，finally恢复enabled/warn_only。赋值、
采样和优先级公式不改，不改变cumsum或CUBLAS环境，不宣称默认CUDA全栈确定。
第二次门禁记录真实重复索引/冲突值，保留全部七项精确对照；第一次失败记录保留。

Agent(cfg,obsdim,actiondim,device)使用连续动作、vector观测、history1、max_action1。
policy_action始终调用确定性native deployment策略；act保持作者预热/探索逻辑。
observe仅传真实(s,a,r,next_s,terminated=True/False,truncated=False)，步骤按真实
交互累计。H末步是本任务真实终止，不能假称continuing-MDP时间截断。

推理终点仅包含部署所需zs_mlp状态编码和policy两组权重、配置及计数，没有价值
网络、奖励/终止辅助模型、Adam、磁盘回放或周期断点。load不会分配原生回放，
推理前向与作者encoder.zs→policy.act逐位核验；不能精确续训。

原生容量与设备
--------------
容量保持1000000。显式cuda:0与原buffer判断的torch.device('cuda')不相等，故本
接口沿原代码选择CPU存观测，不改设备分支。通信CPU观测1,220,000,000字节，
应急976,000,000字节；两个history索引再共8,000,000字节。GPU上的action/reward/
not_done分别84,000,000或60,000,000字节，加priority与mask共8,000,000字节，
网络、Adam及batch工作区另计。24GB GPU可用，但正式准入仍由root实时核算。

原buffer未带索引的cuda分支对vector obs预估有其原生写法；本接口不进入该分支，
不能把显式cuda:0配置的CPU观测记录成全部GPU回放，也不凭CPU门禁估计CUDA峰值。

10k门槛的真实含义
-----------------
作者门槛是replay.size，不是物理步：每个终末另占一个观测槽。必须分开记真实
交互与槽位，不能将该额外槽计算成免费/新增交互，也不能静默改成10001物理步。

在H32通信中首train为9698真实步；H24应急为9601。两者CPU原生时钟fixture已
核验，首学习时replay.size均10001。select_action预热条件为size<10000，train
条件为size>10000，两种比较亦原样保留。预热Box分布由私有seeded RandomState
实现；不声称旧Gym随机流逐位相同。真正学习后探索噪声仍用原Torch .2。

每个train调用有一次价值和一次策略AdamW；每第1、251、501…次RL更新先同步
targets/reward scale并执行250次真实encoder AdamW（enc_horizon5）。不能把这
250次稠密计算说成250个新环境步，也不能只运行一次encoder而冒称原方法。

如果以后独立登记10368步真实工程：
通信671次RL、750次encoder、3次target同步；应急768/1000/4。
若49152步科学：通信39455次RL、39500次encoder、158次同步；应急39552/39750/159。
当前只是预估与CPU时钟核验，没有启动这两个新任务。

终止与多步表示
--------------
原buffer按真实terminated保存not_done=0，并记录env_terminates=True，让done辅助头
参与训练。enc_horizon5的未来损失按此前not_done逐步遮罩，Q_horizon3的回报和
discount在首终末后均停止。跨终末槽的后续数据不会成为真实终末bootstrap目标。
公开任务reward不缩放；原生reward_scale/target_reward_scale作为学习内部归一化
保留并单列日志，不能把它们写成新增即时奖励。

已完成的CPU准备门禁
------------------
下述旧zero_env_gate.py结果绑定修复前来源，不追溯认证新priority初始化版本。
它始终CPU且隐藏CUDA。两域完整默认网络、batch256，适配与直接
作者core的参数/targets/AdamW/梯度/RNG精确一致；独立3步done mask、TwoHot、
done后动态遮罩及推理无副作用/重载通过。实际合成优化器步为encoder4、value4、
policy4；0环境、0持久模型。

时钟另用明确不执行优化器的native调度fixture（较小容量11000、batch8），核验
251个RL调度调用对应500个encoder调度调用。它不是实际执行了500次神经更新。
CPU准备不能替代正式GPU完整250步burst或真实环境门禁，也不是学习效果证据。
新priority初始化的独立CPU检查只验证定义无效槽、零RNG和采样一致性，不执行
encoder/RL优化器。新scoped profile CUDA完整门禁见GPU_GATE.rst，尚未实际执行；
默认profile第一次CUDA失败不改判为通过。主调度者已在
mrq_reserve_validation_20261106独立预注册补位计划，与四个封闭科学cohort分开；
只有实际CUDA和两个真实工程门禁通过才可运行该补位科学任务。
