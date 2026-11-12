MR.Q 独立 CUDA 纯数学门禁
========================

此文件是准备说明，不是 CUDA 验收结果。本次不启动 GPU，不调用真实环境，
不改变已经封闭的四个方法的科学批次。独立补位计划位于
``/hy-tmp/resource_allocation_results/chapter4/mrq_reserve_validation_20261106/study_protocol.json``。

运行入口为本目录 ``gpu_zero_env_gate.py``，精确 argv 与环境保存在
``gpu_gate_command.json``。运行时必须沿用已登记运行环境的 LD_LIBRARY_PATH；
只有原并行训练释放资源后，由主调度者执行。运行前要求物理 GPU UUID 为
``GPU-a3130b13-8288-caaa-0f96-f2916c92d8e3``，逻辑设备为 ``cuda:0``，
空闲显存至少 6GiB。门禁不改变 TF32、cuDNN、AdamW 数学。

默认 CUDA 首次门禁已经完成，结果 FAIL_NATIVE_NONDETERMINISM，原报告完整保留
于独立根 preflight/cuda/。真实执行 1500/12/12 次 encoder/value/policy 更新，
所有网络、targets、Adam、梯度、RNG均逐位一致，只有 replay.priority 的单个
位置出现不一致，直接作者重复也存在该现象。这不能冒称默认 CUDA 已可复现。

第二次采用显式软件执行 profile ``mrq-priority-index-put-strict-v1``：仅在
原生 update_priority 调用期间临时启用严格 torch deterministic algorithms，
finally 恢复原 enabled 和 warn_only 两项状态。三份学习器都使用同一操作范围。
作者 indexed assignment、priority 公式和重复索引规约原样保留，不替换聚合。
全局 strict 在 Torch2.5.1 会拒绝原生浮点 CUDA cumsum，因此不改变 cumsum，
不启用全局 strict，不修改 CUBLAS_WORKSPACE_CONFIG，不作全栈确定性保证。

新的实际 CUDA 检查为四种原标志组合分别执行正常/异常退出，验证 guard 完整
恢复且不消耗 RNG；真实 RL 每次记录重复采样索引、冲突 priority 值和最终写入值。
``native_determinism_not_forced_or_repaired=false`` 明确表示已使用上述控制，
其余七类逐位 exact 验收一项不放宽。第二次完整门禁仍待实际运行。

实际运行内容
------------

两个任务的公开输入尺寸分别为 305/19 与 244/13。每个尺寸均独立建立
适配器、直接作者实现、直接作者重复三份学习器。三者保留作者全部超参数：
256 batch、512 隐层、1000000 原生回放容量、5 步表征展开、3 步价值目标、
10000 回放槽预热以及每 250 次 RL 更新的 250 次表征更新。

全部轨迹均为合成数据，真实环境交互为零。合成数据经过原生 add 与 train
预热，之后每份学习器真实执行第一个 250 次 encoder AdamW burst，随后完成
两次 value/policy AdamW 更新。每个尺寸实际执行 encoder 750、value 6、
policy 6 次；合计为 1500/12/12 次。optimizer post hook 的实际调用计数与
Adam step 状态同时核验，没有用 clock stub 代替神经网络更新。

原生 ReplayBuffer 在每次终止后多占一个终态观测槽，不能将 size/ind
冒充真实交互步数。通信首次 RL 的合成 add 时钟为 9698，应急为 9601。
对应预热结束的 size/ind 分别为 10000/10000 和 9999/10000。
所有这些数值只属于门禁合成数据，不进入科学交互预算。

显式工程修复在 ObservedReplay.__init__ 的 super 后将 priority 置零，作者
vendor 原字节不改。原生 cumsum 对整个容量读取 priority*mask，未写槽的
torch.empty 值若为 NaN，即使 mask=0 仍会污染结果。修复只定义这些无效槽，
真实 transition 的 add 仍按作者规则写入 max_priority，不改变有效槽优先级、
采样公式或学习数学。门禁检查初始化无 RNG 消耗、人工 NaN 尾部反例，以及
有限有效优先级分布和同 RNG 采样结果保持一致；反例不声称原生正常输出必然 NaN。

为定义完全相同的比较输入，直接作者参考也将未写入的 priority 尾部置零。
对照结论因此明确针对这种有定义的初始数据，不冒称未经修改的完整软件栈。
初始 target
参数另加 0.125 合成偏移；原生首更新必须真实硬复制以覆盖偏移，回调逐网络
核对复制值，第二次 RL 不得再次复制或改变 target。

每次实际 RL 后，比较全部六个网络参数/目标、三份 AdamW 状态、六个网络梯度、
CPU/CUDA/NumPy RNG、完整 priority/mask 以及已写入回放数据和索引。比较是按字节
精确比较，失败保留差异位置和幅度。适配器与原生不等且原生重复相等时记为
FAIL_ADAPTER_MISMATCH；原生自己重复不等则记为 FAIL_NATIVE_NONDETERMINISM。
后者不会借 CPU 精确结果改判为 PASS。

额外独立 CUDA 公式检查覆盖终止后屏蔽、3 步回报、5 步真实终止采样、
reward two-hot、LAP priority 与更新赋值。小回放仅用于这些独立公式检查，
不替换三个学习器的完整原生回放。确定性部署、DEV 不改变学习状态/RNG、
仅 state encoder 与 actor 的保存重载也在实际 CUDA 执行。推理临时文件只在
独立 preflight 下创建并立即清除，没有持久模型、回放或优化器文件。

输出与验收边界
--------------

第二次真实门禁输出为独立根的 ``preflight/cuda_attempt02/gate.json``。
旧 preflight/cuda 原件及其源码另已保全，不覆盖。只有全部实际 CUDA
检查通过并且两种比较均精确时，status 才是 PASS。输出包含实际 Torch/CUDA
版本、nvidia-smi 查询的物理 UUID、显存峰值、每个优化器的真实次数、分类对照
哈希、全部相对本目录的源文件 SHA256，以及前后源码不变检查。

``--dry-run`` 仅解析标准库合同，完全不导入 Torch，不接触 GPU，输出必须另放
``preflight/prepared_cuda_gate_attempt02.json``，状态为 PREPARED_NOT_EXECUTED。
此准备状态、旧 CPU gate、源码静态检查均不能代替真实 CUDA PASS。
CUDA PASS 也不代替随后两个真实环境工程预检，更不代表科学训练有效性结论。

priority 修复前的 agent、source_manifest、DELIVERY、PROVENANCE 与 README
原字节和 SHA 保存在独立 ``preflight/priority_initialization_review/before/``。
当前来源清单与交付状态另行更新，旧 CPU 全数学门禁仅证明旧来源；新初始化
专用 CPU 检查也不能替代新版本的完整实际 CUDA 验收。
