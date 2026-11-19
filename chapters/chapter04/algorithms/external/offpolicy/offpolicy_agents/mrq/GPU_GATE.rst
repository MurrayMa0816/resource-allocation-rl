MR.Q：CUDA 合成检查
====================

用途与运行配置
--------------

入口为 gpu_zero_env_gate.py，参数及环境见 gpu_gate_command.json。
已登记配置使用逻辑 cuda:0，绑定物理设备
GPU-a3130b13-8288-caaa-0f96-f2916c92d8e3，要求空闲显存至少 6 GiB，并沿用登记环境
的 LD_LIBRARY_PATH。检查保持 TF32、cuDNN 和 AdamW 数学设置。

检查使用合成数据，不调用真实环境。两个任务尺寸为 305/19 和 244/13，每个
尺寸均建立适配器、直接作者实现及直接作者重复三份学习器。保留 batch=256、
hidden=512、回放容量 1000000、5 步表示展开、3 步 Q 目标、10000 回放槽预热及
每 250 次 RL 更新的 250 次表示更新。

确定性设置与已知限制
--------------------

默认 CUDA 检查记录为 FAIL_NATIVE_NONDETERMINISM：网络、targets、Adam、梯度
和 RNG 逐位一致，replay.priority 部分位置不一致，直接作者重复也出现该问题。
原记录位于 preflight/cuda/，其结果不构成默认 CUDA 可逐位复现的证明。

mrq-priority-index-put-strict-v1 只在 update_priority 期间临时启用严格
Torch deterministic algorithms，finally 恢复原 enabled 和 warn_only 状态。
三份学习器均使用这一范围；indexed assignment、priority 公式和重复索引规约
保持。Torch 2.5.1 的全局 strict 会拒绝浮点 CUDA cumsum，因此不启用全局
strict、不改变 cumsum 或 CUBLAS_WORKSPACE_CONFIG，也不保证全栈确定性。
所保留的早期准备记录未提供该 profile 的完整 CUDA PASS 结论；论文采用的
训练与评价记录以章级模型、结果索引为准。

检查内容
--------

每份学习器完成首个 250 次 encoder AdamW burst 和两次 value/policy 更新。
每个尺寸三份学习器合计 encoder/value/policy 为 750/6/6 次，两域为 1500/12/12；
以 optimizer post hook 和 Adam step 同时计数。首次 RL 的合成 add 时钟为通信
9698、应急 9601，对应预热末的 size/ind 为 10000/10000 和 9999/10000。
这些都是合成检查计数，不计入真实环境预算。

ObservedReplay 在 super 后将未写入 priority 置零。直接作者参考采用相同初值，
避免 torch.empty 中的 NaN 经 priority*mask 污染 cumsum；有效转移仍由原生 add
写入 max_priority。检查覆盖无 RNG 消耗、人工 NaN 尾部反例、有限优先级分布和
同 RNG 的采样一致性；反例不表示原生输出必然为 NaN。

初始 target 参数加 .125 合成偏移，核对首更新的硬复制及第二次更新不重复复制。
每次 RL 后逐字节比较六个网络参数/目标、三份 AdamW 状态、六组梯度、
CPU/CUDA/NumPy RNG、完整 priority/mask 及已写入回放的数据和索引。适配器不同
而原生重复一致时记录 FAIL_ADAPTER_MISMATCH；原生自己重复不同时记录
FAIL_NATIVE_NONDETERMINISM。

额外检查覆盖终止屏蔽、3 步回报、5 步终止采样、reward TwoHot、LAP priority、
更新赋值、确定性部署、评价的学习状态/RNG 隔离及 encoder/actor 保存重载。
小回放只用于独立公式检查，不替代三份学习器的完整回放。strict guard 需覆盖
四种原标志组合下的正常和异常退出，核对标志恢复且不消耗 RNG；每次 RL 记录
重复索引、冲突 priority 及最终写入值。结果中的
native_determinism_not_forced_or_repaired=false 表明使用了上述控制。

输出与结论范围
--------------

strict profile 的输出约定为 preflight/cuda_attempt02/gate.json，包含实际
Torch/CUDA 版本、GPU UUID、峰值显存、优化器计数、分类比较摘要、源码 SHA256
和运行前后源码比较。只有全部 CUDA 检查及两组逐位对照均通过时，status 才为 PASS。

--dry-run 只解析标准库配置，不导入 Torch 或使用 GPU，输出状态为
PREPARED_NOT_EXECUTED；约定输出为 preflight/prepared_cuda_gate_attempt02.json。
该状态、旧 CPU 结果和静态源码检查不能替代实际 CUDA PASS。CUDA 合成检查通过
也不能替代真实环境预检或证明科学训练有效。
